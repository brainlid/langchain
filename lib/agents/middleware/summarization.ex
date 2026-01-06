defmodule LangChain.Agents.Middleware.Summarization do
  @moduledoc """
  Middleware that automatically manages conversation length through intelligent summarization.

  This middleware monitors token usage and automatically summarizes older messages when
  a threshold is exceeded, preserving recent messages for context continuity.

  ## Purpose

  Long conversations present several problems:
  - Increased API costs
  - Slower response times
  - Risk of exceeding model context limits
  - Potential API errors

  This middleware solves these problems by:
  - Monitoring total token count
  - Summarizing older messages when threshold is exceeded
  - Preserving recent messages for continuity
  - Protecting AI/Tool message pairs from separation

  ## Configuration

      # Default configuration
      {Summarization, []}

      # Custom configuration
      {Summarization, [
        model: custom_model,                    # Model for summarization (defaults to agent model)
        max_tokens_before_summary: 170_000,    # Token threshold (default: 170k)
        messages_to_keep: 6,                   # Recent messages to preserve (default: 6)
        summary_prompt: custom_prompt,         # Custom summarization prompt
        token_counter: &custom_counter/1       # Custom token counting function
      ]}

  ## Configuration Options

  - `:model` - LLM to use for summarization. Defaults to the agent's model.
  - `:max_tokens_before_summary` - Token threshold that triggers summarization. Default: 170,000
  - `:messages_to_keep` - Number of recent messages to preserve intact. Default: 6
  - `:summary_prompt` - Custom prompt for summarization. Uses intelligent default.
  - `:token_counter` - Function to count tokens. Defaults to approximate counting.

  ## Position in Middleware Stack

  Should run relatively early in before_model phase, after message generation
  but before any processing that expects specific message structures:

  1. TodoListMiddleware
  2. FilesystemMiddleware
  3. SubAgentMiddleware
  4. **SummarizationMiddleware** ← Position
  5. AnthropicPromptCachingMiddleware
  6. PatchToolCallsMiddleware
  7. HumanInTheLoopMiddleware

  ## How It Works

  ### 1. Token Monitoring
  Before each model call, counts total tokens in message history.

  ### 2. Threshold Check
  If tokens exceed threshold, triggers summarization.

  ### 3. Safe Cutoff Detection
  Finds safe points to cut the conversation that don't separate:
  - Assistant messages with tool_calls from their corresponding tool results
  - Related message pairs

  ### 4. Message Partitioning
  - **To summarize**: Older messages before cutoff point
  - **To preserve**: Recent messages after cutoff point

  ### 5. Summary Generation
  Uses LLM to generate concise summary of older messages.

  ### 6. State Update
  Replaces older messages with summary messages, preserving recent messages.

  ## Example

      # Create agent with summarization
      {:ok, agent} = Agent.new(
        model: model,
        middleware: [
          {Summarization, [
            max_tokens_before_summary: 150_000,
            messages_to_keep: 8
          ]}
        ]
      )

      # Summarization happens automatically during execution
      {:ok, state} = Agent.execute(agent, state)

  ## Safe Cutoff Algorithm

  The middleware protects AI/Tool message pairs from separation:

  1. Calculate target cutoff: `message_count - messages_to_keep`
  2. Search backwards from target to find safe cutoff point
  3. A point is safe if:
     - It's not an assistant message with tool_calls
     - The next message isn't a tool result for this assistant
  4. If no safe point found, summarize nothing (keeps all messages)

  ## Error Handling

  - Falls back to keeping all messages if summarization fails
  - Logs errors but doesn't halt agent execution
  - Graceful degradation ensures agent continues working

  ## Performance Considerations

  - Token counting is approximate (fast estimation)
  - Summarization only runs when threshold exceeded
  - Summary generation is async-compatible
  - Minimal overhead when under threshold
  """

  @behaviour LangChain.Agents.Middleware

  alias LangChain.Agents.State
  alias LangChain.Agents.AgentServer
  alias LangChain.Message
  alias LangChain.Chains.SummarizeConversationChain
  alias LangChain.Utils
  alias LangChain.Utils.ChainResult

  require Logger

  @default_max_tokens 170_000
  @default_messages_to_keep 6
  @search_range_for_tool_pairs 5

  @default_summary_prompt """
  You are a Context Extraction Assistant. Your objective is to extract the highest quality and most relevant context from the conversation history below.

  You're nearing the total number of input tokens you can accept, so you must extract the most important information from the conversation history. This extracted context will replace the conversation history presented below.

  **Instructions:**
  - Extract and record all of the most important context from the conversation history
  - Focus on information relevant to the overall goal
  - Ensure you don't repeat actions already completed
  - Be concise but comprehensive
  - Preserve key decisions, outcomes, and state information
  - Note any tools used and their results

  Respond ONLY with the extracted context. Do not include any additional commentary.

  **Conversation to summarize:**
  <%= @conversation %>
  """

  @impl true
  def init(opts) do
    config = %{
      model: Keyword.get(opts, :model),
      max_tokens_before_summary:
        Keyword.get(opts, :max_tokens_before_summary, @default_max_tokens),
      messages_to_keep: Keyword.get(opts, :messages_to_keep, @default_messages_to_keep),
      summary_prompt: Keyword.get(opts, :summary_prompt, @default_summary_prompt),
      token_counter: Keyword.get(opts, :token_counter, &count_tokens_approximately/1)
    }

    {:ok, config}
  end

  @impl true
  def before_model(%State{messages: messages} = state, config) when is_list(messages) do
    # Access the token counter directly from config
    token_counter = config.token_counter
    total_tokens = token_counter.(messages)

    if total_tokens >= config.max_tokens_before_summary do
      Logger.debug(
        "SummarizationMiddleware: Token threshold exceeded (#{total_tokens} >= #{config.max_tokens_before_summary}). Starting summarization..."
      )

      # Broadcast debug event: summarization starting
      AgentServer.publish_debug_event_from(
        state.agent_id,
        {:middleware_action, __MODULE__,
         {:summarization_started,
          "#{total_tokens} tokens (threshold: #{config.max_tokens_before_summary})"}}
      )

      case summarize_messages(messages, state, config) do
        {:ok, updated_messages} ->
          Logger.info(
            "SummarizationMiddleware: Summarization completed. Reduced from #{length(messages)} to #{length(updated_messages)} messages"
          )

          # Broadcast debug event: summarization completed
          AgentServer.publish_debug_event_from(
            state.agent_id,
            {:middleware_action, __MODULE__,
             {:summarization_completed,
              "#{length(messages)} → #{length(updated_messages)} messages"}}
          )

          {:ok, %{state | messages: updated_messages}}

        {:error, reason} ->
          Logger.error(
            "SummarizationMiddleware: Summarization failed: #{inspect(reason)}. Keeping all messages."
          )

          # Broadcast debug event: summarization failed
          AgentServer.publish_debug_event_from(
            state.agent_id,
            {:middleware_action, __MODULE__, {:summarization_failed, inspect(reason)}}
          )

          {:ok, state}
      end
    else
      {:ok, state}
    end
  end

  def before_model(%State{} = state, _config) do
    {:ok, state}
  end

  # Private functions

  defp summarize_messages(messages, state, config) do
    # Split system message from rest
    {system_message, rest} = Utils.split_system_message(messages)

    # Find safe cutoff point
    cutoff_index = find_safe_cutoff(rest, config.messages_to_keep)

    if cutoff_index == 0 do
      # No safe cutoff found, keep all messages
      Logger.warning("SummarizationMiddleware: No safe cutoff point found. Keeping all messages.")

      {:ok, messages}
    else
      # Split messages into those to summarize and those to keep
      messages_to_summarize = Enum.take(rest, cutoff_index)
      messages_to_keep = Enum.drop(rest, cutoff_index)

      # Generate summary
      case generate_summary(messages_to_summarize, state, config) do
        {:ok, summary_text} ->
          # Create summary messages
          summary_messages = create_summary_messages(summary_text)

          # Rebuild message list
          rebuilt_messages =
            case system_message do
              nil -> summary_messages ++ messages_to_keep
              %Message{} -> [system_message | summary_messages] ++ messages_to_keep
            end

          {:ok, rebuilt_messages}

        {:error, _reason} = error ->
          error
      end
    end
  end

  defp find_safe_cutoff(messages, messages_to_keep) do
    messages_count = length(messages)
    target_cutoff = messages_count - messages_to_keep

    if target_cutoff <= 0 do
      0
    else
      # Search backwards from target to find safe cutoff
      Enum.find_value(target_cutoff..0//-1, 0, fn index ->
        if is_safe_cutoff_point?(messages, index) do
          index
        else
          nil
        end
      end)
    end
  end

  defp is_safe_cutoff_point?(messages, index) do
    if index <= 0 or index >= length(messages) do
      true
    else
      # Get the message at the cutoff point (last message to include in summary)
      message_at_cutoff = Enum.at(messages, index - 1)

      # Check if this is an assistant message with tool calls
      case message_at_cutoff do
        %Message{role: :assistant, tool_calls: tool_calls}
        when is_list(tool_calls) and tool_calls != [] ->
          # This is an assistant with tool calls
          # Check if the next few messages contain tool results
          has_pending_tool_results?(messages, index, @search_range_for_tool_pairs)

        _ ->
          # Not an assistant with tool calls, safe to cut here
          true
      end
    end
  end

  defp has_pending_tool_results?(messages, start_index, search_range) do
    # Check if any of the next N messages are tool results
    messages
    |> Enum.slice(start_index, search_range)
    |> Enum.any?(fn
      %Message{role: :tool} -> true
      _ -> false
    end)
  end

  defp generate_summary(messages_to_summarize, state, config) do
    # Convert messages to text format for summarization
    conversation_text = messages_to_text(messages_to_summarize)

    # Determine which model to use for summarization
    model = config.model || get_model_from_state(state)

    if model do
      # Create summarizer chain
      summarizer =
        SummarizeConversationChain.new!(%{
          llm: model,
          keep_count: 0,
          threshold_count: 0,
          override_system_prompt: config.summary_prompt
        })

      # Run summarization
      case SummarizeConversationChain.run(summarizer, conversation_text) do
        {:ok, chain} ->
          # Extract summary text from chain using standard utility
          ChainResult.to_string(chain)

        {:error, _chain, reason} ->
          {:error, reason}
      end
    else
      {:error, "No model available for summarization"}
    end
  end

  defp get_model_from_state(%State{metadata: metadata}) do
    # Try to get model from state metadata
    # This would be set by the agent when creating the state
    Map.get(metadata, :model) || Map.get(metadata, "model")
  end

  defp get_model_from_state(_), do: nil

  defp messages_to_text(messages) do
    messages
    |> Enum.map(&SummarizeConversationChain.for_summary_text/1)
    |> Enum.join("\n")
  end

  defp create_summary_messages(summary_text) do
    [
      Message.new_user!("Summarize our conversation up to this point for future reference."),
      Message.new_assistant!(summary_text)
    ]
  end

  # Approximate token counting This is a fast estimation based on word count and
  # character count More accurate than character count alone, faster than
  # tokenizing the data directly
  defp count_tokens_approximately(messages) when is_list(messages) do
    messages
    |> Enum.reduce(0, fn message, acc ->
      acc + count_message_tokens(message)
    end)
  end

  defp count_message_tokens(%Message{} = message) do
    # Base tokens for message structure
    base_tokens = 4

    # Count content tokens
    content_tokens =
      cond do
        is_list(message.content) ->
          Enum.reduce(message.content, 0, fn part, acc ->
            acc + count_content_part_tokens(part)
          end)

        is_binary(message.content) ->
          estimate_text_tokens(message.content)

        true ->
          0
      end

    # Count tool call tokens
    tool_call_tokens =
      if message.tool_calls && message.tool_calls != [] do
        Enum.reduce(message.tool_calls, 0, fn call, acc ->
          # Tool call structure + name + arguments
          name_tokens = estimate_text_tokens(call.name || "")
          args_tokens = estimate_text_tokens(Jason.encode!(call.arguments || %{}))
          acc + 10 + name_tokens + args_tokens
        end)
      else
        0
      end

    # Count tool result tokens
    tool_result_tokens =
      if message.tool_results && message.tool_results != [] do
        Enum.reduce(message.tool_results, 0, fn result, acc ->
          content_tokens =
            cond do
              is_list(result.content) ->
                Enum.reduce(result.content, 0, fn part, part_acc ->
                  part_acc + count_content_part_tokens(part)
                end)

              is_binary(result.content) ->
                estimate_text_tokens(result.content)

              true ->
                0
            end

          acc + 10 + content_tokens
        end)
      else
        0
      end

    base_tokens + content_tokens + tool_call_tokens + tool_result_tokens
  end

  defp count_content_part_tokens(%{type: :text, content: text}) when is_binary(text) do
    estimate_text_tokens(text)
  end

  defp count_content_part_tokens(%{type: :image}), do: 1000
  defp count_content_part_tokens(%{type: :image_url}), do: 1000
  defp count_content_part_tokens(_), do: 10

  # Rough estimation: ~1.3 tokens per word, 4 chars per token on average
  defp estimate_text_tokens(text) when is_binary(text) do
    word_count = text |> String.split() |> length()
    char_count = String.length(text)

    # Use average of two estimation methods
    word_based = ceil(word_count * 1.3)
    char_based = ceil(char_count / 4)

    div(word_based + char_based, 2)
  end

  defp estimate_text_tokens(_), do: 0
end
