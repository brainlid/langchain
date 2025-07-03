# Original sources of inspiration:
# - https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/
# - https://github.com/langchain-ai/langchainjs/blob/0.3.6/langchain/src/memory/summary.ts#L14 - prompt used for memory summarization
# - https://github.com/langchain-ai/langchainjs/blob/0.3.6/langchain/src/memory/prompt.ts#L3 - default summary prompt
defmodule LangChain.Chains.SummarizeConversationChain do
  @moduledoc """
  When an AI conversation has many back-and-forth messages (from user to
  assistant to user to assistant, etc.), the number of messages and the total
  token count can be large. Large token counts present the following problems:

  - Increased cost/price per generation
  - Increased generation times
  - Risk of exceeding the total token limit, resulting in an error

  This chain is run as a separate process to summarize and condense a separate
  conversation chain. It is assumed that the chain the user sees in the UI
  retains all their original messages and they are not seeing the full, raw
  message list.

  We don't want to perform more work than we need to, so we'll only kick off the
  summary process once the number of messages has reached some threshold, then
  we'll retain a configured number of the most recent messages to help retain
  continuity for the conversation.

  ## Options

  - `:llm` - The LLM to use for performing the summarization. There is no need for streaming.
  - `:keep_count` - The number of raw messages to retain. It will be the
    most recent messages and defaults to 2 (a user and assistant message).
  - `:threshold_count` - The total number of messages (excluding the system
    message) that must be present before the summarizing operation is performed.
    Running the summarization on a short conversation chain will return the
    chain unchanged and not make any calls to an LLM.
  - `:override_system_prompt` - When the system prompt should be customized for the instructions on how to summarize, this can be used to provide a customized replacement of the system prompt.
  - `:messages` - When explicit control of multiple messages is needed, they can be provided as a list. They can be `LangChain.PromptTemplate`s and the concatenated list of messages will be in the `@conversation` param. When this is used, any value in  `:override_system_prompt` is ignored.

  ## Examples
  A basic example that processes the messages in a separate LLMChain, returning
  an updated chain with summarized contents.


      {:ok, summarized_chain} =
        %{
          llm: ChatOpenAI.new!(%{model: "gpt-4o-mini", stream: false}),
          keep_count: 2,
          threshold_count: 6
        }
        |> SummarizeConversationChain.new!()
        |> SummarizeConversationChain.summarize(chain_to_summarize)

  Using a `:with_fallback` option to still try and summarize if the LLM errors from the Azure host OpenAI.

      # Azure configured OpenAI LLM
      fallback_llm =
        ChatOpenAI.new!(%{
          stream: false,
          endpoint: System.fetch_env!("AZURE_OPENAI_ENDPOINT"),
          api_key: System.fetch_env!("AZURE_OPENAI_KEY")
        })

      {:ok, summarized_chain} =
        %{
          llm: ChatOpenAI.new!(%{model: "gpt-4o-mini", stream: false}),
          keep_count: 2,
          threshold_count: 6
        }
        |> SummarizeConversationChain.new!()
        |> SummarizeConversationChain.evaluate(chain_to_summarize, with_fallbacks: [fallback_llm])

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.Message.ContentPart
  alias LangChain.PromptTemplate
  alias __MODULE__
  alias LangChain.Chains.LLMChain
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.Utils.ChainResult
  alias LangChain.Message.ContentPart

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :keep_count, :integer, default: 2
    field :threshold_count, :integer, default: 6
    field :override_system_prompt, :string
    field :messages, {:array, :any}, virtual: true
    field :verbose, :boolean, default: false
  end

  @type t :: %SummarizeConversationChain{}

  @create_fields [
    :llm,
    :keep_count,
    :threshold_count,
    :override_system_prompt,
    :messages,
    :verbose
  ]
  @required_fields [:llm, :keep_count, :threshold_count]

  @default_system_prompt ~s|You expertly summarize a conversation into concise bullet points that capture significant details and sentiment for future reference. Summarize the conversation starting with the initial user message. Return only the summary with no additional commentary.

<example>
- User initiated travel planning for a 2-week Italy trip in September
- Destinations chosen: Rome (4 days), Florence (4 days), and Amalfi Coast (5 days)
- Transportation advice: Trains for city travel, car rental option for Amalfi Coast
- Hotel budget set at $200/night with specific recommendations provided for each location:
  * Rome: Hotel De Russie
  * Florence: Hotel Davanzati
  * Amalfi: Hotel Marina Riviera
- Specific travel dates: September 10-24
- Must-see attractions discussed for each location with emphasis on Vatican and Uffizi Gallery
- Discussion of advance booking strategies for popular attractions
- User showed particular interest in skip-the-line tickets for Vatican and Uffizi
- Booking recommendation: 2-3 months advance booking for major attractions
- Conversation concluded with offer of guided tour options and official booking resources
</example>|

  @doc """
  Start a new SummarizeConversationChain configuration.

      {:ok, summarizer} = SummarizeConversationChain.new(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        keep_count: 2,
        threshold_count: 6
      })
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %SummarizeConversationChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Start a new SummarizeConversationChain and return it or raise an error if invalid.

      chain = SummarizeConversationChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        keep_count: 2,
        threshold_count: 6
      })
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, changeset} ->
        raise LangChainError, changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> validate_number(:keep_count, greater_than_or_equal_to: 0)
    |> validate_number(:threshold_count, greater_than_or_equal_to: 2)
    |> Utils.validate_llm_is_struct()
  end

  @doc """
  Run a SummarizeConversationChain to summarize a text representation of a sequence of user and assistant messages.

      new_title = SummarizeConversationChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: false},
        keep_count: 2,
        threshold_count: 6
      })
      |> SummarizeConversationChain.run(text_to_summarize)

  """
  @spec run(t(), String.t(), opts :: Keyword.t()) ::
          {:ok, LLMChain.t()} | {:error, LLMChain.t(), LangChainError.t()}
  def run(%SummarizeConversationChain{} = summarizer, text_to_summarize, opts \\ []) do
    messages =
      summarizer
      |> get_messages()
      |> PromptTemplate.to_messages!(%{
        conversation: text_to_summarize
      })

    %{llm: summarizer.llm, stream: false, verbose: summarizer.verbose}
    |> LLMChain.new!()
    |> LLMChain.add_messages(messages)
    |> LLMChain.run(opts)
  end

  @doc """
  Summarize the `to_summarize` LLMChain using the
  `%SummarizeConversationChain{}` configuration and `opts`. Returns a new,
  potentially modified `LLMChain` after completing the summarization process.

  If the `threshold_count` is greater than the current number of summarizable
  messages (ie user and assistant roles), then nothing is modified and the
  original LLMChain is returned.

  ## Options

  - `:with_fallbacks` - An optional set of fallback LLMs to use if the
    summarization process fails. See `LangChain.Chains.LLMChain.run/2` for
    details.
  """
  @spec summarize(t(), LLMChain.t(), opts :: Keyword.t()) :: LLMChain.t()
  def summarize(
        %SummarizeConversationChain{} = summarizer,
        %LLMChain{} = to_summarize,
        opts \\ []
      ) do
    text_to_summarize = combine_messages_for_summary_text(summarizer, to_summarize)

    # if there is something to summarize, do it
    if text_to_summarize do
      summarizer
      |> run(text_to_summarize, opts)
      |> ChainResult.to_string()
      |> case do
        {:ok, summary} ->
          Logger.debug("SummarizeConversationChain completed summarization")
          if summarizer.verbose, do: IO.inspect(summary, label: "SUMMARY GENERATED")

          # splice the messages together. Keep the system, insert modified user
          # and assistant messages, keep the # of messages to keep.
          splice_messages_with_summary(summarizer, to_summarize, summary)

        {:error, _summarizer_chain, reason} ->
          Logger.error(
            "SummarizeConversationChain failed. Reason: #{inspect(reason)}. Returning original chain"
          )

          to_summarize
      end
    else
      # nothing to summarize, return the original chain
      to_summarize
    end
  end

  @doc """
  Create a single text message to represent the current set of messages being
  summarized from the `to_summarize` chain. Uses the settings from SummarizeConversationChain.
  A `nil` is returned when the threshold has not been reached for running the
  summary procedure.

  This combines the user and assistant messages into a single string that can be summarized. This does not summarize a `system` message and does not include the last `n` messages for the `keep_count`.
  """
  @spec combine_messages_for_summary_text(t(), LLMChain.t()) :: nil | String.t()
  def combine_messages_for_summary_text(
        %SummarizeConversationChain{} = summarizer,
        %LLMChain{} = to_summarize
      ) do
    {_system, messages} = Utils.split_system_message(to_summarize.messages)

    messages_count = length(messages)

    # if the summarize process is needed, combine the messages into a single text string
    cond do
      summarizer.keep_count >= messages_count ->
        # keep all the messages we have, nothing to summarize
        nil

      summarizer.threshold_count > messages_count ->
        # haven't reached the threshold
        nil

      summarizer.threshold_count <= messages_count ->
        # get the messages that are part of the summary
        summarize_messages = Enum.take(messages, messages_count - summarizer.keep_count)

        # generate the combined text output
        summarize_messages
        |> Enum.map(&for_summary_text(&1))
        |> Enum.join("\n")

      true ->
        # anything else
        nil
    end
  end

  # Create the user and assistant messages that contain the summarized text. If no text was summarized, returns an empty list.
  @doc false
  @spec create_summary_messages(nil | String.t()) :: [Message.t()]
  def create_summary_messages(nil), do: []

  def create_summary_messages(summary_text) when is_binary(summary_text) do
    [
      Message.new_user!(
        "Summarize our entire conversation up to this point for future reference."
      ),
      Message.new_assistant!(summary_text)
    ]
  end

  # Splice the summarized text into a compressed version of the messages. Keep
  # the system message and the desired keep count. Keep the user + AI message
  # count valid.
  @doc false
  @spec splice_messages_with_summary(t(), LLMChain.t(), nil | String.t()) :: LLMChain.t()
  def splice_messages_with_summary(
        %SummarizeConversationChain{} = summarizer,
        %LLMChain{} = to_summarize,
        summary_text
      ) do
    # Extract the first element of the larger list
    {system_message, rest} = Utils.split_system_message(to_summarize.messages)

    # Extract the ending items to keep
    keeping_items = Enum.take(rest, -summarizer.keep_count)

    summary_messages = create_summary_messages(summary_text)

    # add the system message to the front if it has a system message
    starting_messages =
      case system_message do
        nil ->
          summary_messages

        %Message{} = system ->
          [system | summary_messages]
      end

    new_messages = starting_messages ++ keeping_items

    %LLMChain{
      to_summarize
      | messages: new_messages,
        last_message: List.last(new_messages)
    }
  end

  # Convert each `%Message{}` into a text message like `<user>The user message
  # contents.</user>` so they can be combined in a single message to be
  # summarized.
  @doc false
  def for_summary_text(%Message{role: :tool, tool_results: tool_results} = _message)
      when is_list(tool_results) do
    # When summarizing a tool result, provide the string version of what was returned.
    text_results =
      tool_results
      |> Enum.map(fn %ToolResult{} = result ->
        if result.is_error do
          "- Tool '#{result.name}' ERRORED: " <> ContentPart.content_to_string(result.content)
        else
          "- Tool '#{result.name}' SUCCEEDED: " <> ContentPart.content_to_string(result.content)
        end
      end)
      |> Enum.join("\n")
      |> remove_xml_tags()

    "<tool>Tool results\n#{text_results}</tool>"
  end

  # Handle ContentPart messages
  def for_summary_text(%Message{role: role, content: content_parts} = _message)
      when is_list(content_parts) do
    tag =
      case role do
        :user ->
          "user"

        :assistant ->
          "AI"
      end

    parts_text =
      content_parts
      |> Enum.map(fn %ContentPart{} = part ->
        case part.type do
          :text ->
            "#{part.content}"

          :image_url ->
            "(Omitted URL to an image)"

          :image ->
            "(Omitted image data)"

          _other ->
            ""
        end
      end)
      |> Enum.join("\n")
      |> remove_xml_tags()

    "<#{tag}>#{parts_text}</#{tag}>"
  end

  def for_summary_text(%Message{role: :assistant, tool_calls: calls} = _message)
      when is_list(calls) do
    # When summarizing an assistant message with one or more tool calls, omit the parameters.
    text_calls =
      calls
      |> Enum.map(fn %ToolCall{} = call ->
        "- Tool '#{call.name}' called"
      end)
      |> Enum.join("\n")
      |> remove_xml_tags()

    "<AI>Using tools\n#{text_calls}</AI>"
  end

  # Remove conflicting XML tags from the message contents.
  defp remove_xml_tags(content) do
    content
    |> String.replace("<user>", "")
    |> String.replace("</user>", "")
    |> String.replace("<AI>", "")
    |> String.replace("</AI>", "")
    |> String.replace("<tool>", "")
    |> String.replace("</tool>", "")
  end

  # Build the messages when not explicitly provided.
  defp get_messages(%SummarizeConversationChain{messages: nil} = summarizer) do
    system_prompt = summarizer.override_system_prompt || @default_system_prompt

    [
      Message.new_system!(system_prompt),
      PromptTemplate.new!(%{
        role: :user,
        text: "<%= @conversation %>"
      })
    ]
  end

  # Return any explicitly provided messages.
  defp get_messages(%SummarizeConversationChain{messages: messages}) when is_list(messages) do
    messages
  end
end
