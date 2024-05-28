defmodule LangChain.Chains.LLMChain do
  @doc """
  Define an LLMChain. This is the heart of the LangChain library.

  The chain deals with tools, a tool map, delta tracking, last_message tracking,
  conversation messages, and verbose logging. This helps by separating these
  responsibilities from the LLM making it easier to support additional LLMs
  because the focus is on communication and formats instead of all the extra
  logic.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.PromptTemplate
  alias __MODULE__
  alias LangChain.Message
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.LangChainError
  alias LangChain.Utils

  @primary_key false
  embedded_schema do
    field :llm, :any, virtual: true
    field :verbose, :boolean, default: false
    # verbosely log each delta message.
    field :verbose_deltas, :boolean, default: false
    field :tools, {:array, :any}, default: [], virtual: true
    # set and managed privately through tools
    field :_tool_map, :map, default: %{}, virtual: true

    # List of `Message` structs for creating the conversation with the LLM.
    field :messages, {:array, :any}, default: [], virtual: true

    # Custom context data made available to tools when executed.
    # Could include information like account ID, user data, etc.
    field :custom_context, :any, virtual: true

    # Track the current merged `%MessageDelta{}` struct received when streamed.
    # Set to `nil` when there is no current delta being tracked. This happens
    # when the final delta is received that completes the message. At that point,
    # the delta is converted to a message and the delta is set to nil.
    field :delta, :any, virtual: true
    # Track the last `%Message{}` received in the chain.
    field :last_message, :any, virtual: true
    # Track if the state of the chain expects a response from the LLM. This
    # happens after sending a user message or when a tool_call is received, or
    # when we've provided a tool response and the LLM needs to respond.
    field :needs_response, :boolean, default: false

    # A callback function to execute when messages are added. Don't allow caller
    # to setup in `.new` function. Want to set it from the `.run` function to
    # avoid multiple chain instances (across processes) from both firing
    # callbacks.
    field :callback_fn, :any, virtual: true
  end

  @type t :: %LLMChain{}

  @create_fields [:llm, :tools, :custom_context, :verbose]
  @required_fields [:llm]

  @doc """
  Start a new LLMChain configuration.

      {:ok, chain} = LLMChain.new(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: true},
        messages: [%Message.new_system!("You are a helpful assistant.")]
      })
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(attrs \\ %{}) do
    %LLMChain{}
    |> cast(attrs, @create_fields)
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Start a new LLMChain configuration and return it or raise an error if invalid.

      chain = LLMChain.new!(%{
        llm: %ChatOpenAI{model: "gpt-3.5-turbo", stream: true},
        messages: [%Message.new_system!("You are a helpful assistant.")]
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

  def common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
    |> Utils.validate_llm_is_struct()
    |> build_tools_map_from_tools()
  end

  @doc false
  def build_tools_map_from_tools(changeset) do
    tools = get_field(changeset, :tools, [])

    # get a list of all the tools indexed into a map by name
    fun_map =
      Enum.reduce(tools, %{}, fn f, acc ->
        Map.put(acc, f.name, f)
      end)

    put_change(changeset, :_tool_map, fun_map)
  end

  @doc """
  Add a tool to an LLMChain.
  """
  @spec add_tools(t(), Function.t() | [Function.t()]) :: t() | no_return()
  def add_tools(%LLMChain{tools: existing} = chain, tools) do
    updated = existing ++ List.wrap(tools)

    chain
    |> change()
    |> cast(%{tools: updated}, [:tools])
    |> build_tools_map_from_tools()
    |> apply_action!(:update)
  end

  @doc """
  Run the chain on the LLM using messages and any registered functions. This
  formats the request for a ChatLLMChain where messages are passed to the API.

  When successful, it returns `{:ok, updated_chain, message_or_messages}`

  ## Options

  - `:while_needs_response` - repeatedly evaluates functions and submits to the
    LLM so long as we still expect to get a response. Best fit for
    conversational LLMs where a `ToolResult` is used by the LLM to continue.
  - `:callback_fn` - the callback function to execute as messages are received.

  The `callback_fn` is a function that receives one argument. It is the
  LangChain structure for the received message or event. It may be a
  `MessageDelta` or a `Message`. Use pattern matching to respond as desired.
  """
  @spec run(t(), Keyword.t()) :: {:ok, t(), Message.t() | [Message.t()]} | {:error, String.t()}
  def run(chain, opts \\ [])

  def run(%LLMChain{} = chain, opts) do
    # set the callback function on the chain
    chain = %LLMChain{chain | callback_fn: Keyword.get(opts, :callback_fn)}

    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if chain.verbose, do: IO.inspect(chain.messages, label: "MESSAGES")

    tools = chain.tools
    if chain.verbose, do: IO.inspect(tools, label: "TOOLS")

    if Keyword.get(opts, :while_needs_response, false) do
      run_while_needs_response(chain)
    else
      # run the chain and format the return
      case do_run(chain) do
        {:ok, chain} ->
          {:ok, chain, chain.last_message}

        {:error, _reason} = error ->
          error
      end
    end
  end

  # Repeatedly run the chain while `needs_response` is true. This will execute
  # tools and re-submit the tool result to the LLM giving the LLM an
  # opportunity to execute more tools or return a response.
  @spec run_while_needs_response(t()) :: {:ok, t(), Message.t()} | {:error, String.t()}
  defp run_while_needs_response(%LLMChain{needs_response: false} = chain) do
    {:ok, chain, chain.last_message}
  end

  defp run_while_needs_response(%LLMChain{needs_response: true} = chain) do
    chain
    |> execute_tool_calls()
    |> do_run()
    |> case do
      {:ok, updated_chain} ->
        run_while_needs_response(updated_chain)

      {:error, reason} ->
        {:error, reason}
    end
  end

  # internal reusable function for running the chain
  @spec do_run(t()) :: {:ok, t()} | {:error, String.t()}
  defp do_run(%LLMChain{} = chain) do
    # submit to LLM. The "llm" is a struct. Match to get the name of the module
    # then execute the `.call` function on that module.
    %module{} = chain.llm

    # handle and output response
    case module.call(chain.llm, chain.messages, chain.tools, chain.callback_fn) do
      {:ok, [%Message{} = message]} ->
        if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
        {:ok, add_message(chain, message)}

      {:ok, [%Message{} = message, _others] = messages} ->
        if chain.verbose, do: IO.inspect(messages, label: "MULTIPLE MESSAGE RESPONSE")
        # return the list of message responses. Happens when multiple
        # "choices" are returned from LLM by request.
        {:ok, add_message(chain, message)}

      {:ok, [%MessageDelta{} | _] = deltas} ->
        if chain.verbose_deltas, do: IO.inspect(deltas, label: "DELTA MESSAGE LIST RESPONSE")
        updated_chain = apply_deltas(chain, deltas)

        if chain.verbose,
          do: IO.inspect(updated_chain.last_message, label: "COMBINED DELTA MESSAGE RESPONSE")

        {:ok, updated_chain}

      # When calling the latest OpenAI models through Azure it sometimes returns an empty list
      # as the first element of a list of MessageDeltas. This is a workaround to handle that.
      # It would likely be better to handle this in ChatOpenAI instead.
      {:ok, [[] | _] = deltas} ->
        if chain.verbose_deltas, do: IO.inspect(deltas, label: "DELTA MESSAGE LIST RESPONSE")
        updated_chain = apply_deltas(chain, deltas)

        if chain.verbose,
          do: IO.inspect(updated_chain.last_message, label: "COMBINED DELTA MESSAGE RESPONSE")

        {:ok, updated_chain}

      {:ok, [[%MessageDelta{} | _] | _] = deltas} ->
        if chain.verbose_deltas, do: IO.inspect(deltas, label: "DELTA MESSAGE LIST RESPONSE")
        updated_chain = apply_deltas(chain, deltas)

        if chain.verbose,
          do: IO.inspect(updated_chain.last_message, label: "COMBINED DELTA MESSAGE RESPONSE")

        {:ok, updated_chain}

      {:ok, %Message{} = message} ->
        if chain.verbose,
          do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE NO WRAPPED ARRAY")

        {:ok, add_message(chain, message)}

      {:error, reason} ->
        if chain.verbose, do: IO.inspect(reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(reason)}")
        {:error, reason}
    end
  end

  @doc """
  Update the LLMChain's `custom_context` map. Passing in a `context_update` map
  will by default merge the map into the existing `custom_context`.

  Use the `:as` option to:
  - `:merge` - Merge update changes in. Default.
  - `:replace` - Replace the context with the `context_update`.
  """
  @spec update_custom_context(t(), context_update :: %{atom() => any()}, opts :: Keyword.t()) ::
          t() | no_return()
  def update_custom_context(chain, context_update, opts \\ [])

  def update_custom_context(
        %LLMChain{custom_context: %{} = context} = chain,
        %{} = context_update,
        opts
      ) do
    new_context =
      case Keyword.get(opts, :as) || :merge do
        :merge ->
          Map.merge(context, context_update)

        :replace ->
          context_update

        other ->
          raise LangChain.LangChainError,
                "Invalid update_custom_context :as option of #{inspect(other)}"
      end

    %LLMChain{chain | custom_context: new_context}
  end

  def update_custom_context(
        %LLMChain{custom_context: nil} = chain,
        %{} = context_update,
        _opts
      ) do
    # can't merge a map with `nil`. Replace it.
    %LLMChain{chain | custom_context: context_update}
  end

  @doc """
  Apply a received MessageDelta struct to the chain. The LLMChain tracks the
  current merged MessageDelta state. When the final delta is received that
  completes the message, the LLMChain is updated to clear the `delta` and the
  `last_message` and list of messages are updated.
  """
  @spec apply_delta(t(), MessageDelta.t()) :: t()
  def apply_delta(%LLMChain{delta: nil} = chain, %MessageDelta{} = new_delta) do
    %LLMChain{chain | delta: new_delta}
  end

  def apply_delta(%LLMChain{delta: %MessageDelta{} = delta} = chain, %MessageDelta{} = new_delta) do
    merged = MessageDelta.merge_delta(delta, new_delta)
    delta_to_message_when_complete(%LLMChain{chain | delta: merged})
  end

  @doc """
  Convert any hanging delta of the chain to a message and append to the chain.

  If the delta is `nil`, the chain is returned unmodified.
  """
  @spec delta_to_message_when_complete(t()) :: t()
  def delta_to_message_when_complete(
        %LLMChain{delta: %MessageDelta{status: status} = delta} = chain
      )
      when status in [:complete, :length] do
    # it's complete. Attempt to convert delta to a message
    case MessageDelta.to_message(delta) do
      {:ok, %Message{} = message} ->
        fire_callback(chain, message)
        add_message(%LLMChain{chain | delta: nil}, message)

      {:error, reason} ->
        # should not have failed, but it did. Log the error and return
        # the chain unmodified.
        Logger.warning("Error applying delta message. Reason: #{inspect(reason)}")
        chain
    end
  end

  def delta_to_message_when_complete(%LLMChain{} = chain) do
    # either no delta or incomplete
    chain
  end

  @doc """
  Apply a list of deltas to the chain.
  """
  @spec apply_deltas(t(), list()) :: t()
  def apply_deltas(%LLMChain{} = chain, deltas) when is_list(deltas) do
    deltas
    |> List.flatten()
    |> Enum.reduce(chain, fn d, acc -> apply_delta(acc, d) end)
  end

  @doc """
  Add a received Message struct to the chain. The LLMChain tracks the
  `last_message` received and the complete list of messages exchanged. Depending
  on the message role, the chain may be in a pending or incomplete state where
  a response from the LLM is anticipated.
  """
  @spec add_message(t(), Message.t()) :: t()
  def add_message(%LLMChain{} = chain, %Message{} = new_message) do
    needs_response =
      cond do
        new_message.role in [:user, :tool] -> true
        Message.is_tool_call?(new_message) -> true
        new_message.role in [:system, :assistant] -> false
      end

    %LLMChain{
      chain
      | messages: chain.messages ++ [new_message],
        last_message: new_message,
        needs_response: needs_response
    }
  end

  def add_message(%LLMChain{} = _chain, %PromptTemplate{} = template) do
    raise LangChain.LangChainError,
          "PromptTemplates must be converted to messages. You can use LLMChain.apply_prompt_templates/3. Received: #{inspect(template)}"
  end

  @doc """
  Add a set of Message structs to the chain. This enables quickly building a chain
  for submitting to an LLM.
  """
  @spec add_messages(t(), [Message.t()]) :: t()
  def add_messages(%LLMChain{} = chain, messages) do
    Enum.reduce(messages, chain, fn msg, acc ->
      add_message(acc, msg)
    end)
  end

  @doc """
  Apply a set of PromptTemplates to the chain. The list of templates can also
  include Messages with no templates. Provide the inputs to apply to the
  templates for rendering as a message. The prepared messages are applied to the
  chain.
  """
  @spec apply_prompt_templates(t(), [Message.t() | PromptTemplate.t()], %{atom() => any()}) ::
          t() | no_return()
  def apply_prompt_templates(%LLMChain{} = chain, templates, %{} = inputs) do
    messages = PromptTemplate.to_messages!(templates, inputs)
    add_messages(chain, messages)
  end

  @doc """
  Convenience function for setting the prompt text for the LLMChain using
  prepared text.
  """
  @spec quick_prompt(t(), String.t()) :: t()
  def quick_prompt(%LLMChain{} = chain, text) do
    messages = [
      Message.new_system!(),
      Message.new_user!(text)
    ]

    add_messages(chain, messages)
  end

  @doc """
  If the `last_message` from the Assistant includes one or more `ToolCall`s, then the linked
  tool is executed. If there is no `last_message` or the `last_message` is
  not a `tool_call`, the LLMChain is returned with no action performed.
  This makes it safe to call any time.

  The `context` is additional data that will be passed to the executed tool.
  The value given here will override any `custom_context` set on the LLMChain.
  If not set, the global `custom_context` is used.
  """
  @spec execute_tool_calls(t(), context :: nil | %{atom() => any()}) :: t()
  def execute_tool_calls(chain, context \\ nil)
  def execute_tool_calls(%LLMChain{last_message: nil} = chain, _context), do: chain

  def execute_tool_calls(
        %LLMChain{last_message: %Message{} = message} = chain,
        context
      ) do
    if Message.is_tool_call?(message) do
      # context to use
      use_context = context || chain.custom_context
      verbose = chain.verbose

      # Get all the tools to call. Accumulate them into a map.
      # Stored as
      grouped =
        Enum.reduce(message.tool_calls, %{async: [], sync: [], invalid: []}, fn call, acc ->
          case chain._tool_map[call.name] do
            %Function{async: true} = func ->
              Map.put(acc, :async, acc.async ++ [{call, func}])

            %Function{async: false} = func ->
              Map.put(acc, :sync, acc.sync ++ [{call, func}])

            # invalid tool call
            nil ->
              Map.put(acc, :invalid, acc.invalid ++ [{call, nil}])
          end
        end)

      # execute all the async calls. This keeps the responses in order too.
      async_results =
        grouped[:async]
        |> Enum.map(fn {call, func} ->
          Task.async(fn ->
            execute_tool_call(call, func, verbose: verbose, context: use_context)
          end)
        end)
        |> Task.await_many()

      sync_results =
        Enum.map(grouped[:sync], fn {call, func} ->
          execute_tool_call(call, func, verbose: verbose, context: use_context)
        end)

      # log invalid tool calls
      invalid_calls =
        Enum.map(grouped[:invalid], fn {call, _} ->
          text = "Tool call made to #{call.name} but tool not found"
          Logger.warning(text)

          ToolResult.new!(%{tool_call_id: call.call_id, content: text, is_error: true})
        end)

      combined_results = async_results ++ sync_results ++ invalid_calls

      # create a single tool message that contains all the tool results
      message =
        Message.new_tool_result!(%{content: message.content, tool_results: combined_results})

      if chain.verbose, do: IO.inspect(message, label: "TOOL RESULTS")
      fire_callback(chain, message)

      # add all the message to the chain
      LLMChain.add_message(chain, message)
    else
      # Not a complete tool call
      chain
    end
  end

  @doc """
  Execute the tool call with the tool. Returns the tool's message response.
  """
  @spec execute_tool_call(ToolCall.t(), Function.t(), Keyword.t()) :: ToolResult.t()
  def execute_tool_call(%ToolCall{} = call, %Function{} = function, opts \\ []) do
    verbose = Keyword.get(opts, :verbose, false)
    context = Keyword.get(opts, :context, nil)

    try do
      if verbose, do: IO.inspect(function.name, label: "EXECUTING FUNCTION")

      case Function.execute(function, call.arguments, context) do
        {:ok, result} ->
          if verbose, do: IO.inspect(result, label: "FUNCTION RESULT")
          # successful execution.
          ToolResult.new!(%{
            tool_call_id: call.call_id,
            content: result,
            name: function.name,
            display_text: function.display_text
          })

        {:error, reason} when is_binary(reason) ->
          if verbose, do: IO.inspect(reason, label: "FUNCTION ERROR")

          ToolResult.new!(%{
            tool_call_id: call.call_id,
            content: reason,
            name: function.name,
            display_text: function.display_text,
            is_error: true
          })
      end
    rescue
      err ->
        Logger.error("Function #{function.name} failed in execution. Exception: #{inspect(err)}")

        ToolResult.new!(%{
          tool_call_id: call.call_id,
          content: "ERROR executing tool: #{inspect(err)}",
          is_error: true
        })
    end
  end

  # Fire the callback if set.
  defp fire_callback(%LLMChain{callback_fn: nil}, _data), do: :ok

  # OPTIONAL: Execute callback function
  defp fire_callback(%LLMChain{callback_fn: callback_fn}, data) when is_function(callback_fn) do
    data
    |> List.wrap()
    |> List.flatten()
    |> Enum.each(fn item -> callback_fn.(item) end)

    :ok
  end

  @doc """
  Remove an incomplete MessageDelta from `delta` and add a Message with the
  desired status to the chain.
  """
  def cancel_delta(%LLMChain{delta: nil} = chain, _message_status), do: chain

  def cancel_delta(%LLMChain{delta: delta} = chain, message_status) do
    # remove the in-progress delta
    updated_chain = %LLMChain{chain | delta: nil}

    case MessageDelta.to_message(%MessageDelta{delta | status: :complete}) do
      {:ok, message} ->
        message = %Message{message | status: message_status}
        add_message(updated_chain, message)

      {:error, reason} ->
        Logger.error("Error attempting to cancel_delta. Reason: #{inspect(reason)}")
        chain
    end
  end
end
