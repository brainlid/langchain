defmodule LangChain.Chains.LLMChain do
  @doc """
  Define an LLMChain. This is the heart of the LangChain library.

  The chain deals with tools, a tool map, delta tracking, last_message tracking,
  conversation messages, and verbose logging. This helps by separating these
  responsibilities from the LLM making it easier to support additional LLMs
  because the focus is on communication and formats instead of all the extra
  logic.

  ## Callbacks

  Callbacks are fired as specific events occur in the chain as it is running.
  The set of events are defined in `LangChain.Chains.ChainCallbacks`.

  To be notified of an event you care about, register a callback handler with
  the chain. Multiple callback handlers can be assigned. The callback handler
  assigned to the `LLMChain` is not provided to an LLM chat model. For callbacks
  on a chat model, set them there.

  ### Registering a callback handler

  A handler is a map with key name for the callback to fire. A function is
  assigned to the map key. Refer to the documentation for each function as they
  arguments vary.

  If we want to be notified when an LLM Assistant chat response message has been
  processed and it is complete, this is how we could receive that event in our
  running LiveView:

      live_view_pid = self()

      handler = %{
        on_message_processed: fn _chain, message ->
          send(live_view_pid, {:new_assistant_response, message})
        end
      }

      LLMChain.new!(%{...})
      |> LLMChain.add_callback(handler)
      |> LLMChain.run()

  In the LiveView, a `handle_info` function executes with the received message.
  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Callbacks
  alias LangChain.Chains.ChainCallbacks
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

    # A set of message pre-processors to execute on received messages.
    field :message_processors, {:array, :any}, default: [], virtual: true

    # The maximum consecutive LLM response failures permitted before failing the
    # process.
    field :max_retry_count, :integer, default: 3
    # Internal failure count tracker. Is reset on a successful assistant
    # response.
    field :current_failure_count, :integer, default: 0

    # Track the current merged `%MessageDelta{}` struct received when streamed.
    # Set to `nil` when there is no current delta being tracked. This happens
    # when the final delta is received that completes the message. At that point,
    # the delta is converted to a message and the delta is set to nil.
    field :delta, :any, virtual: true
    # Track the last `%Message{}` received in the chain.
    field :last_message, :any, virtual: true
    # Track if the state of the chain expects a response from the LLM. This
    # happens after sending a user message, when a tool_call is received, or
    # when we've provided a tool response and the LLM needs to respond.
    field :needs_response, :boolean, default: false

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []
  end

  # default to 2 minutes
  @task_await_timeout 2 * 60 * 1000

  @type t :: %LLMChain{}

  @typedoc """
  The expected return types for a Message processor function. When successful,
  it returns a `:continue` with an Message to use as a replacement. When it
  fails, a `:halt` is returned along with an updated `LLMChain.t()` and a new
  user message to be returned to the LLM reporting the error.
  """
  @type processor_return :: {:continue, Message.t()} | {:halt, t(), Message.t()}

  @typedoc """
  A message processor is an arity 2 function that takes an LLMChain and a
  Message. It is used to "pre-process" the received message from the LLM.
  Processors can be chained together to preform a sequence of transformations.
  """
  @type message_processor :: (t(), Message.t() -> processor_return())

  @create_fields [:llm, :tools, :custom_context, :max_retry_count, :callbacks, :verbose]
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
  Register a set of processors to on received assistant messages.
  """
  @spec message_processors(t(), [message_processor()]) :: t()
  def message_processors(%LLMChain{} = chain, processors) do
    %LLMChain{chain | message_processors: processors}
  end

  @doc """
  Run the chain on the LLM using messages and any registered functions. This
  formats the request for a ChatLLMChain where messages are passed to the API.

  When successful, it returns `{:ok, updated_chain, message_or_messages}`

  ## Options

  - `:mode` - It defaults to run the chain one time, stopping after receiving a
    response from the LLM. Supports `:until_success` and
    `:while_needs_response`.

  - `mode: :until_success` - (for non-interactive processing done by the LLM
    where it may repeatedly fail and need to re-try) Repeatedly evaluates a
    received message through any message processors, returning any errors to the
    LLM until it either succeeds or exceeds the `max_retry_count`. Ths includes
    evaluating received `ToolCall`s until they succeed. If an LLM makes 3
    ToolCalls in a single message and 2 succeed while 1 fails, the success
    responses are returned to the LLM with the failure response of the remaining
    `ToolCall`, giving the LLM an opportunity to resend the failed `ToolCall`,
    and only the failed `ToolCall` until it succeeds or exceeds the
    `max_retry_count`. In essence, once we have a successful response from the
    LLM, we don't return any more to it and don't want any further responses.

  - `mode: :while_needs_response` - (for interactive chats that make
    `ToolCalls`) Repeatedly evaluates functions and submits to the LLM so long
    as we still expect to get a response. Best fit for conversational LLMs where
    a `ToolResult` is used by the LLM to continue. After all `ToolCall` messages
    are evaluated, the `ToolResult` messages are returned to the LLM giving it
    an opportunity to use the `ToolResult` information in an assistant response
    message. In essence, this mode always gives the LLM the last word.
  """
  @spec run(t(), Keyword.t()) ::
          {:ok, t(), Message.t() | [Message.t()]} | {:error, t(), String.t()}
  def run(chain, opts \\ [])

  def run(%LLMChain{} = chain, opts) do
    # set the callback function on the chain
    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if chain.verbose, do: IO.inspect(chain.messages, label: "MESSAGES")

    tools = chain.tools
    if chain.verbose, do: IO.inspect(tools, label: "TOOLS")

    case Keyword.get(opts, :mode, nil) do
      nil ->
        # run the chain and format the return
        case do_run(chain) do
          {:ok, chain} ->
            {:ok, chain, chain.last_message}

          {:error, _chain, _reason} = error ->
            error
        end

      :while_needs_response ->
        run_while_needs_response(chain)

      :until_success ->
        run_until_success(chain)
    end
  end

  # Repeatedly run the chain until we get a successful ToolResponse or processed
  # assistant message. Once we've reached success, it is not submitted back to the LLM,
  # the process ends there.
  @spec run_until_success(t()) :: {:ok, t(), Message.t()} | {:error, t(), String.t()}
  defp run_until_success(%LLMChain{last_message: %Message{} = last_message} = chain) do
    stop_or_recurse =
      cond do
        chain.current_failure_count >= chain.max_retry_count ->
          {:error, chain, "Exceeded max failure count"}

        last_message.role == :tool && !Message.tool_had_errors?(last_message) ->
          # a successful tool result is success
          {:ok, chain, last_message}

        last_message.role == :assistant ->
          # it was successful if we didn't generate a user message in response to
          # an error.
          {:ok, chain, last_message}

        true ->
          :recurse
      end

    case stop_or_recurse do
      :recurse ->
        chain
        |> do_run()
        |> case do
          {:ok, updated_chain} ->
            updated_chain
            |> execute_tool_calls()
            |> run_until_success()

          {:error, updated_chain, reason} ->
            {:error, updated_chain, reason}
        end

      other ->
        # return the error or success result
        other
    end
  end

  # Repeatedly run the chain while `needs_response` is true. This will execute
  # tools and re-submit the tool result to the LLM giving the LLM an
  # opportunity to execute more tools or return a response.
  @spec run_while_needs_response(t()) :: {:ok, t(), Message.t()} | {:error, t(), String.t()}
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

      {:error, updated_chain, reason} ->
        {:error, updated_chain, reason}
    end
  end

  # internal reusable function for running the chain
  @spec do_run(t()) :: {:ok, t()} | {:error, t(), String.t()}
  defp do_run(%LLMChain{current_failure_count: current_count, max_retry_count: max} = chain)
       when current_count >= max do
    Callbacks.fire(chain.callbacks, :on_retries_exceeded, [chain])
    {:error, chain, "Exceeded max failure count"}
  end

  defp do_run(%LLMChain{} = chain) do
    # submit to LLM. The "llm" is a struct. Match to get the name of the module
    # then execute the `.call` function on that module.
    %module{} = chain.llm

    # handle and output response
    case module.call(chain.llm, chain.messages, chain.tools) do
      {:ok, [%Message{} = message]} ->
        if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
        {:ok, process_message(chain, message)}

      {:ok, [%Message{} = message, _others] = messages} ->
        if chain.verbose, do: IO.inspect(messages, label: "MULTIPLE MESSAGE RESPONSE")
        # return the list of message responses. Happens when multiple
        # "choices" are returned from LLM by request.
        {:ok, process_message(chain, message)}

      {:ok, %Message{} = message} ->
        if chain.verbose,
          do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE NO WRAPPED ARRAY")

        {:ok, process_message(chain, message)}

      {:ok, [%MessageDelta{} | _] = deltas} ->
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

      {:error, reason} ->
        if chain.verbose, do: IO.inspect(reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(reason)}")
        {:error, chain, reason}
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
        process_message(%LLMChain{chain | delta: nil}, message)

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

  # Process an assistant message sequentially through each message processor.
  @doc false
  @spec run_message_processors(t(), Message.t()) ::
          Message.t() | {:halted, Message.t(), Message.t()}
  def run_message_processors(
        %LLMChain{message_processors: processors} = chain,
        %Message{role: :assistant} = message
      )
      when is_list(processors) and processors != [] do
    # start `processed_content` with the message's content
    message = %Message{message | processed_content: message.content}

    processors
    |> Enum.reduce_while(message, fn proc, m = _acc ->
      try do
        case proc.(chain, m) do
          {:cont, updated_msg} ->
            if chain.verbose, do: IO.inspect(proc, label: "MESSAGE PROCESSOR EXECUTED")
            {:cont, updated_msg}

          {:halt, %Message{} = returned_message} ->
            if chain.verbose, do: IO.inspect(proc, label: "MESSAGE PROCESSOR HALTED")
            # for debugging help, return the message so-far that failed in the
            # processor
            {:halt, {:halted, m, returned_message}}
        end
      rescue
        err ->
          Logger.error("Exception raised in processor #{inspect(proc)}")

          {:halt,
           {:halted,
            Message.new_user!("ERROR: An exception was raised! Exception: #{inspect(err)}")}}
      end
    end)
  end

  # the message is not an assistant message. Skip message processing.
  def run_message_processors(%LLMChain{} = _chain, %Message{} = message) do
    message
  end

  @doc """
  Process a newly message received from the LLM. Messages with a role of
  `:assistant` may be processed through the `message_processors` before being
  generally available or being notified through a callback.
  """
  @spec process_message(t(), Message.t()) :: t()
  def process_message(%LLMChain{} = chain, %Message{} = message) do
    case run_message_processors(chain, message) do
      {:halted, failed_message, new_message} ->
        if chain.verbose do
          IO.inspect(failed_message, label: "PROCESSOR FAILED ON MESSAGE")
          IO.inspect(new_message, label: "PROCESSOR FAILURE RESPONSE MESSAGE")
        end

        # add the received assistant message, then add the newly created user
        # return message and return the updated chain
        chain
        |> increment_current_failure_count()
        |> add_message(failed_message)
        |> add_message(new_message)
        |> fire_callback_and_return(:on_message_processing_error, [failed_message])
        |> fire_callback_and_return(:on_error_message_created, [new_message])

      %Message{role: :assistant} = updated_message ->
        if chain.verbose, do: IO.inspect(updated_message, label: "MESSAGE PROCESSED")

        chain
        |> add_message(updated_message)
        |> reset_current_failure_count_if(fn -> !Message.is_tool_related?(updated_message) end)
        |> fire_callback_and_return(:on_message_processed, [updated_message])
    end
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
        |> Task.await_many(@task_await_timeout)

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
      result_message =
        Message.new_tool_result!(%{content: message.content, tool_results: combined_results})

      # add the tool result message to the chain
      updated_chain = LLMChain.add_message(chain, result_message)

      # if the tool results had an error, increment the failure counter. If not,
      # clear it.
      updated_chain =
        if Message.tool_had_errors?(result_message) do
          # something failed, increment our error counter
          LLMChain.increment_current_failure_count(updated_chain)
        else
          # no errors, clear any errors
          LLMChain.reset_current_failure_count(updated_chain)
        end

      # fire the callbacks
      if chain.verbose, do: IO.inspect(result_message, label: "TOOL RESULTS")

      fire_callback_and_return(updated_chain, :on_tool_response_created, [result_message])
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

  @doc """
  Increments the internal current_failure_count. Returns and incremented and
  updated struct.
  """
  @spec increment_current_failure_count(t()) :: t()
  def increment_current_failure_count(%LLMChain{} = chain) do
    %LLMChain{chain | current_failure_count: chain.current_failure_count + 1}
  end

  @doc """
  Reset the internal current_failure_count to 0. Useful after receiving a
  successfully returned and processed message from the LLM.
  """
  @spec reset_current_failure_count(t()) :: t()
  def reset_current_failure_count(%LLMChain{} = chain) do
    %LLMChain{chain | current_failure_count: 0}
  end

  @doc """
  Reset the internal current_failure_count to 0 if the function provided returns
  `true`. Helps to make the change conditional.
  """
  @spec reset_current_failure_count_if(t(), (-> boolean())) :: t()
  def reset_current_failure_count_if(%LLMChain{} = chain, fun) do
    if fun.() == true do
      %LLMChain{chain | current_failure_count: 0}
    else
      chain
    end
  end

  @doc """
  Add another callback to the list of callbacks.
  """
  @spec add_callback(t(), ChainCallbacks.chain_callback_handler()) :: t()
  def add_callback(%LLMChain{callbacks: callbacks} = chain, additional_callback) do
    %LLMChain{chain | callbacks: callbacks ++ [additional_callback]}
  end

  @doc """
  Add a `LangChain.ChatModels.LLMCallbacks` callback map to the chain's `:llm` model if
  it supports the `:callback` key.
  """
  @spec add_llm_callback(t(), map()) :: t()
  def add_llm_callback(%LLMChain{llm: model} = chain, callback_map) do
    %LLMChain{chain | llm: ChatModel.add_callback(model, callback_map)}
  end

  # a pipe-friendly execution of callbacks that returns the chain
  defp fire_callback_and_return(%LLMChain{} = chain, callback_name, additional_arguments)
       when is_list(additional_arguments) do
    Callbacks.fire(chain.callbacks, callback_name, [chain] ++ additional_arguments)
    chain
  end
end
