defmodule LangChain.Chains.LLMChain do
  @doc """
  Define an LLMChain. This is the heart of the LangChain library.

  The chain deals with tools, a tool map, delta tracking, tracking the messages
  exchanged during a run, the last_message tracking, conversation messages, and
  verbose logging. Messages and tool results support multi-modal ContentParts,
  enabling richer responses (text, images, files, thinking, etc.). ToolResult
  content can be a list of ContentParts. The chain also supports
  `async_tool_timeout` and improved fallback handling.

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

  ## Fallbacks

  When running a chain, the `:with_fallbacks` option can be used to provide a
  list of fallback chat models to try when a failure is encountered.

  When working with language models, you may often encounter issues from the
  underlying APIs, whether these be rate limiting, downtime, or something else.
  Therefore, as you go to move your LLM applications into production it becomes
  more and more important to safeguard against these. That's what fallbacks are
  designed to provide.

  A **fallback** is an alternative plan that may be used in an emergency.

  A `before_fallback` function can be provided to alter or return a different
  chain to use with the fallback LLM model. This is important because often, the
  prompts needed for will differ for a fallback LLM. This means if your OpenAI
  completion fails, a different prompt may be needed when retrying it with an
  Anthropic fallback.

  ### Fallback for LLM API Errors

  This is perhaps the most common use case for fallbacks. A request to an LLM
  API can fail for a variety of reasons - the API could be down, you could have
  hit rate limits, any number of things. Therefore, using fallbacks can help
  protect against these types of failures.

  ## Fallback Examples

  A simple fallback that tries a different LLM chat model

      fallback_llm = ChatAnthropic.new!(%{stream: false})

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!("OpenAI system prompt"))
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run(with_fallbacks: [fallback_llm])

  Note the `with_fallbacks: [fallback_llm]` option when running the chain.

  This example uses the `:before_fallback` option to provide a function that can
  modify or return an alternate chain when used with a certain LLM. Also note
  the utility function `LangChain.Utils.replace_system_message!/2` is used for
  swapping out the system message when falling back to a different LLM.

      fallback_llm = ChatAnthropic.new!(%{stream: false})

      {:ok, updated_chain} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!("OpenAI system prompt"))
        |> LLMChain.add_message(Message.new_user!("Why is the sky blue?"))
        |> LLMChain.run(
          with_fallbacks: [fallback_llm],
          before_fallback: fn chain ->
            case chain.llm do
              %ChatAnthropic{} ->
                # replace the system message
                %LLMChain{
                  chain
                  | messages:
                      Utils.replace_system_message!(
                        chain.messages,
                        Message.new_system!("Anthropic system prompt")
                      )
                }

              _open_ai ->
                chain
            end
          end
        )

  See `LangChain.Chains.LLMChain.run/2` for more details.

  ## Run Until Tool Used

  The `run_until_tool_used/3` function makes it easy to instruct an LLM to use a
  set of tools and then call a specific tool to present the results. This is
  particularly useful for complex workflows where you want the LLM to perform
  multiple operations and then finalize with a specific action.

  This works well for receiving a final structured output after multiple tools
  are used.

  When the specified tool is successfully called, the chain stops processing and
  returns the result. This prevents unnecessary additional LLM calls and
  provides a clear termination point for your workflow.

      {:ok, %LLMChain{} = updated_chain, %ToolResult{} = tool_result} =
        %{llm: ChatOpenAI.new!(%{stream: false})}
        |> LLMChain.new!()
        |> LLMChain.add_tools([special_search, report_results])
        |> LLMChain.add_message(Message.new_system!())
        |> LLMChain.add_message(Message.new_user!("..."))
        |> LLMChain.run_until_tool_used("final_summary")

  The function returns a tuple with three elements:
  - `:ok` - Indicating success
  - The updated chain with all messages and tool calls
  - The specific tool result that matched the requested tool name

  To prevent runaway function calls, a default `max_runs` value of 25 is set.
  You can adjust this as needed:

      # Allow up to 50 runs before timing out
      LLMChain.run_until_tool_used(chain, "final_summary", max_runs: 50)

  The function also supports fallbacks, allowing you to gracefully handle LLM
  failures:

      LLMChain.run_until_tool_used(chain, "final_summary",
        max_runs: 10,
        with_fallbacks: [fallback_llm],
        before_fallback: fn chain ->
          # Modify chain before using fallback LLM
          chain
        end
      )

  See `LangChain.Chains.LLMChain.run_until_tool_used/3` for more details.

  """
  use Ecto.Schema
  import Ecto.Changeset
  require Logger
  alias LangChain.Callbacks
  alias LangChain.Chains.ChainCallbacks
  alias LangChain.PromptTemplate
  alias __MODULE__
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.NativeTool

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
    # Internally managed. The list of exchanged messages during a `run` function
    # execution. A single run can result in a number of newly created messages.
    # It generates an Assistant message with one or more ToolCalls, the message
    # with tool results where some of them may have failed, requiring the LLM to
    # try again. This list tracks the full set of exchanged messages during a
    # single run.
    field :exchanged_messages, {:array, :any}, default: [], virtual: true
    # Track if the state of the chain expects a response from the LLM. This
    # happens after sending a user message, when a tool_call is received, or
    # when we've provided a tool response and the LLM needs to respond.
    field :needs_response, :boolean, default: false

    # The timeout for async tool execution. An async Task execution is used when
    # running a tool that has `async: true` set. Time is in milliseconds.
    field :async_tool_timeout, :integer

    # A list of maps for callback handlers
    field :callbacks, {:array, :map}, default: []
  end

  # default to 2 minutes
  @default_task_await_timeout 2 * 60 * 1000

  @type t :: %LLMChain{}

  @typedoc """
  The expected return types for a Message processor function. When successful,
  it returns a `:cont` with an Message to use as a replacement. When it
  fails, a `:halt` is returned along with an updated `LLMChain.t()` and a new
  user message to be returned to the LLM reporting the error.
  """
  @type processor_return :: {:cont, Message.t()} | {:halt, t(), Message.t()}

  @typedoc """
  A message processor is an arity 2 function that takes an
  `LangChain.Chains.LLMChain` and a `LangChain.Message`. It is used to
  "pre-process" the received message from the LLM. Processors can be chained
  together to perform a sequence of transformations.

  The return of the processor is a tuple with a keyword and a message. The
  keyword is either `:cont` or `:halt`. If `:cont` is returned, the
  message is used as the next message in the chain. If `:halt` is returned, the
  halting message is returned to the LLM as an error and no further processors
  will handle the message.

  An example of this is the `LangChain.MessageProcessors.JsonProcessor` which
  parses the message content as JSON and returns the parsed data as a map. If
  the content is not valid JSON, the processor returns a halting message with an
  error message for the LLM to respond to.
  """
  @type message_processor :: (t(), Message.t() -> processor_return())

  @create_fields [
    :llm,
    :tools,
    :custom_context,
    :max_retry_count,
    :callbacks,
    :verbose,
    :verbose_deltas,
    :async_tool_timeout
  ]
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
  @spec add_tools(t(), NativeTool.t() | Function.t() | [Function.t()]) :: t() | no_return()
  def add_tools(%LLMChain{tools: existing} = chain, tools) do
    updated = existing ++ List.wrap(tools)

    chain
    |> change()
    |> cast(%{tools: updated}, [:tools])
    |> build_tools_map_from_tools()
    |> apply_action!(:update)
  end

  @doc """
  Register a set of processors to be applied to received assistant messages.
  """
  @spec message_processors(t(), [message_processor()]) :: t()
  def message_processors(%LLMChain{} = chain, processors) do
    %LLMChain{chain | message_processors: processors}
  end

  @doc """
  Run the chain on the LLM using messages and any registered functions. This
  formats the request for a ChatLLMChain where messages are passed to the API.

  When successful, it returns `{:ok, updated_chain}`

  ## Options

  - `:mode` - It defaults to run the chain one time, stopping after receiving a
    response from the LLM. Supports `:until_success` and
    `:while_needs_response`.

  - `mode: :until_success` - (for non-interactive processing done by the LLM
    where it may repeatedly fail and need to re-try) Repeatedly evaluates a
    received message through any message processors, returning any errors to the
    LLM until it either succeeds or exceeds the `max_retry_count`. This includes
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

  - `with_fallbacks: [...]` - Provide a list of chat models to use as a fallback
    when one fails. This helps a production system remain operational when an
    API limit is reached, an LLM service is overloaded or down, or something
    else new an exciting goes wrong.

    When all fallbacks fail, a `%LangChainError{type: "all_fallbacks_failed"}`
    is returned in the error response.

  - `before_fallback: fn chain -> modified_chain end` - A `before_fallback`
    function is called before the LLM call is made. **NOTE: When provided, it
    also fires for the first attempt.** This allows a chain to be modified or
    replaced before running against the configured LLM. This is helpful, for
    example, when a different system prompt is needed for Anthropic vs OpenAI.

  ## Mode Examples

  **Use Case**: A chat with an LLM where functions are available to the LLM:

      LLMChain.run(chain, mode: :while_needs_response)

  This will execute any LLM called functions, returning the result to the LLM,
  and giving it a chance to respond to the results.

  **Use Case**: An application that exposes a function to the LLM, but we want
  to stop once the function is successfully executed. When errors are
  encountered, the LLM should be given error feedback and allowed to try again.

      LLMChain.run(chain, mode: :until_success)

  """
  @spec run(t(), Keyword.t()) :: {:ok, t()} | {:error, t(), LangChainError.t()}
  def run(chain, opts \\ [])

  def run(%LLMChain{} = chain, opts) do
    try do
      raise_on_obsolete_run_opts(opts)
      raise_when_no_messages(chain)
      initial_run_logging(chain)

      # clear the set of exchanged messages.
      chain = clear_exchanged_messages(chain)

      # determine which function to run based on the mode.
      function_to_run =
        case Keyword.get(opts, :mode, nil) do
          nil ->
            &do_run/1

          :while_needs_response ->
            &run_while_needs_response/1

          :until_success ->
            &run_until_success/1
        end

      # Add telemetry for chain execution
      metadata = %{
        chain_type: "llm_chain",
        mode: Keyword.get(opts, :mode, "default"),
        message_count: length(chain.messages),
        tool_count: length(chain.tools)
      }

      LangChain.Telemetry.span([:langchain, :chain, :execute], metadata, fn ->
        # Run the chain and return the success or error results. NOTE: We do not add
        # the current LLM to the list and process everything through a single
        # codepath because failing after attempted fallbacks returns a different
        # error.
        if Keyword.has_key?(opts, :with_fallbacks) do
          # run function and using fallbacks as needed.
          with_fallbacks(chain, opts, function_to_run)
        else
          # run it directly right now and return the success or error
          function_to_run.(chain)
        end
      end)
    rescue
      err in LangChainError ->
        {:error, chain, err}
    end
  end

  defp initial_run_logging(%LLMChain{verbose: false} = _chain), do: :ok

  defp initial_run_logging(%LLMChain{verbose: true} = chain) do
    # set the callback function on the chain
    if chain.verbose, do: IO.inspect(chain.llm, label: "LLM")

    if chain.verbose, do: IO.inspect(chain.messages, label: "MESSAGES")

    if chain.verbose, do: IO.inspect(chain.tools, label: "TOOLS")

    :ok
  end

  defp with_fallbacks(%LLMChain{} = chain, opts, run_fn) do
    # Sources of inspiration:
    # - https://python.langchain.com/v0.1/docs/guides/productionization/fallbacks/
    # - https://python.langchain.com/docs/how_to/fallbacks/
    # - https://python.langchain.com/docs/how_to/fallbacks/

    llm_list = Keyword.fetch!(opts, :with_fallbacks)
    before_fallback_fn = Keyword.get(opts, :before_fallback, nil)

    # try the chain where we go through the full list of LLMs to try. Add the
    # current LLM as the first so all are processed the same way.
    try_chain_with_llm(chain, [chain.llm | llm_list], before_fallback_fn, run_fn)
  end

  # nothing left to try
  defp try_chain_with_llm(chain, [], _before_fallback_fn, _run_fn) do
    {:error, chain,
     LangChainError.exception(
       type: "all_fallbacks_failed",
       message: "Failed all attempts to generate response"
     )}
  end

  defp try_chain_with_llm(chain, [llm | tail], before_fallback_fn, run_fn) do
    use_chain = %LLMChain{chain | llm: llm}

    use_chain =
      if before_fallback_fn do
        # use the returned chain from the before_fallback function.
        before_fallback_fn.(use_chain)
      else
        use_chain
      end

    try do
      case run_fn.(use_chain) do
        {:ok, result} ->
          {:ok, result}

        {:error, _error_chain, reason} ->
          # run attempt received an error. Try again with the next LLM
          Logger.warning("LLM call failed, using next fallback. Reason: #{inspect(reason)}")

          try_chain_with_llm(use_chain, tail, before_fallback_fn, run_fn)
      end
    rescue
      err ->
        # Log the error and stack trace, then try again.
        Logger.error(
          "Rescued from exception during with_fallback processing. Error: #{inspect(err)}\nStack trace:\n#{Exception.format(:error, err, __STACKTRACE__)}"
        )

        try_chain_with_llm(use_chain, tail, before_fallback_fn, run_fn)
    end
  end

  # Repeatedly run the chain until we get a successful ToolResponse or processed
  # assistant message. Once we've reached a successful response, it is not
  # submitted back to the LLM, the process ends there.
  @spec run_until_success(t()) :: {:ok, t()} | {:error, t(), LangChainError.t()}
  defp run_until_success(
         %LLMChain{last_message: %Message{} = last_message} = chain,
         force_recurse \\ false
       ) do
    stop_or_recurse =
      cond do
        force_recurse ->
          :recurse

        chain.current_failure_count >= chain.max_retry_count ->
          {:error, chain,
           LangChainError.exception(
             type: "exceeded_failure_count",
             message: "Exceeded max failure count"
           )}

        last_message.role == :tool && !Message.tool_had_errors?(last_message) ->
          # a successful tool result has no errors
          {:ok, chain}

        last_message.role == :assistant ->
          # it was successful if we didn't generate a user message in response to
          # an error.
          {:ok, chain}

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
  @spec run_while_needs_response(t()) :: {:ok, t()} | {:error, t(), LangChainError.t()}
  defp run_while_needs_response(%LLMChain{needs_response: false} = chain) do
    {:ok, chain}
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

  @doc """
  Run the chain until a specific tool call is made. This makes it easy for an
  LLM to make multiple tool calls and call a specific tool to return a result,
  signaling the end of the operation.

  ## Options

  - `max_runs`: The maximum number of times to run the chain. To prevent runaway
    calls, it defaults to 25. When exceeded, a `%LangChainError{type: "exceeded_max_runs"}`
    is returned in the error response.

  - `with_fallbacks: [...]` - Provide a list of chat models to use as a fallback
    when one fails. This helps a production system remain operational when an
    API limit is reached, an LLM service is overloaded or down, or something
    else new an exciting goes wrong.

    When all fallbacks fail, a `%LangChainError{type: "all_fallbacks_failed"}`
    is returned in the error response.

  - `before_fallback: fn chain -> modified_chain end` - A `before_fallback`
    function is called before the LLM call is made. **NOTE: When provided, it
    also fires for the first attempt.** This allows a chain to be modified or
    replaced before running against the configured LLM. This is helpful, for
    example, when a different system prompt is needed for Anthropic vs OpenAI.
  """
  @spec run_until_tool_used(t(), String.t()) ::
          {:ok, t(), Message.t()} | {:error, t(), LangChainError.t()}
  def run_until_tool_used(%LLMChain{} = chain, tool_name, opts \\ []) do
    chain
    |> raise_when_no_messages()
    |> initial_run_logging()

    # clear the set of exchanged messages.
    chain = clear_exchanged_messages(chain)

    # Check if the tool_name exists in the registered tools
    if Map.has_key?(chain._tool_map, tool_name) do
      # Preserve fallback options and max_runs count if set explicitly.
      do_run_until_tool_used(chain, tool_name, Keyword.put_new(opts, :max_runs, 25))
    else
      {:error, chain,
       LangChainError.exception(
         type: "invalid_tool_name",
         message: "Tool name '#{tool_name}' not found in available tools"
       )}
    end
  end

  defp do_run_until_tool_used(%LLMChain{} = chain, tool_name, opts) do
    max_runs = Keyword.get(opts, :max_runs)

    if max_runs <= 0 do
      {:error, chain,
       LangChainError.exception(
         type: "exceeded_max_runs",
         message: "Exceeded maximum number of runs"
       )}
    else
      # Decrement max_runs for next recursion
      next_opts = Keyword.put(opts, :max_runs, max_runs - 1)

      # Add telemetry for run_until_tool_used chain execution
      metadata = %{
        chain_type: "llm_chain",
        mode: "run_until_tool_used",
        message_count: length(chain.messages),
        tool_count: length(chain.tools)
      }

      run_result =
        try do
          LangChain.Telemetry.span([:langchain, :chain, :execute], metadata, fn ->
            # Run the chain and return the success or error results. NOTE: We do
            # not add the current LLM to the list and process everything through a
            # single codepath because failing after attempted fallbacks returns a
            # different error.
            #
            # The run_until_success passes in a `true` force it to recuse and call
            # even if a ToolResult was successfully run. We check _which_ tool
            # result was returned here and make a separate decision.
            if Keyword.has_key?(opts, :with_fallbacks) do
              # run function and using fallbacks as needed.
              with_fallbacks(chain, opts, &run_until_success(&1, true))
            else
              # run it directly right now and return the success or error
              run_until_success(chain, true)
            end
          end)
        rescue
          err in LangChainError ->
            {:error, chain, err}
        end

      case run_result do
        {:ok, updated_chain} ->
          # Check if the last message contains a tool call matching the
          # specified name
          case updated_chain.last_message do
            %Message{role: :tool, tool_results: tool_results} when is_list(tool_results) ->
              matching_call = Enum.find(tool_results, &(&1.name == tool_name))

              if matching_call do
                {:ok, updated_chain, matching_call}
              else
                # If no matching tool result found, continue running.
                do_run_until_tool_used(updated_chain, tool_name, next_opts)
              end

            _ ->
              # If no tool results in last message, continue running
              do_run_until_tool_used(updated_chain, tool_name, next_opts)
          end

        {:error, updated_chain, reason} ->
          {:error, updated_chain, reason}
      end
    end
  end

  # internal reusable function for running the chain
  @spec do_run(t()) :: {:ok, t()} | {:error, t(), LangChainError.t()}
  defp do_run(%LLMChain{current_failure_count: current_count, max_retry_count: max} = chain)
       when current_count >= max do
    Callbacks.fire(chain.callbacks, :on_retries_exceeded, [chain])

    {:error, chain,
     LangChainError.exception(
       type: "exceeded_failure_count",
       message: "Exceeded max failure count"
     )}
  end

  defp do_run(%LLMChain{} = chain) do
    # submit to LLM. The "llm" is a struct. Match to get the name of the module
    # then execute the `.call` function on that module.
    %module{} = chain.llm

    # wrap and link the model's callbacks.
    use_llm = Utils.rewrap_callbacks_for_model(chain.llm, chain.callbacks, chain)

    # filter out any empty lists in the list of messages.
    message_response =
      case module.call(use_llm, chain.messages, chain.tools) do
        {:ok, messages} when is_list(messages) ->
          {:ok, Enum.reject(messages, &(&1 == []))}

        non_list_or_error ->
          non_list_or_error
      end

    # handle and output response
    case message_response do
      {:ok, [%Message{} = message]} ->
        if chain.verbose, do: IO.inspect(message, label: "SINGLE MESSAGE RESPONSE")
        {:ok, process_message(chain, message)}

      {:ok, [%Message{} = message | _others] = messages} ->
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

      {:error, %LangChainError{} = reason} ->
        if chain.verbose, do: IO.inspect(reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(reason)}")
        {:error, chain, reason}

      {:error, string_reason} when is_binary(string_reason) ->
        if chain.verbose, do: IO.inspect(string_reason, label: "ERROR")
        Logger.error("Error during chat call. Reason: #{inspect(string_reason)}")
        {:error, chain, LangChainError.exception(message: string_reason)}
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
  Apply a list of deltas to the chain. When the final delta is received that
  completes the message, the LLMChain is updated to clear the `delta` and the
  `last_message` and list of messages are updated. The message is processed and
  fires any registered callbacks.
  """
  @spec apply_deltas(t(), list()) :: t()
  def apply_deltas(%LLMChain{} = chain, deltas) when is_list(deltas) do
    chain
    |> merge_deltas(deltas)
    |> delta_to_message_when_complete()
  end

  @doc """
  Merge a list of deltas into the chain.
  """
  @spec merge_deltas(t(), list()) :: t()
  def merge_deltas(%LLMChain{} = chain, deltas) do
    deltas
    |> List.flatten()
    |> Enum.reduce(chain, fn d, acc -> merge_delta(acc, d) end)
  end

  @doc """
  Merge a received MessageDelta struct into the chain's current delta. The
  LLMChain tracks the current merged MessageDelta state. This is able to merge
  in TokenUsage received after the final delta.
  """
  @spec merge_delta(t(), MessageDelta.t() | TokenUsage.t() | {:error, LangChainError.t()}) :: t()
  def merge_delta(%LLMChain{} = chain, %MessageDelta{} = new_delta) do
    merged = MessageDelta.merge_delta(chain.delta, new_delta)
    %LLMChain{chain | delta: merged}
  end

  def merge_delta(%LLMChain{} = chain, %TokenUsage{} = usage) do
    # OpenAI returns the token usage in a separate chunk after the last delta. We want to merge it into the final delta.
    fake_delta = MessageDelta.new!(%{role: :assistant, metadata: %{usage: usage}})

    merged = MessageDelta.merge_delta(chain.delta, fake_delta)
    %LLMChain{chain | delta: merged}
  end

  # Handle when the server is overloaded and cancelled the stream on the server side.
  def merge_delta(%LLMChain{} = chain, {:error, %LangChainError{type: "overloaded"}}) do
    cancel_delta(chain, :cancelled)
  end

  @doc """
  Drop the current delta. This is useful when needing to ignore a partial or
  complete delta because the message may be handled in a different way.
  """
  @spec drop_delta(t()) :: t()
  def drop_delta(%LLMChain{} = chain) do
    %LLMChain{chain | delta: nil}
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

  # Process an assistant message sequentially through each message processor.
  @doc false
  @spec run_message_processors(t(), Message.t()) ::
          Message.t() | {:halted, Message.t(), Message.t()}
  def run_message_processors(
        %LLMChain{message_processors: processors} = chain,
        %Message{role: :assistant} = message
      )
      when is_list(processors) and processors != [] do
    # start `processed_content` with the message's content as a string
    message = %Message{message | processed_content: ContentPart.parts_to_string(message.content)}

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
        |> fire_usage_callback_and_return(:on_llm_token_usage, [updated_message])
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
        exchanged_messages: chain.exchanged_messages ++ [new_message],
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
        |> Task.await_many(chain.async_tool_timeout || @default_task_await_timeout)

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
        Message.new_tool_result!(%{content: nil, tool_results: combined_results})

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

      updated_chain
      |> fire_callback_and_return(:on_message_processed, [result_message])
      |> fire_callback_and_return(:on_tool_response_created, [result_message])
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

    metadata = %{
      tool_name: function.name,
      tool_call_id: call.call_id,
      async: function.async
    }

    LangChain.Telemetry.span([:langchain, :tool, :call], metadata, fn ->
      try do
        if verbose, do: IO.inspect(function.name, label: "EXECUTING FUNCTION")

        case Function.execute(function, call.arguments, context) do
          {:ok, %ToolResult{} = result} ->
            # allow the tool execution to return a ToolResult. Just set the
            # tool_call_id and fallback settings for name and display_text. This
            # allows the tool to explicitly set the options for the ToolResult.
            %{
              result
              | tool_call_id: call.call_id,
                name: result.name || function.name,
                display_text: result.display_text || function.display_text
            }

          {:ok, llm_result, processed_result} ->
            if verbose, do: IO.inspect(processed_result, label: "FUNCTION PROCESSED RESULT")
            # successful execution and storage of processed_content.
            ToolResult.new!(%{
              tool_call_id: call.call_id,
              content: llm_result,
              processed_content: processed_result,
              name: function.name,
              display_text: function.display_text
            })

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
          Logger.error(
            "Function #{function.name} failed in execution. Exception: #{LangChainError.format_exception(err, __STACKTRACE__)}"
          )

          ToolResult.new!(%{
            tool_call_id: call.call_id,
            content: "ERROR executing tool: #{inspect(err)}",
            is_error: true
          })
      end
    end)
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

  # a pipe-friendly execution of callbacks that returns the chain
  defp fire_callback_and_return(%LLMChain{} = chain, callback_name, additional_arguments)
       when is_list(additional_arguments) do
    Callbacks.fire(chain.callbacks, callback_name, [chain] ++ additional_arguments)
    chain
  end

  # fire token usage callback in a pipe-friendly function
  defp fire_usage_callback_and_return(
         %LLMChain{} = chain,
         callback_name,
         [%{metadata: %{usage: %TokenUsage{} = usage}}]
       ) do
    Callbacks.fire(chain.callbacks, callback_name, [chain, usage])
    chain
  end

  defp fire_usage_callback_and_return(%LLMChain{} = chain, _callback_name, _additional_arguments),
    do: chain

  defp clear_exchanged_messages(%LLMChain{} = chain) do
    %LLMChain{chain | exchanged_messages: []}
  end

  defp raise_on_obsolete_run_opts(opts) do
    if Keyword.has_key?(opts, :callback_fn) do
      raise LangChainError,
            "The LLMChain.run option `:callback_fn` was removed; see `add_callback/2` instead."
    end
  end

  # Raise an exception when there are no messages in the LLMChain (checked when running)
  defp raise_when_no_messages(%LLMChain{messages: []} = _chain) do
    raise LangChainError, "LLMChain cannot be run without messages"
  end

  defp raise_when_no_messages(%LLMChain{} = chain), do: chain
end
