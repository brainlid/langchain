defmodule LangChain.Chains.DataExtractionChain do
  @moduledoc """
  Defines an LLMChain for performing data extraction from a body of text.

  Provide the schema for desired information to be parsed into. It is treated as
  though there are 0 to many instances of the data structure being described so
  information is returned as an array.

  The result is always a list. If the LLM returns a single map instead of an
  array, it is automatically wrapped in a list so callers can rely on a
  consistent return type.

  Originally based on:
  - https://github.com/langchain-ai/langchainjs/blob/main/langchain/src/chains/openai_functions/extraction.ts#L43

  ## Example

      # JSONSchema definition of data we want to capture or extract.
      schema_parameters = %{
        type: "object",
        properties: %{
          person_name: %{type: "string"},
          person_age: %{type: "number"},
          person_hair_color: %{type: "string"},
          dog_name: %{type: "string"},
          dog_breed: %{type: "string"}
        },
        required: []
      }

      # Model setup
      {:ok, chat} = ChatOpenAI.new(%{temperature: 0})

      # run the chain on the text information
      data_prompt =
        "Alex is 5 feet tall. Claudia is 4 feet taller than Alex and jumps higher than him.
        Claudia is a brunette and Alex is blonde. Alex's dog Frosty is a labrador and likes to play hide and seek."

      {:ok, result} = LangChain.Chains.DataExtractionChain.run(chat, schema_parameters, data_prompt)

      # Example result
      [
        %{
          "dog_breed" => "labrador",
          "dog_name" => "Frosty",
          "person_age" => nil,
          "person_hair_color" => "blonde",
          "person_name" => "Alex"
        },
        %{
          "dog_breed" => nil,
          "dog_name" => nil,
          "person_age" => nil,
          "person_hair_color" => "brunette",
          "person_name" => "Claudia"
        }
      ]

  If the LLM returns a single map (e.g. when only one entity is found), it is
  wrapped in a list automatically:

      # Single-entity result normalised to a list
      [
        %{
          "person_name" => "Alex",
          "person_age" => nil,
          ...
        }
      ]

  ## Accessing the LLMChain

  `run/4` returns only the extracted data. When more than the data is needed,
  for instance to log or report the token usage of the extraction, use
  `run_chain/4` to get the executed `LangChain.Chains.LLMChain` and
  `extract_result/1` to pull the data out of it:

      {:ok, chain} = LangChain.Chains.DataExtractionChain.run_chain(chat, schema_parameters, data_prompt)

      usage = LangChain.TokenUsage.get(chain.last_message)
      {:ok, result} = LangChain.Chains.DataExtractionChain.extract_result(chain)

  ## Callbacks

  The `LLMChain` used for the extraction is built internally, so handlers
  registered on the `llm` itself are not used. Pass `:callbacks` to observe the
  run as it happens, which is the only way to see streamed deltas:

      LangChain.Chains.DataExtractionChain.run(chat, schema_parameters, data_prompt,
        callbacks: [%{on_llm_token_usage: fn _chain, usage -> log_usage(usage) end}]
      )

  Handlers are registered on the internally run `LLMChain`, so the full set of
  `LangChain.Chains.ChainCallbacks` events is available.

  The `schema_parameters` in the previous example can also be expressed using a
  list of `LangChain.FunctionParam` structs. An equivalent version looks like
  this:

      alias LangChain.FunctionParam

      schema_parameters = [
        FunctionParam.new!(%{name: "person_name", type: :string}),
        FunctionParam.new!(%{name: "person_age", type: :number}),
        FunctionParam.new!(%{name: "person_hair_color", type: :string}),
        FunctionParam.new!(%{name: "dog_name", type: :string}),
        FunctionParam.new!(%{name: "dog_breed", type: :string})
      ]
      |> FunctionParam.to_parameters_schema()

  ## Provider Strategy vs. Tool Strategy

  This chain can ask the LLM to return JSON constrained to a schema natively
  ("structured outputs"), without any tool/function calling. The chain checks
  whether `llm`'s struct type natively supports this (see
  `LangChain.ChatModels.ChatModel.supports_json_output?/1`) — i.e. whether its
  bare struct defines both `:json_schema` and `:json_response` fields. If it
  does, the chain runs against a patched copy of `llm` with `json_response:
  true` and `json_schema` set to `schema_parameters` as given, and parses the
  plain JSON response directly with `extract_result/1` instead of requiring a
  tool call. Some models require extra parameters alongside the schema itself
  (e.g. `LangChain.ChatModels.ChatOpenAIResponses` requires a separate
  `:json_schema_name`); the chain fills these in with sensible defaults when
  the struct defines them.

  When `opts[:strategy]` is **not given**, the chain defaults to
  `:provider_strategy`, but fails gracefully: if `llm`'s struct type doesn't
  support it, it falls back to `:tool_strategy` — asking the LLM to call an
  `information_extraction` tool built from `schema_parameters`, as shown
  above.

  When `opts[:strategy]` **is given** explicitly, it is used strictly — there is
  no fallback. Passing `strategy: :provider_strategy` for an `llm` whose struct
  type doesn't support it raises from `run_chain/4` (and `run/4` will return an
  `{:error, %LangChainError{}}`):

      # Always uses tool calling, regardless of what `llm` supports
      {:ok, result} =
        DataExtractionChain.run(chat, schema_parameters, data_prompt, strategy: :tool_strategy)

      # Strictly requires provider_strategy support; raises from run_chain/4 otherwise
      {:ok, chain} =
        DataExtractionChain.run_chain(chat, schema_parameters, data_prompt, strategy: :provider_strategy)

  Note: the exact wire format for structured output still differs by provider
  (compare `:json_schema` on `LangChain.ChatModels.ChatAnthropic` vs
  `LangChain.ChatModels.ChatMistralAI`, which expects the entire response
  format nested under `:json_schema`); the defaults applied here favor the
  common bare-schema shape and may not be fully correct for every provider.

  """
  use Ecto.Schema
  require Logger
  alias LangChain.PromptTemplate
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.LangChainError
  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatModel
  alias LangChain.MessageProcessors.JsonProcessor

  @function_name "information_extraction"
  @extraction_template ~s"Extract and save the relevant entities mentioned in the following passage together with their properties. Use the value `null` when missing in the passage.

Passage:
<%= @input %>"

  # Sensible defaults applied to extra fields some models require alongside
  # `:json_schema`/`:json_response`, when the struct defines them and they're
  # not already set. Only fields explicitly known to exist across the chat
  # model modules are listed here.
  @extra_field_defaults %{json_schema_name: @function_name}

  @doc """
  Coerces the extraction tool's `info` argument to a list of rows.

  Models sometimes return one JSON object instead of a one-element array; `run/4`
  uses this so callers always get `{:ok, list}`.
  """
  @spec normalize_extraction_info(term()) :: {:ok, [any()]} | {:error, LangChainError.t()}
  def normalize_extraction_info(info) when is_list(info), do: {:ok, info}

  def normalize_extraction_info(info) when is_map(info), do: {:ok, [info]}

  def normalize_extraction_info(other) do
    {:error,
     LangChainError.exception("Extracted data must be a list or map, got: #{inspect(other)}")}
  end

  @doc """
  Run the data extraction chain and return the executed `LangChain.Chains.LLMChain`.

  Use this instead of `run/4` when the chain itself is needed and not just the
  extracted data. The chain gives access to the returned messages, token usage,
  and everything else recorded during execution.

      {:ok, chain} = DataExtractionChain.run_chain(chat, schema_parameters, data_prompt)

      # inspect the token usage of the extraction
      usage = LangChain.TokenUsage.get(chain.last_message)

      # get the extracted data from the chain
      {:ok, result} = DataExtractionChain.extract_result(chain)

  Follows the same return pattern as `LangChain.Chains.LLMChain.run/2`.

  ## Options

  - `:strategy` - either `:provider_strategy` or `:tool_strategy`. When
    omitted, defaults to `:provider_strategy` but falls back to
    `:tool_strategy` if `llm` doesn't support it. When given explicitly, it is
    used strictly, raising if `llm` doesn't support it. See the "Provider
    Strategy vs. Tool Strategy" section in the module docs.
  - `:verbose` - when `true`, enables verbose logging on the internally run
    `LLMChain`. Defaults to `false`.
  - `:callbacks` - a list of callback handler maps to register on the
    internally run `LLMChain`. See `LangChain.Chains.ChainCallbacks` for the
    available events. Defaults to `[]`.
  """
  @spec run_chain(ChatModel.t(), json_schema :: map(), prompt :: [any()], opts :: Keyword.t()) ::
          {:ok, LLMChain.t()} | {:error, LLMChain.t(), LangChainError.t()}
  def run_chain(llm, json_schema, prompt, opts \\ []) do
    case Keyword.fetch(opts, :strategy) do
      :error ->
        # No explicit strategy: default to provider_strategy, but fail
        # gracefully by falling back to tool_strategy when unsupported.
        if ChatModel.supports_json_output?(llm.__struct__) do
          run_chain_provider_strategy(llm, json_schema, prompt, opts)
        else
          Logger.warning(
            "#{inspect(llm.__struct__)} does not support :provider_strategy (it does not define both :json_schema and :json_response fields). Falling back to :tool_strategy since no :strategy was explicitly specified."
          )

          run_chain_tool_strategy(llm, json_schema, prompt, opts)
        end

      {:ok, :tool_strategy} ->
        run_chain_tool_strategy(llm, json_schema, prompt, opts)

      {:ok, :provider_strategy} ->
        # Explicit strategy: used strictly, no fallback.
        if ChatModel.supports_json_output?(llm.__struct__) do
          run_chain_provider_strategy(llm, json_schema, prompt, opts)
        else
          raise LangChainError,
                "`llm`'s struct type does not support :provider_strategy"
        end

      {:ok, other} ->
        raise LangChainError,
              "Invalid :strategy #{inspect(other)}. Expected :tool_strategy or :provider_strategy."
    end
  end

  defp run_chain_tool_strategy(llm, json_schema, prompt, opts) do
    verbose = Keyword.get(opts, :verbose, false)
    callbacks = Keyword.get(opts, :callbacks, [])

    messages =
      [
        Message.new_system!(
          "You are a helpful assistant that extracts structured data from text passages. Only use the functions you have been provided with. Extract the data in a single tool use."
        ),
        PromptTemplate.new!(%{role: :user, text: @extraction_template})
      ]
      |> PromptTemplate.to_messages!(%{input: prompt})

    %{llm: llm, verbose: verbose, callbacks: callbacks}
    |> LLMChain.new!()
    |> LLMChain.add_tools(build_extract_function(json_schema))
    |> LLMChain.add_messages(messages)
    |> LLMChain.run()
  end

  defp run_chain_provider_strategy(llm, json_schema, prompt, opts) do
    verbose = Keyword.get(opts, :verbose, false)
    callbacks = Keyword.get(opts, :callbacks, [])

    messages =
      [
        Message.new_system!(
          "You are a helpful assistant that extracts structured data from text passages. Respond only with JSON matching the required schema. Use the value `null` when missing in the passage."
        ),
        PromptTemplate.new!(%{role: :user, text: @extraction_template})
      ]
      |> PromptTemplate.to_messages!(%{input: prompt})

    %{
      llm: patch_llm_for_provider_strategy(llm, json_schema),
      verbose: verbose,
      callbacks: callbacks
    }
    |> LLMChain.new!()
    |> LLMChain.message_processors([JsonProcessor.new!()])
    |> LLMChain.add_messages(messages)
    |> LLMChain.run()
  end

  # Patches a copy of `llm` with the fields needed to request structured JSON
  # output for `json_schema`, unwrapped and as given. The schema is always
  # taken from the argument passed to this call, never from how `llm` happened
  # to be constructed.
  defp patch_llm_for_provider_strategy(llm, json_schema) do
    llm
    |> Map.put(:json_response, true)
    |> Map.put(:json_schema, json_schema)
    |> apply_extra_field_defaults()
  end

  defp apply_extra_field_defaults(llm) do
    Enum.reduce(@extra_field_defaults, llm, fn {field, default}, acc ->
      if Map.has_key?(acc, field) and is_nil(Map.get(acc, field)) do
        Map.put(acc, field, default)
      else
        acc
      end
    end)
  end

  @doc """
  Return the extracted data from an executed `LangChain.Chains.LLMChain` that
  was run by `run_chain/4`.

  Under `:tool_strategy`, this reads the `info` array from the extraction
  tool call. Under `:provider_strategy`, the response isn't wrapped in an
  `info` envelope, so this instead takes whatever JSON the LLM returned
  (list or map, matching `json_schema` as given) and normalizes it via
  `normalize_extraction_info/1` — this works regardless of the shape of
  `json_schema` passed to `run_chain/4`.

  Returns an error when the LLM did not respond with the expected extraction
  tool call, or (under `:provider_strategy`) valid JSON. When `JsonProcessor`
  halted on invalid JSON, that corrective error message is surfaced directly
  instead of a generic "unexpected response" message.
  """
  @spec extract_result(LLMChain.t()) :: {:ok, result :: [any()]} | {:error, LangChainError.t()}
  def extract_result(%LLMChain{
        last_message: %Message{
          role: :assistant,
          tool_calls: [
            %ToolCall{
              name: @function_name,
              arguments: %{"info" => info}
            }
          ]
        }
      }) do
    normalize_extraction_info(info)
  end

  def extract_result(%LLMChain{
        last_message: %Message{role: :assistant, processed_content: processed_content}
      })
      when is_list(processed_content) or is_map(processed_content) do
    normalize_extraction_info(processed_content)
  end

  # Assuming there was no last message. the extraction did not work due to invalid json schema
  # we propagate the error forward.
  def extract_result(%LLMChain{
        last_message: %Message{role: :user, content: content}
      }) do
    case ContentPart.content_to_string(content) do
      "ERROR: " <> _ = error_text -> {:error, LangChainError.exception(error_text)}
      _ -> {:error, LangChainError.exception("Unexpected response.")}
    end
  end

  def extract_result(%LLMChain{} = chain) do
    {:error, LangChainError.exception("Unexpected response. #{inspect(chain.last_message)}")}
  end

  @doc """
  Run the data extraction chain and return the extracted data.

  When the executed chain is needed as well, for instance to report token usage,
  use `run_chain/4` with `extract_result/1`.

  Accepts the same options as `run_chain/4`.
  """
  @spec run(ChatModel.t(), json_schema :: map(), prompt :: [any()], opts :: Keyword.t()) ::
          {:ok, result :: [any()]} | {:error, LangChainError.t()}
  def run(llm, json_schema, prompt, opts \\ []) do
    try do
      case run_chain(llm, json_schema, prompt, opts) do
        {:ok, %LLMChain{} = chain} ->
          extract_result(chain)

        other ->
          {:error, LangChainError.exception("Unexpected response. #{inspect(other)}")}
      end
    rescue
      exception ->
        Logger.warning(
          "Caught unexpected exception in DataExtractionChain. Error: #{inspect(exception)}"
        )

        {:error,
         LangChainError.exception(
           "Unexpected error in DataExtractionChain. Check logs for details."
         )}
    end
  end

  @doc """
  Build the function to expose to the LLM that can be called for data
  extraction.
  """
  @spec build_extract_function(json_schema :: map()) :: LangChain.Function.t() | no_return()
  def build_extract_function(json_schema) do
    LangChain.Function.new!(%{
      name: @function_name,
      description: "Extracts the relevant information from the passage.",
      function: fn args, _context ->
        # NOTE: The function is not executed here because we won't be returning
        # this to the LLM. The LLMChain does not run the function, but stops at
        # the request for it.
        {:ok, args}
      end,
      parameters_schema: %{
        type: "object",
        properties: %{
          info: %{
            type: "array",
            items: json_schema
          }
        },
        required: ["info"]
      }
    })
  end
end
