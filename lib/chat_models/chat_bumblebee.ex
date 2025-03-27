defmodule LangChain.ChatModels.ChatBumblebee do
  @moduledoc """
  Represents a chat model hosted by Bumblebee and accessed through an
  `Nx.Serving`.

  Many types of models can be hosted through Bumblebee, so this attempts to
  represent the most common features and provide a single implementation where
  possible.

  For streaming responses, the Bumblebee serving must be configured with
  `stream: true` and should include `stream_done: true` as well.

  Example:

      Bumblebee.Text.generation(model_info, tokenizer, generation_config,
        # ...
        stream: true,
        stream_done: true
      )

  This supports a non streaming response as well, in which case, a completed
  `LangChain.Message` is returned at the completion.

  The `stream_done` option sends a final message to let us know when the stream
  is complete and includes some token information.

  The chat model can be created like this and provided to an LLMChain:

      ChatBumblebee.new!(%{
        serving: @serving_name,
        template_format: @template_format,
        receive_timeout: @receive_timeout,
        stream: true
      })

  The `serving` is the module name of the `Nx.Serving` that is hosting the
  model.

  The following are the supported values for `template_format`. These are
  provided by `LangChain.Utils.ChatTemplates`.

  Chat models are trained against specific content formats for the messages.
  Some models have no special concept of a system message. See the
  `LangChain.Utils.ChatTemplates` documentation for specific format examples.

  Using the wrong format with a model may result in poor performance or
  hallucinations. It will not result in an error.

  ## Full example of chat through Bumblebee

  Here's a full example of having a streaming conversation with Llama 2 through
  Bumblebee.

      defmodule MyApp.BumblebeeChat do
        @doc false
        alias LangChain.Message
        alias LangChain.ChatModels.ChatBumblebee
        alias LangChain.Chains.LLMChain

        def run_chat do
          # Used when streaming responses. The function fires as data is received.
          callback_fn = fn
            %LangChain.MessageDelta{} = delta ->
              # write to the console as the response is streamed back
              IO.write(delta.content)

            %LangChain.Message{} = message ->
              # inspect the fully finished message that was assembled from all the deltas
              IO.inspect(message, label: "FULLY ASSEMBLED MESSAGE")
          end

          # create and run the chain
          {:ok, _updated_chain, %Message{} = message} =
            LLMChain.new!(%{
              llm:
                ChatBumblebee.new!(%{
                  serving: Llama2ChatModel,
                  template_format: :llama_2,
                  stream: true
                }),
              verbose: true
            })
            |> LLMChain.add_message(Message.new_system!("You are a helpful assistant."))
            |> LLMChain.add_message(Message.new_user!("What is the capital of Taiwan? And share up to 5 interesting facts about the city."))
            |> LLMChain.run(callback_fn: callback_fn)

          # print the LLM's fully assembled answer
          IO.puts("\\n\\n")
          IO.puts(message.content)
          IO.puts("\\n\\n")
        end
      end

  Then run the code in IEx:

        recompile; MyApp.BumblebeeChat.run_chat
  """
  use Ecto.Schema
  require Logger
  import Ecto.Changeset
  alias __MODULE__
  alias LangChain.ChatModels.ChatModel
  alias LangChain.Message
  alias LangChain.Function
  alias LangChain.TokenUsage
  alias LangChain.LangChainError
  alias LangChain.Utils
  alias LangChain.MessageDelta
  alias LangChain.Utils.ChatTemplates
  alias LangChain.Callbacks
  alias LangChain.Message.ToolCall
  alias LangChain.Utils.Parser.LLAMA_3_1_CustomToolParser
  alias LangChain.Utils.Parser.LLAMA_3_2_CustomToolParser

  @behaviour ChatModel

  @current_config_version 1

  @primary_key false
  embedded_schema do
    # Name of the Nx.Serving to use when working with the LLM.
    field :serving, :any, virtual: true

    # # What sampling temperature to use, between 0 and 2. Higher values like 0.8
    # # will make the output more random, while lower values like 0.2 will make it
    # # more focused and deterministic.
    # field :temperature, :float, default: 1.0

    field :template_format, Ecto.Enum,
      values: [
        :inst,
        :im_start,
        :zephyr,
        :phi_4,
        :llama_2,
        :llama_3,
        :llama_3_1_json_tool_calling,
        :llama_3_1_custom_tool_calling,
        :llama_3_2_custom_tool_calling
      ]

    # The bumblebee model may compile differently based on the stream true/false
    # option on the serving. Therefore, streaming should be enabled on the
    # serving and a stream option here can change the way data is received in
    # code. - https://github.com/elixir-nx/bumblebee/issues/295

    field :stream, :boolean, default: true

    # Seed for randomizing behavior or giving more deterministic output. Helpful
    # for testing.
    field :seed, :integer, default: nil

    # A list of maps for callback handlers (treat as private)
    field :callbacks, {:array, :map}, default: []
  end

  @type t :: %ChatBumblebee{}

  # @type call_response :: {:ok, Message.t() | [Message.t()]} | {:error, String.t()}
  # @type callback_data ::
  #         {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
  #         | {:error, String.t()}
  @type callback_fn :: (Message.t() | MessageDelta.t() -> any())

  @create_fields [
    :serving,
    # :temperature,
    :seed,
    :template_format,
    :stream
  ]
  @required_fields [:serving]

  @doc """
  Setup a ChatBumblebee client configuration.
  """
  @spec new(attrs :: map()) :: {:ok, t} | {:error, Ecto.Changeset.t()}
  def new(%{} = attrs \\ %{}) do
    %ChatBumblebee{}
    |> cast(attrs, @create_fields)
    |> restore_serving_if_string()
    |> common_validation()
    |> apply_action(:insert)
  end

  @doc """
  Setup a ChatBumblebee client configuration and return it or raise an error if invalid.
  """
  @spec new!(attrs :: map()) :: t() | no_return()
  def new!(attrs \\ %{}) do
    case new(attrs) do
      {:ok, chain} ->
        chain

      {:error, %Ecto.Changeset{} = changeset} ->
        raise LangChainError.exception(changeset)
    end
  end

  defp restore_serving_if_string(changeset) do
    case get_field(changeset, :serving) do
      value when is_binary(value) ->
        case Utils.module_from_name(value) do
          {:ok, module} ->
            put_change(changeset, :serving, module)

          {:error, reason} ->
            add_error(changeset, :serving, reason)
        end

      _other ->
        changeset
    end
  end

  defp common_validation(changeset) do
    changeset
    |> validate_required(@required_fields)
  end

  @impl ChatModel
  def call(model, prompt, functions \\ [])

  def call(%ChatBumblebee{} = model, prompt, functions) when is_binary(prompt) do
    messages = [
      Message.new_system!(),
      Message.new_user!(prompt)
    ]

    call(model, messages, functions)
  end

  def call(%ChatBumblebee{} = model, messages, functions) when is_list(messages) do
    metadata = %{
      model: inspect(model.serving),
      template_format: model.template_format,
      message_count: length(messages),
      tools_count: length(functions)
    }

    LangChain.Telemetry.span([:langchain, :llm, :call], metadata, fn ->
      try do
        # Track the prompt being sent
        LangChain.Telemetry.llm_prompt(
          %{system_time: System.system_time()},
          %{model: inspect(model.serving), messages: messages}
        )

        # make base api request and perform high-level success/failure checks
        case do_serving_request(model, messages, functions) do
          {:error, reason} ->
            {:error, reason}

          parsed_data ->
            # Track the response being received
            LangChain.Telemetry.llm_response(
              %{system_time: System.system_time()},
              %{model: inspect(model.serving), response: parsed_data}
            )

            {:ok, parsed_data}
        end
      rescue
        err in LangChainError ->
          {:error, err}
      end
    end)
  end

  @doc false
  @spec do_serving_request(t(), [Message.t()], [Function.t()]) ::
          list() | struct() | {:error, String.t()}

  def do_serving_request(
        %ChatBumblebee{template_format: :llama_3_1_json_tool_calling} = model,
        messages,
        functions
      ) do
    prompt =
      ChatTemplates.apply_chat_template_with_tools!(messages, model.template_format, functions)

    model.serving
    |> Nx.Serving.batched_run(%{text: prompt, seed: model.seed})
    |> do_process_response(model)
  end

  def do_serving_request(
        %ChatBumblebee{template_format: :llama_3_1_custom_tool_calling} = model,
        messages,
        functions
      ) do
    prompt =
      ChatTemplates.apply_chat_template_with_tools!(messages, model.template_format, functions)

    model.serving
    |> Nx.Serving.batched_run(%{text: prompt, seed: model.seed})
    |> do_process_response(model)
  end

  def do_serving_request(
        %ChatBumblebee{template_format: :llama_3_2_custom_tool_calling} = model,
        messages,
        functions
      ) do
    prompt =
      ChatTemplates.apply_chat_template_with_tools!(messages, model.template_format, functions)

    model.serving
    |> Nx.Serving.batched_run(%{text: prompt, seed: model.seed})
    |> do_process_response(model)
  end

  def do_serving_request(%ChatBumblebee{} = model, messages, _functions) do
    prompt = ChatTemplates.apply_chat_template!(messages, model.template_format)

    model.serving
    |> Nx.Serving.batched_run(%{text: prompt, seed: model.seed})
    |> do_process_response(model)
  end

  def do_process_response(
        %{results: [%{text: "[" <> _ = content, token_summary: token_summary}]},
        %ChatBumblebee{template_format: :llama_3_2_custom_tool_calling} = model
      )
      when is_binary(content) do
    if !Code.ensure_loaded?(NimbleParsec) do
      raise "Install NimbleParsec to use custom tool calling"
    end

    fire_token_usage_callback(model, token_summary)

    case LLAMA_3_2_CustomToolParser.parse(content) do
      {:ok, functions} ->
        case Message.new(%{
               role: :assistant,
               status: :complete,
               content: content,
               tool_calls:
                 Enum.with_index(functions, fn i,
                                               %{
                                                 function_name: name,
                                                 parameters: parameters
                                               } ->
                   ToolCall.new!(%{
                     call_id: Integer.to_string(i),
                     name: name,
                     arguments: parameters
                   })
                 end)
             }) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, _} ->
        case Message.new(%{role: :assistant, status: :complete, content: content}) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end
    end
  end

  def do_process_response(
        %{results: [%{text: "<" <> _ = content, token_summary: token_summary}]},
        %ChatBumblebee{template_format: :llama_3_1_custom_tool_calling} = model
      )
      when is_binary(content) do
    if !Code.ensure_loaded?(NimbleParsec) do
      raise "Install NimbleParsec to use custom tool calling"
    end

    fire_token_usage_callback(model, token_summary)

    case LLAMA_3_1_CustomToolParser.parse(content) do
      {:ok,
       %{
         function_name: name,
         parameters: parameters
       }} ->
        case Message.new(%{
               role: :assistant,
               status: :complete,
               content: content,
               tool_calls: [ToolCall.new!(%{call_id: "test", name: name, arguments: parameters})]
             }) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, _} ->
        case Message.new(%{role: :assistant, status: :complete, content: content}) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end
    end
  end

  @doc false
  def do_process_response(
        %{results: [%{text: "{" <> _ = content, token_summary: token_summary}]},
        %ChatBumblebee{template_format: :llama_3_1_json_tool_calling} = model
      )
      when is_binary(content) do
    fire_token_usage_callback(model, token_summary)

    case Jason.decode(content) do
      {:ok,
       %{
         "name" => name,
         "parameters" => parameters
       }} ->
        case Message.new(%{
               role: :assistant,
               status: :complete,
               content: content,
               tool_calls: [ToolCall.new!(%{call_id: "test", name: name, arguments: parameters})]
             }) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end

      {:error, _} ->
        case Message.new(%{role: :assistant, status: :complete, content: content}) do
          {:ok, message} ->
            # execute the callback with the final message
            Callbacks.fire(model.callbacks, :on_llm_new_message, [model, message])
            # return a list of the complete message. As a list for compatibility.
            [message]

          {:error, changeset} ->
            reason = Utils.changeset_error_to_string(changeset)
            Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
            {:error, reason}
        end
    end
  end

  def do_process_response(
        %{results: [%{text: content, token_summary: token_summary}]},
        %ChatBumblebee{} = model
      )
      when is_binary(content) do
    fire_token_usage_callback(model, token_summary)

    # Track non-streaming response completion
    LangChain.Telemetry.emit_event(
      [:langchain, :llm, :response, :non_streaming],
      %{system_time: System.system_time()},
      %{
        model: inspect(model.serving),
        response_size: byte_size(inspect(content))
      }
    )

    case Message.new(%{role: :assistant, status: :complete, content: content}) do
      {:ok, message} ->
        # execute the callback with the final message
        Callbacks.fire(model.callbacks, :on_llm_new_message, [message])
        # return a list of the complete message. As a list for compatibility.
        [message]

      {:error, %Ecto.Changeset{} = changeset} ->
        reason = Utils.changeset_error_to_string(changeset)
        Logger.error("Failed to create non-streamed full message: #{inspect(reason)}")
        {:error, LangChainError.exception(changeset)}
    end
  end

  def do_process_response(stream, %ChatBumblebee{stream: false} = model) do
    # Request is to NOT stream. Consume the full stream and format the data as
    # though it had not been streamed.
    full_data =
      Enum.reduce(stream, %{text: "", token_summary: nil}, fn
        {:done, %{token_summary: token_data}}, %{text: text} ->
          %{text: text, token_summary: token_data}

        data, %{text: text} = acc ->
          Map.put(acc, :text, text <> data)
      end)

    do_process_response(%{results: [full_data]}, model)
  end

  def do_process_response(stream, %ChatBumblebee{} = model) do
    chunk_processor = fn
      {:done, %{token_summary: token_summary}} ->
        fire_token_usage_callback(model, token_summary)

        # Track stream completion
        LangChain.Telemetry.emit_event(
          [:langchain, :llm, :response, streaming: true],
          %{system_time: System.system_time()},
          %{model: inspect(model.serving)}
        )

        final_delta = MessageDelta.new!(%{role: :assistant, status: :complete})
        Callbacks.fire(model.callbacks, :on_llm_new_delta, [final_delta])
        final_delta

      content when is_binary(content) ->
        case MessageDelta.new(%{content: content, role: :assistant, status: :incomplete}) do
          {:ok, delta} ->
            Callbacks.fire(model.callbacks, :on_llm_new_delta, [delta])
            delta

          {:error, %Ecto.Changeset{} = changeset} ->
            reason = Utils.changeset_error_to_string(changeset)

            Logger.error(
              "Failed to process received model's MessageDelta data: #{inspect(reason)}"
            )

            raise LangChainError.exception(changeset)
        end
    end

    result =
      stream
      |> Stream.map(&chunk_processor.(&1))
      |> Enum.to_list()

    # return a list of a list to mirror the way ChatGPT returns data
    [result]
  end

  defp fire_token_usage_callback(model, %{input: input, output: output} = token_summary) do
    Callbacks.fire(model.callbacks, :on_llm_token_usage, [
      TokenUsage.new!(%{input: input, output: output, raw: token_summary})
    ])
  end

  defp fire_token_usage_callback(_model, _token_summary), do: :ok

  @doc """
  Generate a config map that can later restore the model's configuration.
  """
  @impl ChatModel
  @spec serialize_config(t()) :: %{String.t() => any()}
  def serialize_config(%ChatBumblebee{} = model) do
    Utils.to_serializable_map(
      model,
      [
        :serving,
        :template_format,
        :stream,
        :seed
      ],
      @current_config_version
    )
  end

  @doc """
  Restores the model from the config.
  """
  @impl ChatModel
  def restore_from_map(%{"version" => 1} = data) do
    ChatBumblebee.new(data)
  end
end
