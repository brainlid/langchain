defmodule LangChain.Utils do
  @moduledoc """
  Collection of helpful utilities mostly for internal use.
  """
  alias LangChain.LangChainError
  alias Ecto.Changeset
  alias LangChain.Callbacks
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  require Logger

  @doc """
  Only add the key to the map if the value is present. When the value is a list,
  the key will not be added when the list is empty. If the value is `nil`, it
  will not be added.
  """
  @spec conditionally_add_to_map(%{any() => any()}, key :: any(), value :: any()) :: %{
          any() => any()
        }
  def conditionally_add_to_map(map, key, value)

  def conditionally_add_to_map(map, _key, nil), do: map

  def conditionally_add_to_map(map, _key, []), do: map

  def conditionally_add_to_map(map, key, value) do
    Map.put(map, key, value)
  end

  # Generate wrapped LLM callbacks on the model that include the chain as part
  # of the context.
  @doc false
  @spec rewrap_callbacks_for_model(
          llm :: struct(),
          callbacks :: [%{atom() => fun()}],
          context :: struct()
        ) :: struct()
  def rewrap_callbacks_for_model(llm, callbacks, context) do
    to_wrap = [
      :on_llm_new_delta,
      :on_llm_new_message,
      :on_llm_ratelimit_info,
      :on_llm_token_usage,
      :on_llm_response_headers,
      :on_llm_reasoning_delta
    ]

    tool_map = Map.get(context, :_tool_map) || %{}

    # get the LLM callbacks from the chain.
    new_callbacks =
      callbacks
      |> Enum.map(fn callback_map ->
        callback_map
        |> Map.take(to_wrap)
        |> Enum.map(fn
          # For :on_llm_new_delta, augment tool calls with display_text before
          # passing to the callback so consumers get enriched deltas during
          # streaming (not only after post-streaming processing).
          {:on_llm_new_delta, fun} when tool_map != %{} ->
            {:on_llm_new_delta,
             fn deltas ->
               fun.(context, augment_delta_display_text(deltas, tool_map))
             end}

          {key, fun} ->
            # return a wrapped/curried function that embeds the chain context into
            # the call
            {key, fn arg -> fun.(context, arg) end}
        end)
        |> Enum.into(%{})
      end)

    # put those onto the model and return it
    %{llm | callbacks: new_callbacks}
  end

  @doc """
  Translates an error message using gettext.
  """
  def translate_error({msg, opts}) do
    # When using gettext, we typically pass the strings we want
    # to translate as a static argument:
    #
    #     # Translate the number of files with plural rules
    #     dngettext("errors", "1 file", "%{count} files", count)
    #
    # However the error messages in our forms and APIs are generated
    # dynamically, so we need to translate them by calling Gettext
    # with our gettext backend as first argument. Translations are
    # available in the errors.po file (as we use the "errors" domain).
    if count = opts[:count] do
      Gettext.dngettext(LangChain.Gettext, "errors", msg, msg, count, opts)
    else
      Gettext.dgettext(LangChain.Gettext, "errors", msg, opts)
    end
  end

  @doc """
  Translates the errors for a field from a keyword list of errors.
  """
  def translate_errors(errors, field) when is_list(errors) do
    for {^field, {msg, opts}} <- errors, do: translate_error({msg, opts})
  end

  @doc """
  Return changeset errors as text with comma separated description.
  """
  @spec changeset_error_to_string(Ecto.Changeset.t()) :: nil | String.t()
  def changeset_error_to_string(%Ecto.Changeset{valid?: true}), do: nil

  def changeset_error_to_string(%Ecto.Changeset{valid?: false} = changeset) do
    fields = changeset.errors |> Keyword.keys() |> Enum.uniq()

    fields
    |> Enum.reduce([], fn f, acc ->
      field_errors =
        changeset.errors
        |> translate_errors(f)
        |> Enum.join(", ")

      acc ++ ["#{f}: #{field_errors}"]
    end)
    |> Enum.join("; ")
  end

  @doc """
  Validation helper. Validates a struct changeset that the LLM is a struct.
  """
  @spec validate_llm_is_struct(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  def validate_llm_is_struct(changeset) do
    case Changeset.get_change(changeset, :llm) do
      nil -> changeset
      llm when is_struct(llm) -> changeset
      _other -> Changeset.add_error(changeset, :llm, "LLM must be a struct")
    end
  end

  @type callback_data :: Message.t() | [MessageDelta.t()] | TokenUsage.t() | {:error, String.t()}

  @doc """
  Fire a streaming callback if present.
  """
  @spec fire_streamed_callback(
          %{optional(:stream) => boolean(), callbacks: [map()]},
          data :: callback_data() | [callback_data()]
        ) :: :ok | no_return()

  def fire_streamed_callback(model, deltas) when is_list(deltas) do
    #
    # Wrap in a another list for being sent as "args" in an MFA call
    Callbacks.fire(model.callbacks, :on_llm_new_delta, [deltas])
  end

  # received unexpected data in the callback, do nothing.
  def fire_streamed_callback(_model, other) do
    Logger.warning("Received unexpected data in the streamed callback: #{inspect(other)}")
    :ok
  end

  @doc """
  Creates and returns an anonymous function to handle the streaming response
  from an API.

  Accepts the following functions that handle the API-specific requirements:

  - `decode_stream_fn` - a function that parses the raw results from an API. It
    deals with the specifics or oddities of a data source. The results come back
    as `{[list_of_parsed_json_maps], "incomplete text to buffer"}`. In some
    cases, a API may span the JSON data response across messages. This function
    assembles what is complete and returns any incomplete portion that is passed
    in on the next iteration of the function.

  - `transform_data_fn` - a function that is executed to process the parsed
    JSON data in the form of an Elixir map into a LangChain struct of the
    appropriate type.

  - `callback_fn` - a function that receives a successful result of from the
    `transform_data_fn`.
  """
  @spec handle_stream_fn(
          %{optional(:stream) => boolean()},
          decode_stream_fn :: function(),
          transform_data_fn :: function()
        ) :: function()
  def handle_stream_fn(model, decode_stream_fn, transform_data_fn) do
    fn
      {:data, raw_data}, {req, %Req.Response{status: 200} = response} ->
        # Fetch any previously incomplete messages that are buffered in the
        # response struct and pass that in with the data for decode.
        buffered = Req.Response.get_private(response, :lang_incomplete, "")

        if model.verbose_api do
          IO.inspect(raw_data, label: "RCVD RAW CHUNK")
        end

        # decode the received stream data
        {parsed_data, incomplete} =
          decode_stream_fn.({raw_data, buffered})

        if model.verbose_api do
          IO.inspect(parsed_data, label: "READY TO PROCESS")
        end

        # transform what was fully received into MessageDelta structs, that are
        # filtered, then merged together to be processed
        parsed_data =
          parsed_data
          |> Enum.map(transform_data_fn)
          |> Enum.reject(&(&1 == :skip))
          |> List.flatten()

        # execute the callback function for the MessageDeltas and an optional
        # TokenUsage
        fire_streamed_callback(model, parsed_data)
        old_body = if response.body == "", do: [], else: response.body

        # Returns %Req.Response{} where the body contains ALL the stream delta
        # chunks converted to MessageDelta structs. The body is a list of deltas like this...
        #
        # body: [
        #         %LangChain.MessageDelta{
        #           content: nil,
        #           index: 0,
        #           function_name: nil,
        #           role: :assistant,
        #           arguments: nil,
        #           complete: false
        #         },
        #         ...
        #       ]
        #
        # The reason for the inner list is for each entry in the "n" choices. By default only 1.
        updated_response = %{response | body: old_body ++ [parsed_data]}
        # write any incomplete portion to the response's private data for when
        # more data is received.
        updated_response =
          Req.Response.put_private(updated_response, :lang_incomplete, incomplete)

        {:cont, {req, updated_response}}

      {:data, _raw_data}, {req, %Req.Response{status: 401} = _response} ->
        Logger.error("Check API key settings. Request rejected for authentication failure.")
        {:halt, {req, LangChainError.exception("Authentication failure with request")}}

      {:data, raw_data}, {req, %Req.Response{status: status} = response}
      when status in 400..599 ->
        # Buffer error response data across chunks since the JSON may span
        # multiple chunks. Try to decode the accumulated data on each chunk.
        buffered = Req.Response.get_private(response, :error_buffer, "")
        combined = buffered <> raw_data

        case Jason.decode(combined) do
          {:ok, data} ->
            {:halt, {req, %{response | body: transform_data_fn.(data)}}}

          {:error, _reason} ->
            # JSON is incomplete, keep buffering
            updated_response = Req.Response.put_private(response, :error_buffer, combined)
            {:cont, {req, updated_response}}
        end

      {:data, _raw_data}, {req, response} ->
        Logger.error("Unhandled API response!")
        {:halt, {req, response}}
    end
  end

  @doc """
  Extract a structured error from a non-200 streaming response.

  When `handle_stream_fn/3` processes error responses, it buffers the raw JSON
  in the response's private `:error_buffer` key. This function attempts to
  extract a parsed error map from either the response body (if already decoded)
  or the buffered data.

  Returns `{:ok, parsed_map}` with the decoded JSON map, or `:not_found`.

  The caller is responsible for converting the parsed map into the appropriate
  error struct (e.g. `LangChainError`), since the error structure varies by
  provider.
  """
  @spec extract_stream_error(Req.Response.t()) :: {:ok, map()} | :not_found
  def extract_stream_error(%Req.Response{} = response) do
    buffer = Req.Response.get_private(response, :error_buffer, nil)

    cond do
      is_map(response.body) && response.body != %{} ->
        {:ok, response.body}

      is_binary(buffer) ->
        case Jason.decode(buffer) do
          {:ok, data} when is_map(data) -> {:ok, data}
          _ -> :not_found
        end

      true ->
        :not_found
    end
  end

  @doc """
  Put the value in the list at the desired index. If the index does not exist,
  return an updated list where it now exists with the value in that index.
  """
  @spec put_in_list([any()], integer(), any()) :: [any()]
  def put_in_list(list, index, value) do
    if index > Enum.count(list) - 1 do
      list ++ [value]
    else
      List.replace_at(list, index, value)
    end
  end

  @doc """
  Given a struct, create a map with the selected keys converted to strings.
  Additionally includes a `version` number for the data.
  """
  @spec to_serializable_map(struct(), keys :: [atom()], version :: integer()) :: %{
          String.t() => any()
        }
  def to_serializable_map(%module{} = struct, keys, version \\ 1) do
    struct
    |> Map.from_struct()
    |> Map.take(keys)
    |> stringify_keys()
    |> Map.put("module", Atom.to_string(module))
    |> Map.put("version", version)
  end

  @doc """
  Convert map atom keys to strings

  Original source: https://gist.github.com/kipcole9/0bd4c6fb6109bfec9955f785087f53fb
  """
  def stringify_keys(nil), do: nil

  # Handle structs by converting them to maps first
  def stringify_keys(%{__struct__: _} = struct) do
    struct
    |> Map.from_struct()
    |> stringify_keys()
  end

  def stringify_keys(map = %{}) do
    map
    |> Enum.map(fn {k, v} -> {to_string(k), stringify_keys(v)} end)
    |> Enum.into(%{})
  end

  # Walk the list and stringify the keys of
  # of any map members
  def stringify_keys([head | rest]) do
    [stringify_keys(head) | stringify_keys(rest)]
  end

  def stringify_keys(not_a_map) when is_atom(not_a_map) and not is_boolean(not_a_map) do
    Atom.to_string(not_a_map)
  end

  def stringify_keys(not_a_map) do
    not_a_map
  end

  @doc """
  Return an `{:ok, module}` when the string successfully converts to an existing
  module.
  """
  def module_from_name("Elixir." <> _rest = module_name) do
    try do
      {:ok, String.to_existing_atom(module_name)}
    rescue
      _err ->
        Logger.error("Failed to restore using module_name #{inspect(module_name)}. Not found.")
        {:error, "ChatModel module #{inspect(module_name)} not found"}
    end
  end

  def module_from_name(module_name) do
    msg = "Not an Elixir module: #{inspect(module_name)}"
    Logger.error(msg)
    {:error, msg}
  end

  @doc """
  Split the messages into "system" and "other".
  Raises an error with the specified error message if more than 1 system message present.
  Returns a tuple with the single system message and the list of other messages.
  """
  @spec split_system_message([Message.t()], error_message :: String.t()) ::
          {nil | Message.t(), [Message.t()]} | no_return()
  def split_system_message(messages, error_message \\ "Only one system message is allowed") do
    {system, other} = Enum.split_with(messages, &(&1.role == :system))

    if length(system) > 1 do
      raise LangChainError, error_message
    end

    {List.first(system), other}
  end

  @doc """
  Replace the system message with a new system message. This retains all other
  messages as-is. An error is raised if there are more than 1 system messages.
  """
  @spec replace_system_message!([Message.t()], Message.t()) :: [Message.t()] | no_return()
  def replace_system_message!(messages, new_system_message) do
    {_old_system, rest} = split_system_message(messages)
    # return the new system message along with the rest
    [new_system_message | rest]
  end

  @doc """
  Changeset helper function for processing streamed text from an LLM.

  A delta of " " a single empty space is expected. The "cast" process of the
  changeset turns this into `nil` causing us to lose data.

  We want to take whatever we are given here.
  """
  def assign_string_value(changeset, field, attrs) do
    # get both possible versions of the arguments.
    val = Map.get(attrs, field) || Map.get(attrs, to_string(field))
    # if we got a string, use it as-is without casting
    if is_binary(val) do
      Ecto.Changeset.put_change(changeset, field, val)
    else
      changeset
    end
  end

  @doc """
  Migrate a string content to use `LangChain.Message.ContentPart`. This is for
  backward compatibility with models that don't yet support ContentPart while
  providing a more consistent API.

  This can be used with Message contents and ToolResult contents.
  """
  @spec migrate_to_content_parts(Ecto.Changeset.t()) :: Ecto.Changeset.t()
  def migrate_to_content_parts(%Ecto.Changeset{} = changeset) do
    case Changeset.fetch_change(changeset, :content) do
      {:ok, content} when is_binary(content) ->
        Changeset.put_change(changeset, :content, [ContentPart.text!(content)])

      # If a single ContentPart, wrap it in a list
      {:ok, %ContentPart{} = part} ->
        Changeset.put_change(changeset, :content, [part])

      # Don't modify if it's already a list
      {:ok, content} when is_list(content) ->
        changeset

      _ ->
        changeset
    end
  end

  # Set display_text on tool calls in streaming deltas using the chain's tool map.
  # Called from rewrap_callbacks_for_model to enrich deltas before they reach
  # consumer callbacks. Uses the same resolution logic as
  # LLMChain.resolve_display_text: prefer Function.display_text, fall back to
  # humanize_tool_name.
  @doc false
  @spec augment_delta_display_text([MessageDelta.t()], map()) :: [MessageDelta.t()]
  def augment_delta_display_text(deltas, tool_map) when is_list(deltas) do
    Enum.map(deltas, fn
      %{tool_calls: [_ | _] = tcs} = delta ->
        updated_tcs =
          Enum.map(tcs, fn tc ->
            if tc.name != nil and tc.display_text == nil do
              display_text =
                case tool_map[tc.name] do
                  %Function{display_text: dt} when not is_nil(dt) -> dt
                  _ -> humanize_tool_name(tc.name)
                end

              %{tc | display_text: display_text}
            else
              tc
            end
          end)

        %{delta | tool_calls: updated_tcs}

      delta ->
        delta
    end)
  end

  @doc """
  Convert a tool name to a human-friendly display string.

  ## Examples

      iex> LangChain.Utils.humanize_tool_name("file_read")
      "File read"

      iex> LangChain.Utils.humanize_tool_name("search_web")
      "Search web"
  """
  @spec humanize_tool_name(String.t()) :: String.t()
  def humanize_tool_name(name) when is_binary(name) do
    name
    |> String.replace("_", " ")
    |> String.capitalize()
  end
end
