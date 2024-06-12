defmodule LangChain.Utils do
  @moduledoc """
  Collection of helpful utilities mostly for internal use.
  """
  alias LangChain.LangChainError
  alias Ecto.Changeset
  alias LangChain.Callbacks
  alias LangChain.Message
  alias LangChain.MessageDelta
  alias LangChain.TokenUsage
  require Logger

  @doc """
  Only add the key to the map if the value is present. When the value is a list,
  the key will not be added when the list is empty. If the value is `nil`, it
  will not be added.
  """
  @spec conditionally_add_to_map(%{any() => any()}, key :: any(), value :: nil | list()) :: %{
          any() => any()
        }
  def conditionally_add_to_map(map, key, value)

  def conditionally_add_to_map(map, _key, nil), do: map

  def conditionally_add_to_map(map, _key, []), do: map

  def conditionally_add_to_map(map, key, value) do
    Map.put(map, key, value)
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

  @type callback_data :: Message.t() | MessageDelta.t() | TokenUsage.t() | {:error, String.t()}

  @doc """
  Fire a streaming callback if present.
  """
  @spec fire_streamed_callback(
          %{optional(:stream) => boolean(), callbacks: [map()]},
          data :: callback_data() | [callback_data()]
        ) :: :ok | no_return()

  # fire a set of callbacks when receiving a list
  def fire_streamed_callback(model, data) when is_list(data) do
    # Execute callback handler for each received data element
    data
    |> List.flatten()
    |> Enum.each(fn item ->
      fire_streamed_callback(model, item)
    end)
  end

  def fire_streamed_callback(model, %MessageDelta{} = delta) do
    # Execute callback handler for single received delta element
    Callbacks.fire(model.callbacks, :on_llm_new_delta, [model, delta])
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

        # decode the received stream data
        {parsed_data, incomplete} =
          decode_stream_fn.({raw_data, buffered})

        # transform what was fully received into structs
        parsed_data =
          parsed_data
          |> Enum.map(transform_data_fn)
          |> Enum.reject(&(&1 == :skip))

        # execute the callback function for each MessageDelta and an optional
        # TokenUsage
        fire_streamed_callback(model, parsed_data)
        old_body = if response.body == "", do: [], else: response.body

        # Returns %Req.Response{} where the body contains ALL the stream delta
        # chunks converted to MessageDelta structs. The body is a list of lists like this...
        #
        # body: [
        #         [
        #           %LangChain.MessageDelta{
        #             content: nil,
        #             index: 0,
        #             function_name: nil,
        #             role: :assistant,
        #             arguments: nil,
        #             complete: false
        #           }
        #         ],
        #         ...
        #       ]
        #
        # The reason for the inner list is for each entry in the "n" choices. By default only 1.
        updated_response = %{response | body: old_body ++ parsed_data}
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
        case Jason.decode(raw_data) do
          {:ok, data} ->
            {:halt, {req, %{response | body: transform_data_fn.(data)}}}

          {:error, reason} ->
            Logger.error("Failed to JSON decode error response. ERROR: #{inspect(reason)}")

            {:halt,
             {req, LangChainError.exception("Failed to handle error response from server.")}}
        end

      {:data, _raw_data}, {req, response} ->
        Logger.error("Unhandled API response!")
        {:halt, {req, response}}
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
end
