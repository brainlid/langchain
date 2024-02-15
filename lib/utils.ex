defmodule LangChain.Utils do
  @moduledoc """
  Collection of helpful utilities mostly for internal use.
  """
  alias Ecto.Changeset
  require Logger
  alias LangChain.LangChainError

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

  @type callback_data ::
          {:ok, Message.t() | MessageDelta.t() | [Message.t() | MessageDelta.t()]}
          | {:error, String.t()}

  @doc """
  Fire a streaming callback if present.
  """
  @spec fire_callback(
          %{optional(:stream) => boolean()},
          data :: callback_data() | [callback_data()],
          (callback_data() -> any())
        ) :: :ok
  def fire_callback(%{stream: true}, _data, nil) do
    Logger.warning("Streaming call requested but no callback function was given.")
    :ok
  end

  def fire_callback(_model, _data, nil), do: :ok

  def fire_callback(_model, data, callback_fn) when is_function(callback_fn) do
    # OPTIONAL: Execute callback function
    data
    |> List.flatten()
    |> Enum.each(fn item -> callback_fn.(item) end)

    :ok
  end

  @doc """
  Create a function to handle the streaming request. 
  """
  @spec handle_stream_fn(
          %{optional(:stream) => boolean()},
          process_response_fn :: function(),
          callback_fn :: function()
        ) :: function()
  def handle_stream_fn(model, process_response_fn, callback_fn) do
    fn {:data, raw_data}, {req, response} ->
      # cleanup data because it isn't structured well for JSON.
      new_data = decode_streamed_data(raw_data, process_response_fn)
      # execute the callback function for each MessageDelta
      fire_callback(model, new_data, callback_fn)
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
      updated_response = %{response | body: old_body ++ new_data}

      {:cont, {req, updated_response}}
    end
  end

  defp decode_streamed_data(data, process_response_fn) do
    # Data comes back like this:
    #
    # "data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"function_call\":{\"name\":\"calculator\",\"arguments\":\"\"}},\"finish_reason\":null}]}\n\n
    #  data: {\"id\":\"chatcmpl-7e8yp1xBhriNXiqqZ0xJkgNrmMuGS\",\"object\":\"chat.completion.chunk\",\"created\":1689801995,\"model\":\"gpt-4-0613\",\"choices\":[{\"index\":0,\"delta\":{\"function_call\":{\"arguments\":\"{\\n\"}},\"finish_reason\":null}]}\n\n"
    #
    # In that form, the data is not ready to be interpreted as JSON. Let's clean
    # it up first.

    data
    |> String.split("data: ")
    |> Enum.map(fn str ->
      str
      |> String.trim()
      |> case do
        "" ->
          :empty

        "[DONE]" ->
          :empty

        json ->
          json
          |> Jason.decode()
          |> case do
            {:ok, parsed} ->
              parsed

            {:error, reason} ->
              {:error, reason}
          end
          |> process_response_fn.()
      end
    end)
    # returning a list of elements. "junk" elements were replaced with `:empty`.
    # Filter those out down and return the final list of MessageDelta structs.
    |> Enum.filter(fn d -> d != :empty end)
    # if there was a single error returned in a list, flatten it out to just
    # return the error
    |> case do
      [{:error, reason}] ->
        raise LangChainError, reason

      other ->
        other
    end
  end
end
