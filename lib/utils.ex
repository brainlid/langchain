defmodule LangChain.Utils do
  @moduledoc """
  Collection of helpful utilities mostly for internal use.
  """
  alias Ecto.Changeset
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
end
