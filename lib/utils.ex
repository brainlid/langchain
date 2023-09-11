defmodule Langchain.Utils do
  @moduledoc """
  Collection of helpful utilities mostly for internal use.
  """

  @doc """
  Only add the key to the map if the value is present. When the value is a list,
  the key will not be added when the list is empty.
  """
  @spec conditionally_add_to_map(%{atom() => any()}, key :: atom(), value :: nil | list()) :: %{
          atom() => any()
        }
  def conditionally_add_to_map(map, key, value)

  def conditionally_add_to_map(map, _key, []), do: map

  def conditionally_add_to_map(map, key, value) when is_list(value) do
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
      Gettext.dngettext(Langchain.Gettext, "errors", msg, msg, count, opts)
    else
      Gettext.dgettext(Langchain.Gettext, "errors", msg, opts)
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
end
