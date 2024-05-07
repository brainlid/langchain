defmodule LangChain.Images do
  @moduledoc """
  Functions for working with `LangChain.GeneratedImage` files.
  """
  require Logger
  alias LangChain.Images.GeneratedImage

  @doc """
  Save the generated image file to a local directory. If the GeneratedFile is an
  URL, it is first downloaded then saved. If the is a Base64 encoded image, it
  is decoded and saved.
  """
  @spec(save_to_file(GeneratedImage.t(), String.t()) :: {:ok, String.t()}, {:error, String.t()})
  def save_to_file(%GeneratedImage{type: :url} = image, target_path) do
    # When a generated image is type `:url`, the content is the URL
    case Req.get(image.content) do
      {:ok, %Req.Response{body: body, status: 200}} ->
        # Save the file locally
        do_write_to_file(body, target_path)

      {:ok, %Req.Response{status: 404}} ->
        {:error, "Image file not found"}

      {:ok, %Req.Response{status: 500}} ->
        {:error, "Failed with server error 500"}

      {:error, reason} ->
        # Handle error
        Logger.error("Failed to download image: #{inspect(reason)}")
        {:error, reason}
    end
  end

  def save_to_file(%GeneratedImage{type: :base64} = image, target_path) do
    case Base.decode64(image.content) do
      {:ok, binary_data} ->
        do_write_to_file(binary_data, target_path)

      :error ->
        {:error, "Failed to base64 decode image data"}
    end
  end

  # write the contents to the file
  @spec do_write_to_file(binary(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  defp do_write_to_file(data, target_path) do
    case File.write(target_path, data) do
      :ok ->
        {:ok, target_path}

      {:error, :eacces} ->
        {:error, "Missing write permissions for the parent directory"}

      {:error, :eexist} ->
        {:error, "A file or directory already exists"}

      {:error, :enoent} ->
        {:error, "File path is invalid"}

      {:error, :enospc} ->
        {:error, "No space left on device"}

      {:error, :enotdir} ->
        {:error, "Part of path is not a directory"}

      {:error, reason} ->
        Logger.error(
          "Failed to save base64 image to file #{inspect(target_path)}. Reason: #{inspect(reason)}"
        )

        {:error, "Unrecognized error reason encountered: #{inspect(reason)}"}
    end
  end
end
