defmodule LangChain.BaseCase do
  @moduledoc """
  This module defines the test case to be used by
  tests that use LangChain features like Chat or LLMs.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      alias LangChain.Message
      alias LangChain.MessageDelta

      # Import conveniences for testing with AI models
      import LangChain.BaseCase
      import LangChain.Utils.ApiOverride

      @doc """
      Helper function for loading an image as base64 encoded text.
      """
      def load_image_base64(filename) do
        Path.join("./test/support/images", filename)
        |> File.read!()
        |> :base64.encode()
      end
    end
  end
end
