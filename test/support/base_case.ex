defmodule Langchain.BaseCase do
  @moduledoc """
  This module defines the test case to be used by
  tests that use Langchain features like Chat or LLMs.
  """

  use ExUnit.CaseTemplate

  using do
    quote do
      alias Langchain.Message

      # Import conveniences for testing with AI models
      import Langchain.BaseCase
      import Langchain.Utils.ApiOverride
    end
  end
end
