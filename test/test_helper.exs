defmodule LangChain.TestHelpers do
  def load_env(key) do
    try do
      System.fetch_env!(key)
    rescue
      _e ->
        ""
    end
  end
end

Application.put_env(
  :langchain,
  :openai_key,
  LangChain.TestHelpers.load_env("OPENAI_API_KEY")
)

Application.put_env(
  :langchain,
  :anthropic_key,
  LangChain.TestHelpers.load_env("ANTHROPIC_API_KEY")
)

Application.put_env(
  :langchain,
  :google_ai_key,
  LangChain.TestHelpers.load_env("GOOGLE_API_KEY")
)

Mimic.copy(LangChain.ChatModels.ChatOpenAI)
Mimic.copy(LangChain.ChatModels.ChatAnthropic)
Mimic.copy(LangChain.ChatModels.ChatMistralAI)
Mimic.copy(LangChain.ChatModels.ChatBumblebee)
Mimic.copy(LangChain.Images.OpenAIImage)

ExUnit.configure(capture_log: true, exclude: [live_call: true])

ExUnit.start()
