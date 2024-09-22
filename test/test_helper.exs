# Load the ENV key for running live OpenAI tests.
Application.put_env(:langchain, :openai_key, System.fetch_env!("OPENAI_API_KEY"))
# Application.put_env(:langchain, :anthropic_key, System.fetch_env!("ANTHROPIC_API_KEY"))
# Application.put_env(:langchain, :google_ai_key, System.fetch_env!("GOOGLE_API_KEY"))

Mimic.copy(LangChain.ChatModels.ChatOpenAI)
Mimic.copy(LangChain.ChatModels.ChatAnthropic)
Mimic.copy(LangChain.ChatModels.ChatMistralAI)
Mimic.copy(LangChain.ChatModels.ChatBumblebee)
Mimic.copy(LangChain.Images.OpenAIImage)

ExUnit.configure(capture_log: true, exclude: [live_call: true])

ExUnit.start()
