# Load the ENV key for running live tests (optional for local testing).
Application.put_env(:langchain, :openai_key, System.get_env("OPENAI_API_KEY", ""))
Application.put_env(:langchain, :anthropic_key, System.get_env("ANTHROPIC_API_KEY", ""))
Application.put_env(:langchain, :google_ai_key, System.get_env("GOOGLE_API_KEY", ""))
Application.put_env(:langchain, :aws_access_key_id, System.get_env("AWS_ACCESS_KEY_ID", ""))
Application.put_env(:langchain, :perplexity_key, System.get_env("PERPLEXITY_API_KEY", ""))
Application.put_env(:langchain, :mistral_api_key, System.get_env("MISTRAL_API_KEY", ""))
Application.put_env(:langchain, :vertex_ai_key, System.get_env("VERTEX_API_KEY", ""))
Application.put_env(:langchain, :vertex_ai_endpoint, System.get_env("VERTEX_API_ENDPOINT", ""))
Application.put_env(:langchain, :xai_api_key, System.get_env("XAI_API_KEY", ""))

Application.put_env(
  :langchain,
  :aws_secret_access_key,
  System.get_env("AWS_SECRET_ACCESS_KEY", "")
)

Application.put_env(
  :langchain,
  :aws_region,
  System.get_env("AWS_REGION", "us-east-1")
)

Mimic.copy(LangChain.Utils.BedrockStreamDecoder)
Mimic.copy(LangChain.Utils.AwsEventstreamDecoder)

Mimic.copy(Req)
Mimic.copy(LangChain.ChatModels.ChatOpenAI)
Mimic.copy(LangChain.ChatModels.ChatAnthropic)
Mimic.copy(LangChain.ChatModels.ChatMistralAI)
Mimic.copy(LangChain.ChatModels.ChatBumblebee)
Mimic.copy(LangChain.ChatModels.ChatOllamaAI)
Mimic.copy(LangChain.Images.OpenAIImage)
Mimic.copy(LangChain.ChatModels.ChatPerplexity)
Mimic.copy(LangChain.Config)
Mimic.copy(LangChain.ChatModels.ChatVertexAI)
Mimic.copy(LangChain.ChatModels.ChatGoogleAI)
Mimic.copy(LangChain.ChatModels.ChatGrok)
ExUnit.configure(capture_log: true, exclude: [live_call: true])

ExUnit.start()
