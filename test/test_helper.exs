# Load the ENV key for running live OpenAI tests.
Application.put_env(:langchain, :openai_key, System.fetch_env!("OPENAI_API_KEY"))
Application.put_env(:langchain, :anthropic_key, System.fetch_env!("ANTHROPIC_API_KEY"))
Application.put_env(:langchain, :google_ai_key, System.fetch_env!("GOOGLE_API_KEY"))
Application.put_env(:langchain, :aws_access_key_id, System.fetch_env!("AWS_ACCESS_KEY_ID"))

Application.put_env(
  :langchain,
  :aws_secret_access_key,
  System.fetch_env!("AWS_SECRET_ACCESS_KEY")
)

Mimic.copy(LangChain.Utils.BedrockStreamDecoder)
Mimic.copy(LangChain.Utils.AwsEventstreamDecoder)

ExUnit.configure(capture_log: true, exclude: [live_call: true])

ExUnit.start()
