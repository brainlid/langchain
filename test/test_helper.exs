# Load the ENV key for running live OpenAI tests.
Application.put_env(:langchain, :openai_key, System.fetch_env!("OPENAI_API_KEY"))

ExUnit.configure(capture_log: true, exclude: [live_call: true, live_call_ollama_ai: true])
ExUnit.start()
