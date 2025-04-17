# Setup code path
langchain_path = Path.join(__DIR__, "..")
lib_path = Path.join(langchain_path, "_build/dev/lib")

# Function to add all dependencies and start them
add_deps = fn ->
  # Get list of all dependency directories
  case File.ls(lib_path) do
    {:ok, deps} ->
      # Add each dependency to the code path
      Enum.each(deps, fn dep ->
        ebin_path = Path.join([lib_path, dep, "ebin"])
        if File.dir?(ebin_path) do
          Code.append_path(ebin_path)
          
          # Try to convert dep string to atom and start the application
          try do
            app_name = String.to_existing_atom(dep)
            Application.ensure_all_started(app_name)
          rescue
            _ -> :ok  # Ignore if we can't convert to atom
          end
        end
      end)
      
    {:error, reason} ->
      IO.puts("Error loading dependencies: #{reason}")
      System.halt(1)
  end
end

# Add all dependencies to code path and start them
add_deps.()

# Properly start Finch
{:ok, _} = Application.ensure_all_started(:finch)
# Start Finch with the expected name
Finch.start_link(name: Req.Finch)

# Import required modules
alias LangChain.ChatModels.ChatGroq
alias LangChain.Message

# Default configuration
default_model = "llama-3.1-8b-instant"
api_key = System.get_env("GROQ_API_KEY")

# Function to print help information
defmodule GroqChatHelp do
  def print_help(default_model) do
    IO.puts("""
    Groq Chat CLI - Interactive chat with Groq AI models
    
    Available commands:
      /model <model_name> - Change the model (default: #{default_model})
      /models             - List available models
      /system <message>   - Set a system message
      /help               - Show this help
      /quit or /exit      - Exit the chat
      
    Available models:
      - llama-3.1-8b-instant
      - llama-3.3-70b-versatile
      - llama3-70b-8192
      - llama3-8b-8192
      - gemma2-9b-it
      
    Example:
      /model llama-3.3-70b-versatile
      /system You are a helpful assistant that speaks in rhyming poetry.
    """)
  end
end

# Check for API key
unless api_key do
  IO.puts("\e[31mError: GROQ_API_KEY environment variable not set.\e[0m")
  IO.puts("Please set it with: export GROQ_API_KEY=your_api_key")
  System.halt(1)
end

# Initialize model and chat history
{:ok, model} = ChatGroq.new(%{
  model: default_model,
  api_key: api_key,
  stream: true  # Enable streaming for a more interactive experience
})

# Start with an empty conversation history
conversation = []
{:ok, system_message} = Message.new(%{role: "system", content: "You are a helpful, concise assistant."})

# Add system message to conversation
conversation = [system_message | conversation]

# Print welcome message
IO.puts("\n\e[1;36m=== Groq Chat CLI ===\e[0m")
IO.puts("Using model: \e[1m#{model.model}\e[0m")
IO.puts("Type /help for commands or /quit to exit\n")

# Chat loop
chat_loop = fn chat_loop, model, conversation ->
  # Prompt for user input
  IO.write("\e[1;32mYou:\e[0m ")
  input = IO.gets("") |> String.trim()
  
  # Handle special commands
  cond do
    input in ["/quit", "/exit"] ->
      IO.puts("\nGoodbye!")
      System.halt(0)
      
    input == "/help" ->
      GroqChatHelp.print_help(model.model)
      chat_loop.(chat_loop, model, conversation)
      
    input == "/models" ->
      IO.puts("\nAvailable models:")
      IO.puts("  - llama-3.1-8b-instant")
      IO.puts("  - llama-3.3-70b-versatile")
      IO.puts("  - llama3-70b-8192")
      IO.puts("  - llama3-8b-8192")
      IO.puts("  - gemma2-9b-it")
      IO.puts("\nCurrent model: #{model.model}")
      chat_loop.(chat_loop, model, conversation)
      
    String.starts_with?(input, "/model ") ->
      new_model_name = String.trim(String.replace(input, "/model ", ""))
      IO.puts("\nSwitching to model: #{new_model_name}")
      
      {:ok, new_model} = ChatGroq.new(%{
        model: new_model_name,
        api_key: api_key,
        stream: true
      })
      
      chat_loop.(chat_loop, new_model, conversation)
      
    String.starts_with?(input, "/system ") ->
      system_content = String.trim(String.replace(input, "/system ", ""))
      {:ok, new_system} = Message.new(%{role: "system", content: system_content})
      
      # Replace existing system message or add a new one
      new_conversation = 
        if Enum.any?(conversation, fn msg -> msg.role == :system end) do
          Enum.map(conversation, fn msg ->
            if msg.role == :system, do: new_system, else: msg
          end)
        else
          [new_system | conversation]
        end
        
      IO.puts("\nSystem message updated")
      chat_loop.(chat_loop, model, new_conversation)
      
    true ->
      # Regular message - add to conversation
      {:ok, user_message} = Message.new(%{role: "user", content: input})
      updated_conversation = [user_message | conversation]
      
      IO.write("\e[1;34mGroq:\e[0m ")
      
      # Call Groq API with streaming
      case ChatGroq.call(model, Enum.reverse(updated_conversation), []) do
        {:ok, streamed_response} ->
          # Process streamed response
          full_content = 
            Enum.reduce(streamed_response, "", fn message_deltas, acc ->
              # Extract content from deltas
              delta_text = Enum.reduce(message_deltas, "", fn delta, delta_acc ->
                delta_acc <> (delta.content || "")
              end)
              
              # Print incrementally
              IO.write(delta_text)
              acc <> delta_text
            end)
          
          IO.puts("\n")
          
          # Add assistant response to conversation
          {:ok, assistant_message} = Message.new(%{role: "assistant", content: full_content})
          updated_conversation = [assistant_message | updated_conversation]
          
          # Continue the chat
          chat_loop.(chat_loop, model, updated_conversation)
          
        {:error, error} ->
          IO.puts("\e[31mError: #{error.message}\e[0m")
          chat_loop.(chat_loop, model, conversation)
      end
  end
end

# Start the chat loop
chat_loop.(chat_loop, model, conversation)