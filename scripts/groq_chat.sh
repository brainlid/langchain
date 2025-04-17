#!/bin/bash

# Exit on any command failure
set -e

# Function to check if the LangChain library is compiled
check_langchain_compiled() {
  local langchain_path=".."
  local ebin_path="${langchain_path}/_build/dev/lib/langchain/ebin"
  
  if [ ! -d "$ebin_path" ]; then
    echo "LangChain library is not compiled. Compiling now..."
    (cd "$langchain_path" && mix compile)
    
    if [ ! -d "$ebin_path" ]; then
      echo "Failed to compile LangChain library."
      echo "Please run 'cd .. && mix deps.get && mix compile' manually."
      exit 1
    fi
  fi
}

# Check if GROQ_API_KEY is set
if [ -z "${GROQ_API_KEY}" ]; then
  echo "Error: GROQ_API_KEY environment variable not set."
  echo "Please set it with: export GROQ_API_KEY=your_api_key"
  echo "Get your API key from: https://console.groq.com/keys"
  exit 1
fi

# Change to the script directory
cd "$(dirname "$0")"

# Check if LangChain is compiled
check_langchain_compiled

# Print a cute ASCII art logo
echo "
 ██████╗ ██████╗  ██████╗  ██████╗      ██████╗██╗  ██╗ █████╗ ████████╗
██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝
██║  ███╗██████╔╝██║   ██║██║   ██║    ██║     ███████║███████║   ██║   
██║   ██║██╔══██╗██║   ██║██║▄▄ ██║    ██║     ██╔══██║██╔══██║   ██║   
╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝    ╚██████╗██║  ██║██║  ██║   ██║   
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝      ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
"

# Run the chat script
elixir groq_chat.exs