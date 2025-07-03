defmodule LangChain.Callbacks do
  @moduledoc """
  Defines the structure of callbacks and provides utilities for executing them.

  See `LangChain.Chains.ChainCallbacks` for the list of callbacks that can be
  used.
  """
  require Logger
  alias LangChain.LangChainError

  @doc """
  Fire a named callback with the list of arguments to pass. Takes a list of
  callback handlers and will execute the callback for each handler that defines
  a handler function for it.
  """
  @spec fire([map()], atom(), [any()]) :: :ok | no_return()
  def fire(callbacks, callback_name, arguments)

  def fire(callbacks, :on_llm_new_message, [messages]) when is_list(messages) do
    Enum.each(messages, fn m ->
      fire(callbacks, :on_llm_new_message, [m])
    end)
  end

  def fire(callbacks, callback_name, arguments) when is_list(callbacks) do
    # A model may contain multiple callback handler maps. Cycle through them to
    # execute the named callback with the arguments if assigned.
    Enum.each(callbacks, fn handlers_map ->
      # find if the callback is in the handler map
      case Map.get(handlers_map, callback_name) do
        nil ->
          # no handler attached
          :ok

        callback_fn when is_function(callback_fn) ->
          try do
            # execute the function
            apply(callback_fn, arguments)
          rescue
            err ->
              msg =
                "Callback handler for #{inspect(callback_name)} raised an exception: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"

              Logger.error(msg)
              raise LangChainError, msg
          end

        other ->
          msg =
            "Unexpected callback handler. Callback #{inspect(callback_name)} was assigned #{inspect(other)}"

          Logger.error(msg)
          raise LangChainError, msg
      end
    end)
  end
end
