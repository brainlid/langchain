defmodule LangChain.Callbacks do
  @moduledoc """
  Defines the structure of callbacks and provides utilities for executing them.

  See `LangChain.Chains.ChainCallbacks` for the list of callbacks that can be
  used.
  """
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
              raise LangChainError,
                    "Callback handler for #{inspect(callback_name)} raised an exception: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"
          end

        other ->
          raise LangChainError,
                "Unexpected callback handler. Callback #{inspect(callback_name)} was assigned #{inspect(other)}"
      end
    end)
  end

  @doc """
  Fold a decision-returning callback across the attached handler maps.

  Where `fire/3` discards what a handler returns, this collects it. A handler
  map that does not define `callback_name` is skipped without consulting the
  reducer.

  `reducer` receives two arguments: a zero-risk `invoke` function that applies
  the handler to a list of arguments, and the current accumulator. It returns
  `{:cont, acc}` to consult the next handler or `{:halt, acc}` to stop. Passing
  `invoke` rather than the raw handler lets the reducer choose the arguments
  per handler (so a handler can see what an earlier one changed) while the
  exception wrapping stays here, identical to `fire/3`.

  ## Example

      Callbacks.reduce(callbacks, :on_thing_reviewed, :allowed, fn invoke, acc ->
        case invoke.([subject]) do
          :ok -> {:cont, acc}
          {:denied, _reason} = denial -> {:halt, denial}
        end
      end)

  """
  @spec reduce([map()], atom(), acc, (([any()] -> any()), acc -> {:cont, acc} | {:halt, acc})) ::
          acc
        when acc: term()
  def reduce(callbacks, callback_name, acc, reducer)
      when is_list(callbacks) and is_function(reducer, 2) do
    Enum.reduce_while(callbacks, acc, fn handlers_map, acc ->
      case Map.get(handlers_map, callback_name) do
        nil ->
          {:cont, acc}

        callback_fn when is_function(callback_fn) ->
          reducer.(invoker(callback_name, callback_fn), acc)

        other ->
          raise LangChainError,
                "Unexpected callback handler. Callback #{inspect(callback_name)} was assigned #{inspect(other)}"
      end
    end)
  end

  defp invoker(callback_name, callback_fn) do
    fn arguments ->
      try do
        apply(callback_fn, arguments)
      rescue
        err ->
          raise LangChainError,
                "Callback handler for #{inspect(callback_name)} raised an exception: #{LangChainError.format_exception(err, __STACKTRACE__, :short)}"
      end
    end
  end
end
