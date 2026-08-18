defmodule LangChain.OpenTelemetry.Enrich do
  @moduledoc """
  Supported way to add application context to the OpenTelemetry spans LangChain
  creates.

  Safe to call unconditionally. Unlike `LangChain.OpenTelemetry` — which only exists
  when the optional `:opentelemetry_api` dependency is present — every function here
  compiles to a no-op when OpenTelemetry is unavailable. Callers need neither their
  own `Code.ensure_loaded?/1` guards nor an OpenTelemetry dependency of their own.

  ## Which to reach for

  Most callers want neither of these. Setting `custom_context[:otel_attributes]` on
  the chain covers the common case, and covers it better, because those attributes
  are applied when each span *opens* rather than after the fact:

      chain
      |> LLMChain.update_custom_context(%{
        otel_attributes: %{"user.id" => user.id, "organization.id" => org.id}
      })

  Use this module for the two things that map cannot do.

  ### `set_current_span_attributes/1` — enrich from inside a running operation

  For values only known once the work is underway: a resolved record id, a
  classification the tool computed, a cache hit or miss.

      # inside a tool function, or an :on_tool_execution_completed callback
      Enrich.set_current_span_attributes(%{"myapp.records_matched" => length(rows)})

  Which span it lands on depends on where you call it, because it always targets the
  innermost open span:

  | Called from | Span it enriches |
  |---|---|
  | A tool's own function body | `execute_tool {tool}` |
  | `:on_message_processed`, `:on_llm_token_usage`, and the tool callbacks (`:on_tool_pre_execution`, `:on_tool_execution_completed`, `:on_tool_execution_failed`, `:on_tool_execution_exception`) | `invoke_agent {chain_type}` |
  | Outside any LangChain operation | Whatever span your app has open, or nothing |

  Attributes set this way apply to that one span. They are not inherited by sibling
  or later spans — use `put_inherited_attributes/1` for that.

  ### `put_inherited_attributes/1` — seed context before the chain runs

  For hosts that establish request context outside LangChain entirely (a Plug, a
  LiveView `mount/3`, an Oban worker) and want every LangChain span in that process
  to carry it, without threading a chain through their code:

      # in a Plug, before any chain exists
      Enrich.put_inherited_attributes(%{"organization.id" => conn.assigns.org.id})

  These ride the OpenTelemetry context, so they reach every span LangChain opens
  afterwards in that process, and follow the trace across process boundaries wherever
  the context is propagated. They are never serialized onto outbound requests the way
  baggage is.

  ## Values

  Both functions coerce values through
  `LangChain.OpenTelemetry.Attributes.attribute_value/1`. Strings, numbers, booleans,
  and homogeneous lists of those stay native; anything else is JSON-encoded. `nil`
  values are dropped. This matters more than it looks: an uncoerced nested map makes
  the SDK raise, the span handler traps the exception, and the span silently vanishes
  from the trace. Since this module is the one place arbitrary caller data reaches
  span attributes, it never skips coercion.
  """

  # Guarded on the `:opentelemetry` module from the `opentelemetry_api` package, the
  # same optional dependency the rest of the OTel integration compiles against. The
  # difference here is that BOTH branches are defined: this module is called by
  # ordinary application code, which must not have to care whether OpenTelemetry is
  # present. Defining the functions only inside the guard — as the rest of the
  # integration does — would raise `UndefinedFunctionError` at the call site instead
  # of doing nothing, making an optional dependency effectively required.
  if Code.ensure_loaded?(:opentelemetry) do
    alias LangChain.OpenTelemetry.Attributes
    alias LangChain.OpenTelemetry.SpanHandler

    @doc """
    Sets attributes on the currently-active span.

    Returns `:ok` always, including when OpenTelemetry is unavailable or no span is
    open — enrichment is never worth failing a request over.

    ## Example

        LangChain.OpenTelemetry.Enrich.set_current_span_attributes(%{
          "myapp.cache" => "hit",
          "myapp.rows" => 42
        })
    """
    @spec set_current_span_attributes(map() | keyword()) :: :ok
    def set_current_span_attributes(attrs) when is_map(attrs) or is_list(attrs) do
      case normalize(attrs) do
        [] ->
          :ok

        normalized ->
          case OpenTelemetry.Tracer.current_span_ctx() do
            :undefined -> :ok
            span_ctx -> OpenTelemetry.Span.set_attributes(span_ctx, normalized)
          end

          :ok
      end
    end

    @doc """
    Seeds attributes that every LangChain span opened later in this process will
    inherit.

    Merges with anything already seeded, with the new values winning. Returns `:ok`
    always.

    Note this attaches a new OpenTelemetry context to the current process and does
    not detach it — appropriate for request-scoped setup code (a Plug, a LiveView
    mount, a job's `perform/1`), where the process ends with the request. Do not call
    it in a loop in a long-lived process.

    ## Example

        # in a Plug
        LangChain.OpenTelemetry.Enrich.put_inherited_attributes(%{
          "organization.id" => conn.assigns.current_org.id,
          "myapp.plan" => conn.assigns.current_org.plan
        })
    """
    @spec put_inherited_attributes(map() | keyword()) :: :ok
    def put_inherited_attributes(attrs) when is_map(attrs) or is_list(attrs) do
      case normalize(attrs) do
        [] ->
          :ok

        normalized ->
          key = SpanHandler.ctx_attributes_key()

          existing =
            case OpenTelemetry.Ctx.get_value(key, []) do
              list when is_list(list) -> list
              _other -> []
            end

          OpenTelemetry.Ctx.get_current()
          |> OpenTelemetry.Ctx.set_value(key, Attributes.merge(existing, normalized))
          |> OpenTelemetry.Ctx.attach()

          :ok
      end
    end

    defp normalize(attrs) do
      Enum.flat_map(attrs, fn {key, value} ->
        case {attribute_key(key), Attributes.attribute_value(value)} do
          {nil, _} -> []
          {_key, nil} -> []
          {key, value} -> [{key, value}]
        end
      end)
    end

    defp attribute_key(key) when is_binary(key), do: key
    defp attribute_key(key) when is_atom(key) and not is_nil(key), do: Atom.to_string(key)
    defp attribute_key(_), do: nil
  else
    @doc """
    No-op: `:opentelemetry_api` is not available in this build.
    """
    @spec set_current_span_attributes(map() | keyword()) :: :ok
    def set_current_span_attributes(attrs) when is_map(attrs) or is_list(attrs), do: :ok

    @doc """
    No-op: `:opentelemetry_api` is not available in this build.
    """
    @spec put_inherited_attributes(map() | keyword()) :: :ok
    def put_inherited_attributes(attrs) when is_map(attrs) or is_list(attrs), do: :ok
  end
end
