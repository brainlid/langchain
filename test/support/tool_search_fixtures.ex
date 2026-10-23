defmodule LangChain.ToolSearchFixtures do
  @moduledoc """
  A tool catalog for tool search tests: a device tool that reports a known
  count, a set of deferred tools in other areas, and two tools that are not
  deferred.
  """

  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall

  @hq_count "17"
  @warehouse_count "42"

  def hq_count, do: @hq_count
  def warehouse_count, do: @warehouse_count

  def system_prompt do
    """
    You are an IT operations assistant. Most of your tools are not loaded yet.
    You can search for tools about devices, billing, support tickets, and user
    accounts. Always use a tool to answer questions about these; never guess.
    Keep replies to one short sentence.
    """
  end

  @doc """
  The device tools. `devices_list` reports a fixed count per site and sends
  `{:tool_ran, "devices_list", site}` to `test_pid`. Pass `options` to add
  tool search options, such as a namespace, to every device tool.
  """
  def device_tools(test_pid, options \\ []) do
    [
      deferred(
        "devices_list",
        "List the managed devices at a site, with the device count.",
        [site: "The site name, such as hq or warehouse."],
        options
      )
      |> with_function(fn %{"site" => site}, _context ->
        send(test_pid, {:tool_ran, "devices_list", site})

        count =
          case String.downcase(site) do
            "hq" -> @hq_count
            "warehouse" -> @warehouse_count
            _ -> "0"
          end

        {:ok, "Site #{site} has #{count} managed devices."}
      end),
      deferred(
        "devices_get",
        "Get one managed device by its id.",
        [id: "The device id."],
        options
      ),
      deferred(
        "devices_restart",
        "Restart a managed device by its id.",
        [id: "The device id."],
        options
      )
    ]
  end

  def other_tools do
    [
      deferred("billing_invoices", "List invoices for the account.", month: "Month, YYYY-MM."),
      deferred("billing_balance", "Get the current account balance.", currency: "Currency."),
      deferred("billing_refund", "Refund an invoice by id.", id: "The invoice id."),
      deferred("tickets_list", "List open support tickets.", queue: "The queue name."),
      deferred("tickets_create", "Open a support ticket.", subject: "The subject line."),
      deferred("tickets_close", "Close a support ticket by id.", id: "The ticket id."),
      deferred("users_list", "List user accounts.", team: "The team name."),
      deferred("users_disable", "Disable a user account by id.", id: "The user id."),
      deferred("users_reset_password", "Reset a user's password.", id: "The user id."),
      plain("get_current_time", "Get the current UTC time."),
      plain("echo", "Repeat the given text back.")
    ]
  end

  def deferred(name, description, params, options \\ []) do
    properties =
      Map.new(params, fn {key, desc} ->
        {Atom.to_string(key), %{"type" => "string", "description" => desc}}
      end)

    Function.new!(%{
      name: name,
      description: description,
      parameters_schema: %{
        "type" => "object",
        "properties" => properties,
        "required" => Map.keys(properties)
      },
      function: fn _args, _context -> {:ok, "Not available in this test."} end,
      options: Keyword.merge([defer_loading: true], options)
    })
  end

  def plain(name, description) do
    Function.new!(%{
      name: name,
      description: description,
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "2026-10-14T12:00:00Z"} end
    })
  end

  defp with_function(%Function{} = function, fun), do: %Function{function | function: fun}

  @doc """
  The `"type"` of every provider item in the assistant messages, in order.
  """
  def provider_item_types(messages) do
    for %Message{role: :assistant, content: parts} when is_list(parts) <- messages,
        %ContentPart{type: :provider_item} = part <- parts,
        do: Keyword.fetch!(part.options, :item)["type"]
  end

  def tool_calls(messages, name) do
    for %Message{role: :assistant, tool_calls: calls} when is_list(calls) <- messages,
        %ToolCall{name: ^name} = call <- calls,
        do: call
  end
end
