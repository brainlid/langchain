defmodule LangChain.ChatModels.ChatAnthropicToolSearchLiveTest do
  @moduledoc """
  Tool search against the live Anthropic API.

  - A model finds a deferred tool with the server-side search tool and calls
    it. The search blocks are replayed unchanged on the next turn, and the
    replayed prefix is read from the prompt cache.
  - A model loads deferred tools through a client tool that returns tool
    references.
  - With thinking on, a turn whose thinking surrounds the search blocks
    replays without a binding error.
  """
  use ExUnit.Case, async: true

  require Logger

  alias LangChain.Chains.LLMChain
  alias LangChain.ChatModels.ChatAnthropic
  alias LangChain.Function
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.NativeTool

  @moduletag live_call: true, live_anthropic: true, timeout: 600_000

  @binding_beta "thinking-binding-controls-2026-08-01"

  # Answers only the deferred device tools can give. The model cannot guess
  # them, so seeing one in the reply shows the tool ran.
  @hq_count "17"
  @warehouse_count "42"

  @system """
  You are an IT operations assistant. Most of your tools are not loaded yet.
  You can search for tools about devices, billing, support tickets, and user
  accounts. Always use a tool to answer questions about these; never guess.
  Keep replies to one short sentence.
  """

  # Pads the system prompt past every model's minimum cacheable length, so the
  # cache assertion measures caching and not the threshold.
  @handbook 1..300
            |> Enum.map_join("\n", fn n ->
              "Handbook rule #{n}: record change #{n} in the ledger with its owner, " <>
                "reviewer, ticket number, and rollback plan before applying it."
            end)

  defp device_tools(test_pid) do
    [
      deferred("devices_list", "List the managed devices at a site, with the device count.",
        site: "The site name, such as hq or warehouse."
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
      deferred("devices_get", "Get one managed device by its id.", id: "The device id."),
      deferred("devices_restart", "Restart a managed device by its id.", id: "The device id.")
    ]
  end

  defp other_tools do
    [
      deferred("billing_invoices", "List invoices for the account.", month: "Month, YYYY-MM."),
      deferred("billing_balance", "Get the current account balance.", currency: "Currency code."),
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

  defp deferred(name, description, params) do
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
      options: [defer_loading: true]
    })
  end

  defp plain(name, description) do
    Function.new!(%{
      name: name,
      description: description,
      parameters_schema: %{"type" => "object", "properties" => %{}},
      function: fn _args, _context -> {:ok, "2026-10-14T12:00:00Z"} end
    })
  end

  defp with_function(%Function{} = function, fun), do: %Function{function | function: fun}

  defp search_blocks(messages) do
    for %Message{role: :assistant, content: parts} when is_list(parts) <- messages,
        %ContentPart{type: :provider_item} = part <- parts,
        do: Keyword.fetch!(part.options, :item)["type"]
  end

  defp tool_calls(messages, name) do
    for %Message{role: :assistant, tool_calls: calls} when is_list(calls) <- messages,
        %ToolCall{name: ^name} = call <- calls,
        do: call
  end

  defp reply_text(%LLMChain{last_message: message}),
    do: ContentPart.content_to_string(message.content) || ""

  defp cache_read(%Message{metadata: %{usage: usage}}),
    do: Map.get(usage.raw || %{}, "cache_read_input_tokens", 0)

  defp cache_read(_message), do: 0

  defp run(chain) do
    case LLMChain.run(chain, mode: :while_needs_response) do
      {:ok, chain} ->
        {:ok, chain}

      {:error, chain, error} ->
        flunk("""
        Run failed: #{inspect(error, limit: :infinity)}
        Last message: #{inspect(chain.last_message, limit: :infinity)}
        """)
    end
  end

  for model <- ["claude-sonnet-5-5", "claude-haiku-4-5"], stream <- [false, true] do
    test "#{model} (stream: #{stream}): searches for a deferred tool and keeps it loaded" do
      model = unquote(model)
      stream = unquote(stream)
      test_pid = self()

      llm =
        ChatAnthropic.new!(%{
          model: model,
          stream: stream,
          max_tokens: 2_000,
          receive_timeout: 120_000,
          cache_control: %{"type" => "ephemeral"}
        })

      tools = [NativeTool.tool_search() | device_tools(test_pid) ++ other_tools()]

      {:ok, chain} =
        %{llm: llm}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!(@system <> "\n" <> @handbook))
        |> LLMChain.add_tools(tools)
        |> LLMChain.add_message(Message.new_user!("How many managed devices are at the hq site?"))
        |> run()

      assert_received {:tool_ran, "devices_list", _site}
      assert reply_text(chain) =~ @hq_count

      first_turn = chain.messages
      assert "server_tool_use" in search_blocks(first_turn)
      assert "tool_search_tool_result" in search_blocks(first_turn)

      # The search is never a tool call the chain runs.
      assert [] = tool_calls(first_turn, "tool_search_tool_bm25")

      # Turn two replays the search blocks. A rejected replay fails the run.
      turn_two_start = length(chain.messages)

      {:ok, chain} =
        chain
        |> LLMChain.add_message(Message.new_user!("And at the warehouse site?"))
        |> run()

      assert reply_text(chain) =~ @warehouse_count
      turn_two = Enum.drop(chain.messages, turn_two_start)
      assert [_ | _] = tool_calls(turn_two, "devices_list")

      researched? = "server_tool_use" in search_blocks(turn_two)
      Logger.info("#{model} stream=#{stream}: searched again on turn two: #{researched?}")

      # The prefix through turn one comes from the cache.
      [first_reply | _] = for %Message{role: :assistant} = m <- turn_two, do: m
      assert cache_read(first_reply) > 0
    end
  end

  for model <- ["claude-sonnet-5-5", "claude-haiku-4-5"], stream <- [false, true] do
    test "#{model} (stream: #{stream}): a client tool loads deferred tools by reference" do
      model = unquote(model)
      stream = unquote(stream)
      test_pid = self()

      load_tools =
        Function.new!(%{
          name: "load_tools",
          description:
            "Load the tools for an area so you can call them. Areas: devices, billing, " <>
              "tickets, users.",
          parameters_schema: %{
            "type" => "object",
            "properties" => %{
              "area" => %{
                "type" => "string",
                "enum" => ["devices", "billing", "tickets", "users"]
              }
            },
            "required" => ["area"]
          },
          function: fn %{"area" => area}, _context ->
            send(test_pid, {:loaded, area})

            references =
              (device_tools(test_pid) ++ other_tools())
              |> Enum.filter(&String.starts_with?(&1.name, area))
              |> Enum.map(&ContentPart.tool_reference!(&1.name))

            {:ok, [ContentPart.text!("Loaded the #{area} tools.") | references]}
          end
        })

      llm =
        ChatAnthropic.new!(%{
          model: model,
          stream: stream,
          max_tokens: 2_000,
          receive_timeout: 120_000
        })

      {:ok, chain} =
        %{llm: llm}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_system!(
            "You are an IT operations assistant. Call load_tools to get the tools for an " <>
              "area before answering questions about it. Keep replies to one sentence."
          )
        )
        |> LLMChain.add_tools([load_tools | device_tools(test_pid) ++ other_tools()])
        |> LLMChain.add_message(Message.new_user!("How many managed devices are at the hq site?"))
        |> run()

      assert_received {:loaded, "devices"}
      assert_received {:tool_ran, "devices_list", _site}
      assert reply_text(chain) =~ @hq_count

      assert Enum.any?(chain.messages, fn
               %Message{role: :tool, tool_results: results} when is_list(results) ->
                 Enum.any?(results, fn %ToolResult{content: content} ->
                   is_list(content) and
                     Enum.any?(content, &match?(%ContentPart{type: :tool_reference}, &1))
                 end)

               _other ->
                 false
             end)
    end
  end

  for stream <- [false, true] do
    test "claude-sonnet-5-5 (stream: #{stream}): thinking around search blocks replays" do
      stream = unquote(stream)
      test_pid = self()

      llm =
        ChatAnthropic.new!(%{
          model: "claude-sonnet-5-5",
          stream: stream,
          max_tokens: 16_000,
          receive_timeout: 300_000,
          thinking: %{
            "type" => "adaptive",
            "display" => "summarized",
            "block_binding" => %{"prefix_mismatch_behavior" => "error"}
          },
          output_config: %{"effort" => "high"},
          beta_headers: [@binding_beta]
        })

      tools = [NativeTool.tool_search() | device_tools(test_pid) ++ other_tools()]

      {:ok, chain} =
        %{llm: llm}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_system!(@system))
        |> LLMChain.add_tools(tools)
        |> LLMChain.add_message(
          Message.new_user!(
            "Compare the managed device counts at the hq and warehouse sites and tell me " <>
              "the difference. Think it through before searching."
          )
        )
        |> run()

      assert reply_text(chain) =~ "25"
      assert "server_tool_use" in search_blocks(chain.messages)

      thinking? =
        Enum.any?(chain.messages, fn
          %Message{role: :assistant, content: parts} when is_list(parts) ->
            Enum.any?(parts, &match?(%ContentPart{type: :thinking}, &1))

          _other ->
            false
        end)

      Logger.info("sonnet-5-5 stream=#{stream}: thinking present on turn one: #{thinking?}")

      # The whole first turn, thinking and search blocks in place, goes back
      # with binding enforced. A moved or altered block is a 400.
      {:ok, chain} =
        chain
        |> LLMChain.add_message(Message.new_user!("Which site has more?"))
        |> run()

      assert reply_text(chain) =~ ~r/warehouse/i
    end
  end
end
