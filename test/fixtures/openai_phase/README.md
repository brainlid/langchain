# OpenAI Responses `phase` fixtures

Response bodies and one event stream recorded from the Responses API, plus one
file built by hand in the same shape. The narration tests decode these.

| File | Output shape | Why it is here |
|---|---|---|
| `commentary_with_tool_calls.json` | `[message(commentary), function_call ×4]` | The opening turn of a tool loop. The model narrates and acts in one response, so the chain keeps running on the tool calls. |
| `commentary_then_answer.json` | `[message(commentary), message(final_answer)]` | The closing turn of a tool loop, and the only shape where a single response holds both phases. The narration is progress reporting; the answer follows it. |
| `two_commentary_items.json` | `[message(commentary), reasoning, message(commentary), function_call ×4]` | Several message items in one response that share a phase, separated by a reasoning item. |
| `commentary_with_tool_calls_stream.txt` | `[message(commentary), function_call ×4]` | The same opening-turn shape as an SSE event stream. `phase` is on `response.output_item.added`, before the first text delta. |
| `synthesized_tool_loop.json` | `turns`: six responses; `after_narration`: three | Not recorded: built by hand in the recorded bodies' shape, because a live model rarely sends these. `turns` is a tool loop whose narration sits beside tool calls, alone, and as two items with no tool calls, closing on an answer. `after_narration` holds what a call made after a commentary-only turn can return instead of acting: `reasoning_only`, `empty` (no output), and `unphased` (a message with no `phase`). `truncated` is a response cut off by `max_output_tokens` partway through a function call's arguments, after its commentary item, with `status: "incomplete"`. `truncated_after_commentary` is cut off after its commentary item, before anything else. `LangChain.ScriptedResponsesAdapter` serves these as SSE as well as JSON. |

`captures/` holds whatever the live capture test last wrote. It is scratch and
is not committed; the files above are the ones tests read.

Regenerate the scratch captures with:

    mix test test/chat_models/chat_open_ai_responses_phase_live_test.exs --include live_open_ai
