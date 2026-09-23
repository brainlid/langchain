# OpenAI Responses `phase` fixtures

Real response bodies and one real event stream, recorded from the Responses API
and kept because they are what the narration tests decode.

| File | Output shape | Why it is here |
|---|---|---|
| `commentary_with_tool_calls.json` | `[message(commentary), function_call ×4]` | The opening turn of a tool loop. The model narrates and acts in one response, so the chain keeps running on the tool calls. |
| `commentary_then_answer.json` | `[message(commentary), message(final_answer)]` | The closing turn of a tool loop, and the only shape where a single response holds both phases. The narration is progress reporting; the answer follows it. |
| `two_commentary_items.json` | `[message(commentary), reasoning, message(commentary), function_call ×4]` | Several message items in one response that share a phase, separated by a reasoning item. |
| `commentary_with_tool_calls_stream.txt` | `[message(commentary), function_call ×4]` | The same opening-turn shape as an SSE event stream. `phase` is on `response.output_item.added`, before the first text delta. |

`captures/` holds whatever the live capture test last wrote. It is scratch and
is not committed; the files above are the ones tests read.

Regenerate the scratch captures with:

    mix test test/chat_models/chat_open_ai_responses_phase_live_test.exs --include live_open_ai
