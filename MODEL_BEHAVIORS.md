# Model Behaviors

This is to document some of the differences between model behaviors. This is specifically about how one LLM responds differently than another. This information is likely to get out of date quickly as models change. Still, it may be helpful as a reference or starting point.

## Pre-filling the assistant's response

- ChatGPT 3.5 and 4 do not complete a pre-filled assistant response as we might hope. Giving it instructions to provide an answer in an `<answer>{{ANSWER}}</answer>` template, then starting an assistant message with `<answer>` to have it complete it, it will not include the closing `</answer>` tag.
  - 2024-04-17

- Anthropic's Claude 3 does respond well to pre-filled assistant responses and it is officially encouraged.
  - 2024-04-17