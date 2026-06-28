[
  # ReqLLM's @spec for generate_text/3 and stream_text/3 declares the second
  # argument as `String.t() | list()`, but the runtime API accepts
  # `%ReqLLM.Context{}` (which is what providers expect). Until that upstream
  # spec is broadened, these calls trip dialyzer despite working correctly.
  {"lib/chat_models/chat_req_llm.ex", :no_return},
  {"lib/chat_models/chat_req_llm.ex", :pattern_match_cov},
  {"lib/chat_models/chat_req_llm.ex", :call},

  # ChatAnthropic streaming defensive clauses. The {:ok, {:error, _}} and
  # `other ->` catchalls in do_api_request/4 are reached at runtime via
  # Mimic-mocked Req.post responses and arbitrary streaming decoder output,
  # so they're load-bearing despite dialyzer's static view.
  {"lib/chat_models/chat_anthropic.ex", :pattern_match},
  {"lib/chat_models/chat_anthropic.ex", :pattern_match_cov},

  # MessageDelta.to_message has a defensive `{:error, _reason}` catchall after
  # `{:error, %Ecto.Changeset{}}`. Message.new currently only returns the
  # changeset error, but the catchall future-proofs against spec changes.
  {"lib/message_delta.ex", :pattern_match_cov},

  # lib/web_socket.ex has several pattern_match warnings against Mint.WebSocket
  # return-shape mismatches (3-tuple vs 2-tuple errors). Pre-existing on main;
  # tracked for a follow-up PR rather than fixed here, since this PR's scope
  # is the dialyzer setup itself.
  {"lib/web_socket.ex", :pattern_match}
]
