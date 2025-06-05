raw_data #=> "event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea\",\"object\":\"response\",\"created_at\":1749097929,\"status\":\"in_progress\",\"background\":false,\"error\":null,\"incomplete_details\":null,\"instructions\":null,\"max_output_tokens\":null,\"model\":\"gpt-4-turbo-2024-04-09\",\"output\":[],\"parallel_tool_calls\":true,\"previous_response_id\":null,\"reasoning\":{\"effort\":null,\"summary\":null},\"service_tier\":\"auto\",\"store\":true,\"temperature\":1.0,\"text\":{\"format\":{\"type\":\"text\"}},\"tool_choice\":\"auto\",\"tools\":[],\"top_p\":1.0,\"truncation\":\"disabled\",\"usage\":null,\"user\":null,\"metadata\":{}}}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "response" => %{
      "background" => false,
      "created_at" => 1749097929,
      "error" => nil,
      "id" => "resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea",
      "incomplete_details" => nil,
      "instructions" => nil,
      "max_output_tokens" => nil,
      "metadata" => %{},
      "model" => "gpt-4-turbo-2024-04-09",
      "object" => "response",
      "output" => [],
      "parallel_tool_calls" => true,
      "previous_response_id" => nil,
      "reasoning" => %{"effort" => nil, "summary" => nil},
      "service_tier" => "auto",
      "status" => "in_progress",
      "store" => true,
      "temperature" => 1.0,
      "text" => %{"format" => %{"type" => "text"}},
      "tool_choice" => "auto",
      "tools" => [],
      "top_p" => 1.0,
      "truncation" => "disabled",
      "usage" => nil,
      "user" => nil
    },
    "sequence_number" => 0,
    "type" => "response.created"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"response" => %{"background" => false, "created_at" => 1749097929, "error" => nil, "id" => "resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea", "incomplete_details" => nil, "instructions" => nil, "max_output_tokens" => nil, "metadata" => %{}, "model" => "gpt-4-turbo-2024-04-09", "object" => "response", "output" => [], "parallel_tool_calls" => true, "previous_response_id" => nil, "reasoning" => %{"effort" => nil, "summary" => nil}, "service_tier" => "auto", "status" => "in_progress", "store" => true, "temperature" => 1.0, "text" => %{"format" => %{"type" => "text"}}, "tool_choice" => "auto", "tools" => [], "top_p" => 1.0, "truncation" => "disabled", "usage" => nil, "user" => nil}, "sequence_number" => 0, "type" => "response.created"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.in_progress\ndata: {\"type\":\"response.in_progress\",\"sequence_number\":1,\"response\":{\"id\":\"resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea\",\"object\":\"response\",\"created_at\":1749097929,\"status\":\"in_progress\",\"background\":false,\"error\":null,\"incomplete_details\":null,\"instructions\":null,\"max_output_tokens\":null,\"model\":\"gpt-4-turbo-2024-04-09\",\"output\":[],\"parallel_tool_calls\":true,\"previous_response_id\":null,\"reasoning\":{\"effort\":null,\"summary\":null},\"service_tier\":\"auto\",\"store\":true,\"temperature\":1.0,\"text\":{\"format\":{\"type\":\"text\"}},\"tool_choice\":\"auto\",\"tools\":[],\"top_p\":1.0,\"truncation\":\"disabled\",\"usage\":null,\"user\":null,\"metadata\":{}}}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "response" => %{
      "background" => false,
      "created_at" => 1749097929,
      "error" => nil,
      "id" => "resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea",
      "incomplete_details" => nil,
      "instructions" => nil,
      "max_output_tokens" => nil,
      "metadata" => %{},
      "model" => "gpt-4-turbo-2024-04-09",
      "object" => "response",
      "output" => [],
      "parallel_tool_calls" => true,
      "previous_response_id" => nil,
      "reasoning" => %{"effort" => nil, "summary" => nil},
      "service_tier" => "auto",
      "status" => "in_progress",
      "store" => true,
      "temperature" => 1.0,
      "text" => %{"format" => %{"type" => "text"}},
      "tool_choice" => "auto",
      "tools" => [],
      "top_p" => 1.0,
      "truncation" => "disabled",
      "usage" => nil,
      "user" => nil
    },
    "sequence_number" => 1,
    "type" => "response.in_progress"
  }
]

[error] Trying to process an unexpected response. %{"response" => %{"background" => false, "created_at" => 1749097929, "error" => nil, "id" => "resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea", "incomplete_details" => nil, "instructions" => nil, "max_output_tokens" => nil, "metadata" => %{}, "model" => "gpt-4-turbo-2024-04-09", "object" => "response", "output" => [], "parallel_tool_calls" => true, "previous_response_id" => nil, "reasoning" => %{"effort" => nil, "summary" => nil}, "service_tier" => "auto", "status" => "in_progress", "store" => true, "temperature" => 1.0, "text" => %{"format" => %{"type" => "text"}}, "tool_choice" => "auto", "tools" => [], "top_p" => 1.0, "truncation" => "disabled", "usage" => nil, "user" => nil}, "sequence_number" => 1, "type" => "response.in_progress"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_item.added\ndata: {\"type\":\"response.output_item.added\",\"sequence_number\":2,\"output_index\":0,\"item\":{\"id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"type\":\"message\",\"status\":\"in_progress\",\"content\":[],\"role\":\"assistant\"}}\n\nevent: response.content_part.added\ndata: {\"type\":\"response.content_part.added\",\"sequence_number\":3,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"annotations\":[],\"text\":\"\"}}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "item" => %{
      "content" => [],
      "id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
      "role" => "assistant",
      "status" => "in_progress",
      "type" => "message"
    },
    "output_index" => 0,
    "sequence_number" => 2,
    "type" => "response.output_item.added"
  },
  %{
    "content_index" => 0,
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "part" => %{"annotations" => [], "text" => "", "type" => "output_text"},
    "sequence_number" => 3,
    "type" => "response.content_part.added"
  }
]

[error] Trying to process an unexpected response. %{"item" => %{"content" => [], "id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "role" => "assistant", "status" => "in_progress", "type" => "message"}, "output_index" => 0, "sequence_number" => 2, "type" => "response.output_item.added"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "part" => %{"annotations" => [], "text" => "", "type" => "output_text"}, "sequence_number" => 3, "type" => "response.content_part.added"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":4,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"Hello\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "Hello",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 4,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "Hello", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 4, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":5,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" there\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " there",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 5,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " there", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 5, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":6,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"!\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "!",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 6,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "!", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 6, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":7,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" I\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " I",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 7,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " I", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 7, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":8,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"'m\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "'m",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 8,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "'m", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 8, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":9,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" always\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":10,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" here\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":11,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" to\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":12,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" help\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " always",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 9,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " here",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 10,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " to",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 11,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " help",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 12,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " always", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 9, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " here", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 10, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " to", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 11, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":13,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" you\"}\n\n"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " help", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 12, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " you",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 13,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " you", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 13, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":14,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" swim\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " swim",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 14,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " swim", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 14, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":15,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" through\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":16,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" any\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " through",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 15,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " any",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 16,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " through", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 15, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " any", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 16, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":17,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" information\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":18,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" ocean\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " information",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 17,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " ocean",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 18,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " information", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 17, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " ocean", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 18, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":19,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" or\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":20,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" dive\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " or",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 19,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " dive",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 20,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " or", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 19, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " dive", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 20, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":21,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" deep\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":22,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" into\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " deep",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 21,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " into",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 22,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " deep", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 21, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " into", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 22, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":23,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" the\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":24,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" sea\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " the",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 23,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " sea",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 24,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " the", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 23, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " sea", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 24, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":25,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" of\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":26,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" knowledge\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " of",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 25,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " knowledge",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 26,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " of", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 25, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " knowledge", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 26, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":27,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\".\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => ".",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 27,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => ".", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 27, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":28,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" What\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " What",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 28,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " What", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 28, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":29,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"’s\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "’s",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 29,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "’s", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 29, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":30,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" on\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " on",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 30,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " on", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 30, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":31,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" your\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " your",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 31,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " your", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 31, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":32,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" mind\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":33,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" today\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " mind",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 32,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => " today",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 33,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " mind", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 32, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " today", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 33, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":34,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"?\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "?",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 34,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "?", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 34, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":35,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" Let\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " Let",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 35,
    "type" => "response.output_text.delta"
  }
]

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " Let", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 35, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":36,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"'s\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => "'s",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 36,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "'s", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 36, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":37,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" make\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " make",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 37,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " make", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 37, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":38,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" a\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " a",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 38,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " a", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 38, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":39,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\" splash\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"sequence_number\":40,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"delta\":\"!\"}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "delta" => " splash",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 39,
    "type" => "response.output_text.delta"
  },
  %{
    "content_index" => 0,
    "delta" => "!",
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 40,
    "type" => "response.output_text.delta"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => " splash", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 39, "type" => "response.output_text.delta"}
[error] Trying to process an unexpected response. %{"content_index" => 0, "delta" => "!", "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 40, "type" => "response.output_text.delta"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"sequence_number\":41,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"text\":\"Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!\"}\n\nevent: response.content_part.done\ndata: {\"type\":\"response.content_part.done\",\"sequence_number\":42,\"item_id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"output_index\":0,\"content_index\":0,\"part\":{\"type\":\"output_text\",\"annotations\":[],\"text\":\"Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!\"}}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "content_index" => 0,
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "sequence_number" => 41,
    "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!",
    "type" => "response.output_text.done"
  },
  %{
    "content_index" => 0,
    "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
    "output_index" => 0,
    "part" => %{
      "annotations" => [],
      "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!",
      "type" => "output_text"
    },
    "sequence_number" => 42,
    "type" => "response.content_part.done"
  }
]

[error] Trying to process an unexpected response. %{"content_index" => 0, "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "sequence_number" => 41, "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!", "type" => "response.output_text.done"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:828: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
"decode_stream" #=> "decode_stream"

[error] Trying to process an unexpected response. %{"content_index" => 0, "item_id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea", "output_index" => 0, "part" => %{"annotations" => [], "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!", "type" => "output_text"}, "sequence_number" => 42, "type" => "response.content_part.done"}
[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:829: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
raw_data #=> "event: response.output_item.done\ndata: {\"type\":\"response.output_item.done\",\"sequence_number\":43,\"output_index\":0,\"item\":{\"id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"type\":\"message\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"annotations\":[],\"text\":\"Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!\"}],\"role\":\"assistant\"}}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":44,\"response\":{\"id\":\"resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea\",\"object\":\"response\",\"created_at\":1749097929,\"status\":\"completed\",\"background\":false,\"error\":null,\"incomplete_details\":null,\"instructions\":null,\"max_output_tokens\":null,\"model\":\"gpt-4-turbo-2024-04-09\",\"output\":[{\"id\":\"msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea\",\"type\":\"message\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"annotations\":[],\"text\":\"Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!\"}],\"role\":\"assistant\"}],\"parallel_tool_calls\":true,\"previous_response_id\":null,\"reasoning\":{\"effort\":null,\"summary\":null},\"service_tier\":\"default\",\"store\":true,\"temperature\":1.0,\"text\":{\"format\":{\"type\":\"text\"}},\"tool_choice\":\"auto\",\"tools\":[],\"top_p\":1.0,\"truncation\":\"disabled\",\"usage\":{\"input_tokens\":30,\"input_tokens_details\":{\"cached_tokens\":0},\"output_tokens\":38,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":68},\"user\":null,\"metadata\":{}}}\n\n"

[(langchain 0.4.0-rc.0) lib/chat_models/chat_open_ai_responses.ex:830: LangChain.ChatModels.ChatOpenAIResponses.decode_stream/2]
done #=> []

[(langchain 0.4.0-rc.0) lib/utils.ex:190: LangChain.Utils.handle_stream_fn/3]
"parsed_data" #=> "parsed_data"

[(langchain 0.4.0-rc.0) lib/utils.ex:191: LangChain.Utils.handle_stream_fn/3]
parsed_data #=> [
  %{
    "item" => %{
      "content" => [
        %{
          "annotations" => [],
          "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!",
          "type" => "output_text"
        }
      ],
      "id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
      "role" => "assistant",
      "status" => "completed",
      "type" => "message"
    },
    "output_index" => 0,
    "sequence_number" => 43,
    "type" => "response.output_item.done"
  },
  %{
    "response" => %{
      "background" => false,
      "created_at" => 1749097929,
      "error" => nil,
      "id" => "resp_68411dc97d608198b7e019307faaa90a0390fe658f11abea",
      "incomplete_details" => nil,
      "instructions" => nil,
      "max_output_tokens" => nil,
      "metadata" => %{},
      "model" => "gpt-4-turbo-2024-04-09",
      "object" => "response",
      "output" => [
        %{
          "content" => [
            %{
              "annotations" => [],
              "text" => "Hello there! I'm always here to help you swim through any information ocean or dive deep into the sea of knowledge. What’s on your mind today? Let's make a splash!",
              "type" => "output_text"
            }
          ],
          "id" => "msg_68411dc9ec3c8198a5a727d3c60b68170390fe658f11abea",
          "role" => "assistant",
          "status" => "completed",
          "type" => "message"
        }
      ],
      "parallel_tool_calls" => true,
      "previous_response_id" => nil,
      "reasoning" => %{"effort" => nil, "summary" => nil},
      "service_tier" => "default",
      "status" => "completed",
      "store" => true,
      "temperature" => 1.0,
      "text" => %{"format" => %{"type" => "text"}},
      "tool_choice" => "auto",
      "tools" => [],
      "top_p" => 1.0,
      "truncation" => "disabled",
      "usage" => %{
        "input_tokens" => 30,
        "input_tokens_details" => %{"cached_tokens" => 0},
        "output_tokens" => 38,
        "output_tokens_details" => %{"reasoning_tokens" => 0},
        "total_tokens" => 68
      },
      "user" => nil
    },
    "sequence_number" => 44,
    "type" => "response.completed"
  }
]
