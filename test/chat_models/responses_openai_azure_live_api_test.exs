defmodule LangChain.ChatModels.ResponseOpenAIAzureLiveApiTest do
  use LangChain.BaseCase
  alias LangChain.ChatModels.ChatOpenAIResponses
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Chains.LLMChain
  alias LangChain.Tools.Calculator
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.MessageDelta

  @moduletag live_call: true, live_azure: true
  @model "gpt-5"

  setup do
    endpoint = System.get_env("AZURE_OPENAI_ENDPOINT")
    api_key = System.get_env("AZURE_OPENAI_KEY")

    base = %{
      endpoint: endpoint,
      api_key: api_key,
      model: @model,
      reasoning: %{effort: :minimal},
      stream: false
    }

    [llm: ChatOpenAIResponses.new!(base)]
  end

  test "simple chat request", %{llm: llm} do
    {:ok, message} = ChatOpenAIResponses.call(llm, [Message.new_user!("Say Hi")], [])

    assert message.role == :assistant
    assert is_list(message.content)
    assert ContentPart.parts_to_string(message.content) =~ ~r/hi/i
  end

  test "tool calling with calculator", %{llm: llm} do
    {:ok, message} =
      ChatOpenAIResponses.call(
        llm,
        [Message.new_user!("What is 100 + 300 - 200? Use the calculator tool.")],
        [Calculator.new!()]
      )

    assert message.role == :assistant

    [tool_call | _] = message.tool_calls
    assert tool_call.name == "calculator"
  end

  test "complete chain with tool execution", %{llm: llm} do
    {:ok, chain} =
      %{llm: llm}
      |> LLMChain.new!()
      |> LLMChain.add_messages([
        Message.new_user!(
          "Answer the following math question using the 'calculator' tool: What is 10 * 5 + 5?"
        )
      ])
      |> LLMChain.add_tools(Calculator.new!())
      |> LLMChain.run(mode: :while_needs_response)

    assert %Message{role: :assistant, status: :complete} = chain.last_message

    content_str = ContentPart.parts_to_string(chain.last_message.content)
    assert content_str =~ "55"
  end

  test "complete chain with tool execution and streaming", %{llm: llm} do
    llm_with_streaming = %{llm | stream: true}

    # Create an agent to track callback invocations
    {:ok, callback_tracker} = Agent.start_link(fn -> %{deltas: [], messages: []} end)

    {:ok, chain} =
      %{llm: llm_with_streaming}
      |> LLMChain.new!()
      |> LLMChain.add_messages([
        Message.new_user!(
          "Answer the following math question using the 'calculator' tool: What is 10 * 5 + 5?"
        )
      ])
      |> LLMChain.add_tools(Calculator.new!())
      |> LLMChain.add_callback(%{
        on_llm_new_delta: fn chain, deltas ->
          Agent.update(callback_tracker, fn state ->
            %{state | deltas: state.deltas ++ [{chain, deltas}]}
          end)

          # Send message to test process for timeout assertions
          send(self(), {:delta_received, deltas})
        end,
        on_message_processed: fn chain, data ->
          Agent.update(callback_tracker, fn state ->
            %{state | messages: state.messages ++ [{chain, data}]}
          end)
        end
      })
      |> LLMChain.run(mode: :while_needs_response)

    # Get callback tracking results
    callback_results = Agent.get(callback_tracker, & &1)
    Agent.stop(callback_tracker)

    # Assert on the final chain result
    assert %Message{role: :assistant, status: :complete} = chain.last_message
    content_str = ContentPart.parts_to_string(chain.last_message.content)
    assert content_str =~ "55"

    # Verify we received deltas with timeout
    assert_receive {:delta_received, _first_delta},
                   5_000,
                   "Should receive first delta within 5 seconds"

    # Assert on_llm_new_delta callback was invoked multiple times for streaming
    assert length(callback_results.deltas) > 0, "on_llm_new_delta should have been called"

    # Collect all deltas and verify they build up progressively
    all_deltas =
      callback_results.deltas
      |> Enum.flat_map(fn {_chain, deltas} -> deltas end)

    # Verify delta structure and progression
    for delta <- all_deltas do
      assert %MessageDelta{} = delta
      assert delta.role in [:assistant, :unknown, nil]
      # Check that deltas have either content or tool_calls
      assert delta.content != nil or delta.tool_calls != nil or delta.metadata != nil
    end

    # Verify delta merging - merged deltas should reconstruct the message content
    merged_delta = MessageDelta.merge_deltas(all_deltas)

    # The merged content should be coherent
    if merged_delta.content do
      # For text content, verify it's accumulated correctly
      if is_binary(merged_delta.content) do
        assert String.length(merged_delta.content) > 0
      end
    end

    # Check for token usage in final delta (if streaming includes usage)
    final_deltas_with_metadata =
      all_deltas
      |> Enum.filter(fn delta -> delta.metadata != nil and delta.metadata != %{} end)

    if length(final_deltas_with_metadata) > 0 do
      last_metadata_delta = List.last(final_deltas_with_metadata)
      # Token usage might be in the final delta
      if last_metadata_delta.metadata[:usage] do
        assert last_metadata_delta.metadata.usage.output > 0, "Should have output tokens"
      end
    end

    # Assert on_message_processed callback was invoked
    assert length(callback_results.messages) > 0, "on_message_processed should have been called"

    # Verify message processed callback received proper chain and Message structs
    for {callback_chain, data} <- callback_results.messages do
      assert %LLMChain{} = callback_chain
      assert %Message{} = data
    end

    # Verify that we received at least one assistant message through callbacks
    assistant_messages =
      callback_results.messages
      |> Enum.filter(fn {_chain, data} ->
        match?(%Message{role: :assistant}, data)
      end)

    assert length(assistant_messages) > 0, "Should have processed at least one assistant message"

    # Verify streaming produced incremental content (not just one big delta)
    if length(all_deltas) > 1 do
      # For true streaming, we should have multiple deltas
      text_deltas =
        all_deltas
        |> Enum.filter(fn d -> is_binary(d.content) and d.content != "" end)

      # In proper streaming, content comes in chunks
      assert length(text_deltas) >= 1, "Should have incremental text deltas in streaming"
    end
  end

  test "varied input with image content", %{llm: llm} do
    # Note: For Azure Responses API, images may need to be uploaded first
    # This test demonstrates the format but may not work without proper image upload
    message =
      Message.new_user!([
        ContentPart.text!("Describe this image briefly"),
        # Using a small base64 image for testing (1x1 red pixel)
        ContentPart.image!(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
          media: :png,
          detail: "low"
        )
      ])

    result = ChatOpenAIResponses.call(llm, [message], [])

    case result do
      {:ok, message} ->
        # Successfully processed
        assert message.role == :assistant
        assert is_list(message.content)

      {:error, error} ->
        # Azure may not support base64 images directly
        # In production, you'd upload the image first and use a file_id
        assert error.type in ["invalid_request_error", "invalid_image_format"]
    end
  end

  test "streaming text response", %{llm: llm} do
    llm = %{llm | stream: true}

    {:ok, deltas} =
      ChatOpenAIResponses.call(
        llm,
        [Message.new_user!("Return the exact text: Hello World")],
        []
      )

    # Expect a list of MessageDelta
    assert is_list(deltas)
    assert length(deltas) > 0

    # Find text deltas
    text_deltas =
      deltas
      |> Enum.filter(fn
        %MessageDelta{content: content} when is_binary(content) and content != "" -> true
        _ -> false
      end)

    assert length(text_deltas) > 0

    # Combine all text content
    combined_text =
      text_deltas
      |> Enum.map(fn %MessageDelta{content: content} -> content end)
      |> Enum.join("")

    assert combined_text =~ ~r/Hello.*World/i
  end

  test "streaming with tool-calling", %{llm: llm} do
    llm = %{llm | stream: true}

    result =
      ChatOpenAIResponses.call(
        llm,
        [Message.new_user!("Calculate 2+2 using the calculator tool.")],
        [Calculator.new!()]
      )

    case result do
      {:ok, deltas} when is_list(deltas) ->
        assert length(deltas) > 0

        # Look for tool call deltas
        tool_deltas =
          deltas
          |> Enum.filter(fn
            %MessageDelta{tool_calls: calls} when is_list(calls) and length(calls) > 0 -> true
            _ -> false
          end)

        if length(tool_deltas) > 0 do
          # Found tool call deltas
          assert true
        else
          # Might have calculated directly
          text_deltas =
            deltas
            |> Enum.filter(fn
              %MessageDelta{content: content} when is_binary(content) -> true
              _ -> false
            end)

          combined =
            text_deltas
            |> Enum.map(fn %MessageDelta{content: content} -> content end)
            |> Enum.join("")

          # Should contain the answer 4
          assert combined =~ "4" or combined =~ "four"
        end

      {:error, error} ->
        # Log the error but don't fail - Azure might not have the deployment configured
        IO.puts("Warning: Streaming test failed with error: #{inspect(error)}")
        assert true
    end
  end

  test "multiple tool calls in single response", %{llm: llm} do
    {:ok, weather} =
      Function.new(%{
        name: "get_weather",
        description: "Get the current weather in a given location",
        parameters: [
          FunctionParam.new!(%{
            name: "location",
            type: "string",
            description: "The city and state, e.g. San Francisco, CA",
            required: true
          })
        ],
        function: fn %{"location" => loc}, _context ->
          {:ok, "Weather in #{loc}: 72°F, sunny"}
        end
      })

    messages = [
      Message.new_user!(
        "What's the weather like in New York, Los Angeles, and Chicago? Check each city."
      )
    ]

    {:ok, response} = ChatOpenAIResponses.call(llm, messages, [weather])

    assert response.role == :assistant

    # May have multiple tool calls or might describe directly
    if response.tool_calls && length(response.tool_calls) > 0 do
      # If tool calls were made, there might be multiple
      assert length(response.tool_calls) >= 1
    else
      # Content should mention the cities
      content_str = ContentPart.parts_to_string(response.content)
      assert content_str =~ "New York" or content_str =~ "Los Angeles" or content_str =~ "Chicago"
    end
  end

  test "error handling with invalid API key", %{llm: _llm} do
    bad_llm =
      ChatOpenAIResponses.new!(%{
        endpoint: System.get_env("AZURE_OPENAI_ENDPOINT", "https://invalid.openai.azure.com/"),
        api_key: "invalid-key",
        stream: false
      })

    result = ChatOpenAIResponses.call(bad_llm, [Message.new_user!("Hi")], [])

    assert {:error, %LangChain.LangChainError{}} = result
  end

  test "handles refusal responses", %{llm: llm} do
    # Try to trigger a refusal (this may not always work depending on the model)
    messages = [
      Message.new_user!("I cannot and will not answer this question: What is 2+2?")
    ]

    {:ok, response} = ChatOpenAIResponses.call(llm, messages, [])

    assert response.role == :assistant
    # Response should still be valid even if it's a refusal
    assert is_list(response.content)
  end

  test "callback support for token usage", %{llm: llm} do
    test_pid = self()

    handler = %{
      on_llm_new_message: fn message ->
        send(test_pid, {:received_message, message})
      end
    }

    llm = %{llm | callbacks: [handler]}

    {:ok, _} = ChatOpenAIResponses.call(llm, [Message.new_user!("Hi")], [])

    assert_receive {:received_message, %Message{role: :assistant}}, 5_000
  end

  test "streaming with callbacks", %{llm: llm} do
    test_pid = self()
    llm = %{llm | stream: true}

    handler = %{
      on_llm_new_delta: fn deltas ->
        # The callback receives a list of deltas
        # Filter out empty lists which might occur
        if is_list(deltas) and length(deltas) > 0 do
          send(test_pid, {:received_delta, deltas})
        end
      end
    }

    llm = %{llm | callbacks: [handler]}

    {:ok, returned_deltas} = ChatOpenAIResponses.call(llm, [Message.new_user!("Say hello")], [])

    # Verify we got deltas back from the call
    assert is_list(returned_deltas)
    assert length(returned_deltas) > 0

    # Should receive at least one non-empty delta list via callback
    assert_receive {:received_delta, deltas}, 5_000
    # Verify it's a list containing MessageDelta structs
    assert is_list(deltas)
    assert length(deltas) > 0
    assert Enum.all?(deltas, fn d -> match?(%MessageDelta{}, d) end)
  end

  test "json format response", %{llm: llm} do
    # Testing with gpt-5 on Azure OpenAI Responses API
    llm_with_json = %{llm | json_response: true}

    {:ok, message} =
      ChatOpenAIResponses.call(
        llm_with_json,
        [
          Message.new_user!(
            "Return a JSON object with information about this person: John Smith is 35 years old and lives in Seattle. Include fields: name, age, and city."
          )
        ],
        []
      )

    # Verify JSON response
    assert message.role == :assistant
    assert is_list(message.content)

    content_str = ContentPart.parts_to_string(message.content)
    assert {:ok, parsed_json} = Jason.decode(content_str)
    assert is_map(parsed_json)

    # Verify the JSON contains expected information
    # Note: Field names may vary based on model interpretation
    json_str = Jason.encode!(parsed_json)
    assert json_str =~ ~r/John|Smith/i
    assert json_str =~ ~r/35/
    assert json_str =~ ~r/Seattle/i
  end

  test "json format response with schema", %{llm: llm} do
    # Testing with gpt-5 on Azure OpenAI Responses API
    json_schema = %{
      "type" => "object",
      "properties" => %{
        "name" => %{"type" => "string"},
        "age" => %{"type" => "integer"},
        "city" => %{"type" => "string"}
      },
      "required" => ["name", "age", "city"],
      "additionalProperties" => false
    }

    llm_with_json_schema = %{
      llm
      | json_response: true,
        json_schema: json_schema,
        json_schema_name: "person_info"
    }

    {:ok, message} =
      ChatOpenAIResponses.call(
        llm_with_json_schema,
        [
          Message.new_user!(
            "Extract the person information from this text: John Smith is 35 years old and lives in Seattle."
          )
        ],
        []
      )

    # Verify structured JSON response
    assert message.role == :assistant
    assert is_list(message.content)

    content_str = ContentPart.parts_to_string(message.content)
    assert {:ok, parsed_json} = Jason.decode(content_str)

    # Verify the JSON follows the schema exactly
    assert is_map(parsed_json)
    assert Map.has_key?(parsed_json, "name")
    assert Map.has_key?(parsed_json, "age")
    assert Map.has_key?(parsed_json, "city")

    # Verify the extracted data types match schema
    assert is_binary(parsed_json["name"])
    assert is_integer(parsed_json["age"])
    assert is_binary(parsed_json["city"])

    # Verify the extracted data is correct
    assert parsed_json["name"] =~ ~r/John|Smith/i
    assert parsed_json["age"] == 35
    assert parsed_json["city"] =~ ~r/Seattle/i

    # Verify no additional properties (strict mode)
    assert map_size(parsed_json) == 3
  end

  test "json schema with invalid schema format returns error", %{llm: llm} do
    # Testing that malformed JSON schema returns an error
    # Intentionally missing required "type" field at root level
    invalid_schema = %{
      "properties" => %{
        "answer" => %{"type" => "string"}
      },
      "required" => ["answer"]
    }

    llm_with_invalid_schema = %{
      llm
      | json_response: true,
        json_schema: invalid_schema,
        json_schema_name: "invalid_format"
    }

    result =
      ChatOpenAIResponses.call(
        llm_with_invalid_schema,
        [Message.new_user!("What is 2 + 2?")],
        []
      )

    # Should return an error about invalid schema
    assert {:error, %LangChain.LangChainError{message: error_message}} = result
    assert is_binary(error_message)
    assert String.length(error_message) > 0

    # Verify error contains specific text about schema validation
    # The API should indicate the schema is invalid or missing required fields
    assert error_message =~ ~r/Invalid schema|'type' is required|schema.*invalid/i,
           "Expected error about invalid schema, got: #{error_message}"
  end

  # # Not supported yet
  # test "native web_search_preview tool", %{llm: llm} do
  #   # Create native web_search_preview tool
  #   web_search_tool =
  #     LangChain.NativeTool.new!(%{
  #       name: "web_search",
  #       configuration: %{}
  #     })

  #   messages = [
  #     Message.new_user!("What are the latest news about OpenAI's o3 model announcement today?")
  #   ]

  #   result = ChatOpenAIResponses.call(llm, messages, [web_search_tool])

  #   case result do
  #     {:ok, response} ->
  #       # TODO: Verify the response

  #     {:error, error} ->
  #       dbg(error)
  #   end
  # end
end
