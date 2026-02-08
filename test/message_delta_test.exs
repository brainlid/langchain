defmodule LangChain.MessageDeltaTest do
  use ExUnit.Case
  doctest LangChain.MessageDelta, import: true
  import LangChain.Fixtures
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.Message.ToolCall
  alias LangChain.LangChainError

  describe "new/1" do
    test "works with minimal attrs" do
      assert {:ok, %MessageDelta{} = msg} = MessageDelta.new(%{})
      assert msg.role == :unknown
      assert msg.content == nil
      assert msg.merged_content == []
      assert msg.status == :incomplete
      assert msg.index == nil
    end

    test "accepts normal content attributes" do
      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => "Hi!",
                 "role" => "assistant",
                 "index" => 1,
                 "status" => "complete"
               })

      assert msg.role == :assistant
      assert msg.content == "Hi!"
      assert msg.merged_content == []
      assert msg.status == :complete
      assert msg.index == 1
    end

    test "accepts tool_call attributes" do
      tool_call =
        ToolCall.new!(%{
          type: :function,
          name: "hello_world",
          call_id: "call_123",
          arguments: Jason.encode!(%{greeting: "Howdy"})
        })

      assert {:ok, %MessageDelta{} = msg} =
               MessageDelta.new(%{
                 "content" => nil,
                 "role" => "assistant",
                 "tool_calls" => [tool_call],
                 "index" => 1,
                 "status" => "complete"
               })

      assert msg.role == :assistant
      assert msg.content == nil
      assert msg.merged_content == []
      assert msg.tool_calls == [tool_call]
      assert msg.status == :complete
      assert msg.index == 1
    end

    test "returns error when invalid" do
      assert {:error, changeset} = MessageDelta.new(%{role: "invalid", index: "abc"})
      assert {"is invalid", _} = changeset.errors[:role]
      assert {"is invalid", _} = changeset.errors[:index]
    end

    test "accepts receiving thinking content as a list of content parts" do
      result =
        MessageDelta.new!(%{
          content: [
            ContentPart.new!(%{
              type: :thinking,
              content: "Let's add these numbers.",
              options: nil
            })
          ],
          role: :assistant
        })

      assert %MessageDelta{
               content: [
                 %ContentPart{type: :thinking, options: nil, content: "Let's add these numbers."}
               ],
               merged_content: [],
               role: :assistant
             } == result
    end
  end

  describe "new!/1" do
    test "returns struct when valid" do
      assert %MessageDelta{
               role: :assistant,
               content: "Hi!",
               merged_content: [],
               status: :incomplete
             } =
               MessageDelta.new!(%{
                 "content" => "Hi!",
                 "role" => "assistant"
               })
    end

    test "raises exception when invalid" do
      assert_raise LangChainError, "role: is invalid; index: is invalid", fn ->
        MessageDelta.new!(%{role: "invalid", index: "abc"})
      end
    end
  end

  describe "merge_delta/2" do
    test "handles merging when no existing delta to merge into" do
      delta = %MessageDelta{
        content: ContentPart.text!("Hello! How can I assist you today?"),
        index: 0,
        role: :assistant,
        status: :incomplete
      }

      merged = MessageDelta.merge_delta(nil, delta)
      assert merged.content == nil
      assert merged.merged_content == [ContentPart.text!("Hello! How can I assist you today?")]
      assert merged.index == 0
      assert merged.role == :assistant
      assert merged.status == :incomplete
    end

    test "migrates string content for first received delta" do
      delta = %MessageDelta{
        content: "Hello! How can I assist you today?",
        index: 0,
        role: :assistant,
        status: :incomplete
      }

      expected = %MessageDelta{
        content: nil,
        merged_content: [ContentPart.text!("Hello! How can I assist you today?")],
        index: 0,
        role: :assistant,
        status: :incomplete
      }

      assert expected == MessageDelta.merge_delta(nil, delta)
    end

    test "correctly merges assistant content message" do
      merged = MessageDelta.merge_deltas(delta_content_sample())

      expected = %LangChain.MessageDelta{
        content: nil,
        merged_content: [ContentPart.text!("Hello! How can I assist you today?")],
        index: 0,
        role: :assistant,
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merge multiple tool calls in a delta" do
      merged = MessageDelta.merge_deltas(deltas_for_multiple_tool_calls())

      expected = %MessageDelta{
        content: nil,
        merged_content: [],
        index: 0,
        tool_calls: [
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_123",
            name: "get_weather",
            arguments: "{\"city\": \"Moab\", \"state\": \"UT\"}",
            index: 0
          },
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_234",
            name: "get_weather",
            arguments: "{\"city\": \"Portland\", \"state\": \"OR\"}",
            index: 1
          },
          %ToolCall{
            status: :incomplete,
            type: :function,
            call_id: "call_345",
            name: "get_weather",
            arguments: "{\"city\": \"Baltimore\", \"state\": \"MD\"}",
            index: 2
          }
        ],
        role: :assistant,
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merges assistant content with a tool_call" do
      merged = delta_content_with_function_call() |> List.flatten() |> MessageDelta.merge_deltas()

      expected = %LangChain.MessageDelta{
        content: nil,
        merged_content: [
          ContentPart.text!(
            "Sure, I can help with that. First, let's check which regions are currently available for deployment on Fly.io. Please wait a moment while I fetch this information for you."
          )
        ],
        index: 0,
        tool_calls: [
          ToolCall.new!(%{call_id: "call_123", name: "regions_list", arguments: "{}", index: 0})
        ],
        role: :assistant,
        status: :complete
      }

      assert merged == expected
    end

    test "correctly merge message with tool_call containing empty spaces" do
      deltas = [
        %LangChain.MessageDelta{
          content: "",
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: nil
        },
        %LangChain.MessageDelta{
          content: "stu",
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: nil
        },
        %LangChain.MessageDelta{
          content: "ff",
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: nil
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "toolu_123",
              name: "get_code",
              arguments: nil,
              index: 1
            }
          ]
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "toolu_123",
              name: "get_code",
              arguments: "{\"code\": \"def my_function(x):\n ",
              index: 1
            }
          ]
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "toolu_123",
              name: "get_code",
              arguments: " ",
              index: 1
            }
          ]
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "toolu_123",
              name: "get_code",
              arguments: "  ",
              index: 1
            }
          ]
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: [
            %LangChain.Message.ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "toolu_123",
              name: "get_code",
              arguments: "return x + 1\"}",
              index: 1
            }
          ]
        }
      ]

      merged = MessageDelta.merge_deltas(deltas)

      assert merged == %LangChain.MessageDelta{
               content: nil,
               merged_content: [ContentPart.text!("stuff")],
               status: :incomplete,
               index: nil,
               role: :assistant,
               tool_calls: [
                 %LangChain.Message.ToolCall{
                   status: :incomplete,
                   type: :function,
                   call_id: "toolu_123",
                   name: "get_code",
                   arguments: "{\"code\": \"def my_function(x):\n    return x + 1\"}",
                   index: 1
                 }
               ]
             }
    end

    test "correctly merges message with tool_call split over multiple deltas and index is not by position" do
      deltas =
        [
          %LangChain.MessageDelta{
            content: "",
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: nil
          },
          %LangChain.MessageDelta{
            content: "stu",
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: nil
          },
          %LangChain.MessageDelta{
            content: "ff",
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: nil
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: "toolu_123",
                name: "do_something",
                arguments: nil,
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: nil,
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "{\"",
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: "value\":",
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: " \"People",
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: nil,
            status: :incomplete,
            index: nil,
            role: :assistant,
            tool_calls: [
              %LangChain.Message.ToolCall{
                status: :incomplete,
                type: :function,
                call_id: nil,
                name: nil,
                arguments: " are people.\"}",
                index: 1
              }
            ]
          },
          %LangChain.MessageDelta{
            content: "",
            status: :complete,
            index: nil,
            role: :assistant,
            tool_calls: nil
          }
        ]

      combined = MessageDelta.merge_deltas(deltas)

      assert combined == %LangChain.MessageDelta{
               content: nil,
               merged_content: [ContentPart.text!("stuff")],
               status: :complete,
               index: nil,
               role: :assistant,
               tool_calls: [
                 %LangChain.Message.ToolCall{
                   status: :incomplete,
                   type: :function,
                   call_id: "toolu_123",
                   name: "do_something",
                   arguments: "{\"value\": \"People are people.\"}",
                   index: 1
                 }
               ]
             }

      # should correctly convert to a message
      {:ok, message} = MessageDelta.to_message(combined)

      assert message == %Message{
               content: [ContentPart.text!("stuff")],
               status: :complete,
               role: :assistant,
               tool_calls: [
                 %LangChain.Message.ToolCall{
                   status: :complete,
                   type: :function,
                   call_id: "toolu_123",
                   name: "do_something",
                   arguments: %{"value" => "People are people."},
                   index: 1
                 }
               ]
             }
    end

    test "handles merging in a thinking content part" do
      delta_1 = %MessageDelta{
        content: [],
        status: :incomplete,
        index: nil,
        role: :assistant
      }

      delta_2 = %MessageDelta{
        content: %ContentPart{
          type: :thinking,
          content: nil,
          options: %{signature: ""}
        },
        status: :incomplete,
        index: 0,
        role: :assistant
      }

      merged = MessageDelta.merge_delta(delta_1, delta_2)

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :thinking,
                   content: nil,
                   options: %{signature: ""}
                 }
               ],
               role: :assistant,
               status: :incomplete,
               index: 0
             }
    end

    test "handles merging a simple set of thinking content parts" do
      merged =
        [
          %MessageDelta{
            content: [],
            status: :incomplete,
            index: nil,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: "Let's think about",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: " this problem",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          }
        ]
        |> MessageDelta.merge_deltas()

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :thinking,
                   content: "Let's think about this problem",
                   options: nil
                 }
               ],
               status: :incomplete,
               index: 0,
               role: :assistant
             }
    end

    test "handles merging a set of thinking content parts with different indexes" do
      merged =
        [
          %MessageDelta{
            content: [],
            status: :incomplete,
            index: nil,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: "First thought",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: "Second thought",
              options: nil
            },
            status: :incomplete,
            index: 1,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: " More first thought",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          }
        ]
        |> MessageDelta.merge_deltas()

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :thinking,
                   content: "First thought More first thought",
                   options: nil
                 },
                 %ContentPart{
                   type: :thinking,
                   content: "Second thought",
                   options: nil
                 }
               ],
               status: :incomplete,
               index: 0,
               role: :assistant
             }
    end

    test "handles merging content parts with gaps in indices" do
      merged =
        [
          %MessageDelta{
            content: [],
            status: :incomplete,
            index: nil,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :text,
              content: "First part",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :text,
              content: "Third part",
              options: nil
            },
            status: :incomplete,
            index: 2,
            role: :assistant
          }
        ]
        |> MessageDelta.merge_deltas()

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :text,
                   content: "First part",
                   options: nil
                 },
                 nil,
                 %ContentPart{
                   type: :text,
                   content: "Third part",
                   options: nil
                 }
               ],
               status: :incomplete,
               index: 2,
               role: :assistant
             }
    end

    test "handles merging content parts with different types" do
      delta_1 = %MessageDelta{
        content: [],
        status: :incomplete,
        index: nil,
        role: :assistant
      }

      delta_2 = %MessageDelta{
        content: %ContentPart{
          type: :text,
          content: "Text content",
          options: nil
        },
        status: :incomplete,
        index: 0,
        role: :assistant
      }

      delta_3 = %MessageDelta{
        content: %ContentPart{
          type: :thinking,
          content: "Thinking content",
          options: nil
        },
        status: :incomplete,
        index: 1,
        role: :assistant
      }

      merged =
        delta_1
        |> MessageDelta.merge_delta(delta_2)
        |> MessageDelta.merge_delta(delta_3)

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :text,
                   content: "Text content",
                   options: nil
                 },
                 %ContentPart{
                   type: :thinking,
                   content: "Thinking content",
                   options: nil
                 }
               ],
               status: :incomplete,
               index: 1,
               role: :assistant
             }
    end

    test "handles merging a thinking part with the signature" do
      merged =
        [
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: "450 + 3 = 453",
              options: nil
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          },
          %MessageDelta{
            content: %ContentPart{
              type: :thinking,
              content: nil,
              options: [
                signature:
                  "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
              ]
            },
            status: :incomplete,
            index: 0,
            role: :assistant
          }
        ]
        |> MessageDelta.merge_deltas()

      assert merged == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{
                   type: :thinking,
                   content: "450 + 3 = 453",
                   options: [
                     signature:
                       "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
                   ]
                 }
               ],
               status: :incomplete,
               index: 0,
               role: :assistant
             }
    end

    # TODO: DO NOT DELETE. THIS IS IMPORTANT.
    @tag live_test: true
    test "correctly merges thinking deltas with signature and usage" do
      deltas = [
        %LangChain.MessageDelta{
          content: [],
          status: :incomplete,
          index: nil,
          role: :assistant,
          tool_calls: nil,
          metadata: %{
            usage: %LangChain.TokenUsage{
              input: 55,
              output: 4,
              raw: %{
                "cache_creation_input_tokens" => 0,
                "cache_read_input_tokens" => 0,
                "input_tokens" => 55,
                "output_tokens" => 4
              }
            }
          }
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: "",
            options: [signature: ""]
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: "Let's ad",
            options: nil
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: "d these numbers.\n400 + 50 = 450\n450 ",
            options: nil
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: "+ 3 = 453\n\nSo 400 + 50",
            options: nil
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: " + 3 = 453",
            options: nil
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :thinking,
            content: nil,
            options: [
              signature:
                "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
            ]
          },
          status: :incomplete,
          index: 0,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :text,
            content: "",
            options: []
          },
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :text,
            content: "The answer is 453.\n\n400 + 50 = 450\n450 + 3 =",
            options: []
          },
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: %LangChain.Message.ContentPart{
            type: :text,
            content: " 453",
            options: []
          },
          status: :incomplete,
          index: 1,
          role: :assistant,
          tool_calls: nil,
          metadata: nil
        },
        %LangChain.MessageDelta{
          content: nil,
          status: :complete,
          index: nil,
          role: :assistant,
          tool_calls: nil,
          metadata: %{
            usage: %LangChain.TokenUsage{
              input: nil,
              output: 80,
              raw: %{"output_tokens" => 80}
            }
          }
        }
      ]

      combined = MessageDelta.merge_deltas(deltas)

      # TODO: Release as an RC?
      # TODO: Breaking change for deltas. Allow for legacy deltas to be merged using string contents? Until all the models are updated, it will be broken.
      # TODO: All other models need to be updated to put streamed text content into a ContentPart.
      # TODO: Also supports receiving streamed multi-modal content.
      # TODO: Update the token usage callback event to still fire, but with the fully completed information?

      assert combined == %LangChain.MessageDelta{
               content: nil,
               merged_content: [
                 %LangChain.Message.ContentPart{
                   type: :thinking,
                   content:
                     "Let's add these numbers.\n400 + 50 = 450\n450 + 3 = 453\n\nSo 400 + 50 + 3 = 453",
                   options: [
                     signature:
                       "ErUBCkYIARgCIkCspHHl1+BPuvAExtRMzy6e6DGYV4vI7D8dgqnzLm7RbQ5e4j+aAopCyq29fZqUNNdZbOLleuq/DYIyXjX4HIyIEgwE4N3Vb+9hzkFk/NwaDOy3fw0f0zqRZhAk4CIwp18hR9UsOWYC+pkvt1SnIOGCXBcLdwUxIoUeG3z6WfNwWJV7fulSvz7EVCN5ypzwKh2m/EY9LS1DK1EdUc770O8XdI/j4i0ibc8zRNIjvA=="
                   ]
                 },
                 %LangChain.Message.ContentPart{
                   type: :text,
                   content: "The answer is 453.\n\n400 + 50 = 450\n450 + 3 = 453",
                   options: []
                 }
               ],
               status: :complete,
               index: 1,
               role: :assistant,
               tool_calls: nil,
               metadata: %{
                 usage: %LangChain.TokenUsage{
                   input: 55,
                   output: 84,
                   raw: %{
                     "cache_creation_input_tokens" => 0,
                     "cache_read_input_tokens" => 0,
                     "input_tokens" => 55,
                     "output_tokens" => 84
                   }
                 }
               }
             }
    end

    test "handles content list with unknown types like reference" do
      # Mistral sometimes returns content as a list with reference and text types
      # This should not crash and should extract the text content
      primary = %MessageDelta{
        role: :assistant,
        merged_content: [],
        status: :incomplete
      }

      delta = %MessageDelta{
        role: :assistant,
        content: [
          %{"reference_ids" => [], "type" => "reference"},
          %{"text" => "{\"entries", "type" => "text"}
        ],
        index: 0,
        status: :incomplete
      }

      merged = MessageDelta.merge_delta(primary, delta)

      # Should have extracted the text content and skipped the reference
      assert merged.merged_content == [ContentPart.text!("{\"entries")]
    end

    test "handles content list with only unknown types" do
      # When content list has only unknown types, should not crash
      primary = %MessageDelta{
        role: :assistant,
        merged_content: [ContentPart.text!("existing")],
        status: :incomplete
      }

      delta = %MessageDelta{
        role: :assistant,
        content: [
          %{"reference_ids" => ["ref1"], "type" => "reference"}
        ],
        index: 0,
        status: :incomplete
      }

      merged = MessageDelta.merge_delta(primary, delta)

      # Should preserve existing content and skip unknown types
      assert merged.merged_content == [ContentPart.text!("existing")]
    end
  end

  describe "merge_deltas/2" do
    test "merges a batch of deltas starting from nil" do
      batch = [
        %MessageDelta{content: "Hello", role: :assistant},
        %MessageDelta{content: " world", role: :assistant},
        %MessageDelta{content: "!", role: :assistant}
      ]

      result = MessageDelta.merge_deltas(nil, batch)

      assert result == %MessageDelta{
               content: nil,
               merged_content: [ContentPart.text!("Hello world!")],
               role: :assistant,
               status: :incomplete
             }
    end

    test "accumulates deltas across multiple batches" do
      accumulated = nil

      batch_1 = [
        %MessageDelta{content: "Hello", role: :assistant},
        %MessageDelta{content: " world", role: :assistant}
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_1)

      assert accumulated.merged_content == [ContentPart.text!("Hello world")]

      batch_2 = [
        %MessageDelta{content: "!", role: :assistant},
        %MessageDelta{content: " How", role: :assistant}
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_2)

      assert accumulated.merged_content == [ContentPart.text!("Hello world! How")]

      batch_3 = [
        %MessageDelta{content: " are you?", role: :assistant, status: :complete}
      ]

      result = MessageDelta.merge_deltas(accumulated, batch_3)

      assert result == %MessageDelta{
               content: nil,
               merged_content: [ContentPart.text!("Hello world! How are you?")],
               role: :assistant,
               status: :complete
             }
    end

    test "handles nested lists (flattening)" do
      accumulated = nil

      # Simulate what OpenAI might return - nested lists
      batch = [
        [
          %MessageDelta{content: "Hello", role: :assistant},
          %MessageDelta{content: " world", role: :assistant}
        ],
        [
          %MessageDelta{content: "!", role: :assistant, status: :complete}
        ]
      ]

      result = MessageDelta.merge_deltas(accumulated, batch)

      assert result == %MessageDelta{
               content: nil,
               merged_content: [ContentPart.text!("Hello world!")],
               role: :assistant,
               status: :complete
             }
    end

    test "handles tool calls across batches" do
      accumulated = nil

      batch_1 = [
        %MessageDelta{content: "", role: :assistant},
        %MessageDelta{
          content: nil,
          role: :assistant,
          tool_calls: [
            %ToolCall{
              status: :incomplete,
              type: :function,
              call_id: "call_123",
              name: "get_weather",
              arguments: nil,
              index: 0
            }
          ]
        }
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_1)

      batch_2 = [
        %MessageDelta{
          content: nil,
          role: :assistant,
          tool_calls: [
            %ToolCall{
              status: :incomplete,
              type: :function,
              call_id: nil,
              name: nil,
              arguments: "{\"city\":",
              index: 0
            }
          ]
        },
        %MessageDelta{
          content: nil,
          role: :assistant,
          tool_calls: [
            %ToolCall{
              status: :incomplete,
              type: :function,
              call_id: nil,
              name: nil,
              arguments: " \"Portland\"}",
              index: 0
            }
          ]
        }
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_2)

      batch_3 = [
        %MessageDelta{content: "", role: :assistant, status: :complete}
      ]

      result = MessageDelta.merge_deltas(accumulated, batch_3)

      assert result == %MessageDelta{
               content: nil,
               merged_content: [],
               role: :assistant,
               status: :complete,
               tool_calls: [
                 %ToolCall{
                   status: :incomplete,
                   type: :function,
                   call_id: "call_123",
                   name: "get_weather",
                   arguments: "{\"city\": \"Portland\"}",
                   index: 0
                 }
               ]
             }
    end

    test "handles thinking and text content parts across batches" do
      accumulated = nil

      batch_1 = [
        %MessageDelta{content: [], role: :assistant},
        %MessageDelta{
          content: %ContentPart{type: :thinking, content: "Let me think", options: nil},
          role: :assistant,
          index: 0
        }
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_1)

      batch_2 = [
        %MessageDelta{
          content: %ContentPart{type: :thinking, content: " about this", options: nil},
          role: :assistant,
          index: 0
        },
        %MessageDelta{
          content: %ContentPart{type: :text, content: "Here's my answer", options: nil},
          role: :assistant,
          index: 1
        }
      ]

      result = MessageDelta.merge_deltas(accumulated, batch_2)

      assert result == %MessageDelta{
               content: nil,
               merged_content: [
                 %ContentPart{type: :thinking, content: "Let me think about this", options: nil},
                 %ContentPart{type: :text, content: "Here's my answer", options: nil}
               ],
               role: :assistant,
               status: :incomplete,
               index: 1
             }
    end

    test "accumulates token usage across batches" do
      accumulated = nil

      batch_1 = [
        %MessageDelta{
          content: "Hello",
          role: :assistant,
          metadata: %{
            usage: %LangChain.TokenUsage{input: 10, output: 5}
          }
        }
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_1)

      batch_2 = [
        %MessageDelta{
          content: " world",
          role: :assistant,
          metadata: %{
            usage: %LangChain.TokenUsage{input: 5, output: 10}
          }
        }
      ]

      result = MessageDelta.merge_deltas(accumulated, batch_2)

      assert result.metadata.usage.input == 15
      assert result.metadata.usage.output == 15
    end

    test "handles empty batch list" do
      accumulated = %MessageDelta{
        content: nil,
        merged_content: [ContentPart.text!("Hello")],
        role: :assistant
      }

      result = MessageDelta.merge_deltas(accumulated, [])

      assert result == accumulated
    end

    test "preserves status updates across batches" do
      accumulated = nil

      batch_1 = [
        %MessageDelta{content: "Hello", role: :assistant, status: :incomplete}
      ]

      accumulated = MessageDelta.merge_deltas(accumulated, batch_1)
      assert accumulated.status == :incomplete

      batch_2 = [
        %MessageDelta{content: " world", role: :assistant, status: :complete}
      ]

      result = MessageDelta.merge_deltas(accumulated, batch_2)
      assert result.status == :complete
    end

    test "handles length status" do
      accumulated = nil

      batch = [
        %MessageDelta{content: "Hello world", role: :assistant, status: :length}
      ]

      result = MessageDelta.merge_deltas(accumulated, batch)

      assert result.status == :length
    end
  end

  describe "to_message/1" do
    test "transform a merged and complete MessageDelta to a Message" do
      # :assistant content type
      delta = %LangChain.MessageDelta{
        content: nil,
        merged_content: [ContentPart.text!("Hello! How can I assist you?")],
        role: :assistant,
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert msg.content == [ContentPart.text!("Hello! How can I assist you?")]

      # :assistant type
      delta = %LangChain.MessageDelta{
        role: :assistant,
        tool_calls: [
          ToolCall.new!(%{
            call_id: "call_123",
            name: "calculator",
            arguments: "{\n  \"expression\": \"100 + 300 - 200\"\n}"
          })
        ],
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert [%ToolCall{} = call] = msg.tool_calls
      assert call.name == "calculator"
      # parses the arguments
      assert call.arguments == %{"expression" => "100 + 300 - 200"}
      assert msg.content == []
    end

    test "does not transform an incomplete MessageDelta to a Message" do
      delta = %LangChain.MessageDelta{
        content: "Hello! How can I assist ",
        merged_content: [ContentPart.text!("Hello! How can I assist ")],
        role: :assistant,
        status: :incomplete
      }

      assert {:error, "Cannot convert incomplete message"} = MessageDelta.to_message(delta)
    end

    test "transforms a delta stopped for length" do
      delta = %LangChain.MessageDelta{
        content: "Hello! How can I assist ",
        merged_content: [ContentPart.text!("Hello! How can I assist ")],
        role: :assistant,
        status: :length
      }

      assert {:ok, message} = MessageDelta.to_message(delta)
      assert message.role == :assistant
      assert message.content == [ContentPart.text!("Hello! How can I assist ")]
      assert message.status == :length
    end

    test "for a function_call, return an error when delta is invalid" do
      # a partially merged delta is invalid. It may have the "complete" flag but
      # if previous message deltas are missing and were not merged, the
      # to_message function will fail.
      delta = %LangChain.MessageDelta{
        role: :assistant,
        tool_calls: [
          ToolCall.new!(%{
            call_id: "call_123",
            name: "calculator",
            arguments: "{\n  \"expression\": \"100 + 300 - 200\""
          })
        ],
        status: :complete
      }

      {:error, reason} = MessageDelta.to_message(delta)
      assert reason == "tool_calls: arguments: invalid json"
    end

    test "allows normal content that starts with lowercase words" do
      # Normal content should not be flagged as malformed tool call
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: [
          ContentPart.text!("get started with the project by reading the documentation")
        ],
        tool_calls: [],
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
    end

    test "allows content with JSON that doesn't look like tool call" do
      # Content with JSON but not tool call pattern
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: [
          ContentPart.text!("Here is the data: {\"name\": \"test\"}")
        ],
        tool_calls: [],
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
    end

    test "allows valid tool calls without triggering malformed detection" do
      # When tool_calls is properly populated, don't check content
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: [],
        tool_calls: [
          ToolCall.new!(%{
            call_id: "call_123",
            name: "get_festival",
            arguments: "{\"id\": \"abc-123\"}"
          })
        ],
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
      assert length(msg.tool_calls) == 1
    end

    test "rejects empty assistant message with no content and no tool_calls" do
      # Mistral sometimes returns completely empty assistant messages
      # which violate conversation flow rules
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: [],
        tool_calls: [],
        status: :complete
      }

      {:error, reason} = MessageDelta.to_message(delta)
      assert reason =~ "Empty assistant message"
    end

    test "rejects empty assistant message with nil content and nil tool_calls" do
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: nil,
        tool_calls: nil,
        status: :complete
      }

      {:error, reason} = MessageDelta.to_message(delta)
      assert reason =~ "Empty assistant message"
    end

    test "handles merged_content with nil values from index padding" do
      # merged_content can have nil values when content parts arrive at
      # non-sequential indices (e.g., thinking at 0, text at 2, leaving 1 as nil)
      delta = %LangChain.MessageDelta{
        role: :assistant,
        merged_content: [
          ContentPart.text!("Hello"),
          nil,
          ContentPart.text!("World")
        ],
        tool_calls: [],
        status: :complete
      }

      {:ok, %Message{} = msg} = MessageDelta.to_message(delta)
      assert msg.role == :assistant
    end
  end

  describe "migrate_to_content_parts/1" do
    test "converts string content to a ContentPart of text" do
      delta = %MessageDelta{
        content: "Hello world",
        role: :assistant,
        status: :incomplete
      }

      upgraded = MessageDelta.migrate_to_content_parts(delta)

      assert upgraded == %MessageDelta{
               content: ContentPart.text!("Hello world"),
               merged_content: [],
               role: :assistant,
               status: :incomplete
             }
    end

    test "leaves existing ContentPart content unchanged" do
      delta = %MessageDelta{
        content: %ContentPart{
          type: :thinking,
          content: "Let's think about this",
          options: nil
        },
        role: :assistant,
        status: :incomplete
      }

      upgraded = MessageDelta.migrate_to_content_parts(delta)

      assert upgraded == delta
    end

    test "handles nil content" do
      delta = %MessageDelta{
        content: nil,
        role: :assistant,
        status: :incomplete
      }

      upgraded = MessageDelta.migrate_to_content_parts(delta)

      assert upgraded == delta
    end

    test "handles empty string content" do
      delta = %MessageDelta{
        content: "",
        role: :assistant,
        status: :incomplete
      }

      upgraded = MessageDelta.migrate_to_content_parts(delta)

      assert upgraded == %MessageDelta{
               content: nil,
               merged_content: [],
               role: :assistant,
               status: :incomplete
             }
    end
  end

end
