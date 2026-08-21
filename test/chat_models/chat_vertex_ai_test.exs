defmodule ChatModels.ChatVertexAITest do
  alias LangChain.ChatModels.ChatVertexAI
  use LangChain.BaseCase
  use Mimic

  doctest LangChain.ChatModels.ChatVertexAI
  alias LangChain.ChatModels.ChatVertexAI
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.Message.ToolCall
  alias LangChain.Message.ToolResult
  alias LangChain.MessageDelta
  alias LangChain.Function
  alias LangChain.FunctionParam
  alias LangChain.LangChainError
  alias LangChain.TokenUsage

  @test_model "gemini-3.7-flash"

  setup do
    {:ok, hello_world} =
      Function.new(%{
        name: "hello_world",
        description: "Give a hello world greeting.",
        function: fn _args, _context -> {:ok, "Hello world!"} end
      })

    model =
      ChatVertexAI.new!(%{
        "model" => @test_model,
        "endpoint" => "http://localhost:1234/"
      })

    %{model: model, hello_world: hello_world}
  end

  describe "new/1" do
    test "works with minimal attr" do
      assert {:ok, %ChatVertexAI{} = vertex_ai} =
               ChatVertexAI.new(%{"model" => @test_model, "endpoint" => "http://localhost:1234/"})

      assert vertex_ai.model == @test_model
    end

    test "returns error when invalid" do
      assert {:error, changeset} = ChatVertexAI.new(%{"model" => nil, "endpoint" => nil})
      refute changeset.valid?
      assert {"can't be blank", _} = changeset.errors[:model]
    end

    test "supports overriding the API endpoint" do
      override_url = "http://localhost:1234/"

      model =
        ChatVertexAI.new!(%{
          model: @test_model,
          endpoint: override_url
        })

      assert model.endpoint == override_url
    end

    test "supports setting json_response and json_schema" do
      json_schema = %{
        "type" => "object",
        "properties" => %{
          "name" => %{"type" => "string"},
          "age" => %{"type" => "integer"}
        }
      }

      {:ok, vertex_ai} =
        ChatVertexAI.new(%{
          "model" => @test_model,
          "endpoint" => "http://localhost:1234/",
          "json_response" => true,
          "json_schema" => json_schema
        })

      assert vertex_ai.json_response == true
      assert vertex_ai.json_schema == json_schema
    end
  end

  describe "build_url/1" do
    test "builds the request URL for the model and action" do
      model =
        ChatVertexAI.new!(%{model: @test_model, endpoint: "http://localhost:1234", stream: false})

      assert ChatVertexAI.build_url(model) ==
               "http://localhost:1234/models/#{@test_model}:generateContent"
    end

    test "uses the streaming action and a well-formed `?alt=sse` query when streaming" do
      model =
        ChatVertexAI.new!(%{model: @test_model, endpoint: "http://localhost:1234", stream: true})

      # The full URL — with the SSE parameter introduced by `?`, not `&`, since
      # there is no other query string (the API key is a bearer header).
      assert ChatVertexAI.build_url(model) ==
               "http://localhost:1234/models/#{@test_model}:streamGenerateContent?alt=sse"
    end

    test "does not put the API key in the URL (it is sent as a bearer token)" do
      model =
        ChatVertexAI.new!(%{
          model: @test_model,
          endpoint: "http://localhost:1234",
          api_key: "secret-key"
        })

      url = ChatVertexAI.build_url(model)

      refute url =~ "key="
      refute url =~ "secret-key"
    end
  end

  describe "for_api/3" do
    setup do
      params = %{
        "model" => @test_model,
        "endpoint" => "http://localhost:1234/",
        "temperature" => 1.0,
        "top_p" => 1.0,
        "top_k" => 1.0
      }

      {:ok, vertex_ai} = ChatVertexAI.new(params)

      %{vertex_ai: vertex_ai, params: params}
    end

    test "generates a map for an API call", %{vertex_ai: vertex_ai} do
      data = ChatVertexAI.for_api(vertex_ai, [], [])
      assert %{"contents" => [], "generationConfig" => config} = data
      assert %{"temperature" => 1.0, "topK" => 1.0, "topP" => 1.0} = config
    end

    test "adds safety settings to the request if present" do
      settings = [
        %{"category" => "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold" => "BLOCK_ONLY_HIGH"}
      ]

      vertex_ai =
        ChatVertexAI.new!(%{
          model: @test_model,
          endpoint: "http://localhost:1234/",
          safety_settings: settings
        })

      data = ChatVertexAI.for_api(vertex_ai, [], [])
      assert %{"safetySettings" => ^settings} = data
    end

    test "does not add safety settings to the request when the list is empty", %{
      vertex_ai: vertex_ai
    } do
      data = ChatVertexAI.for_api(vertex_ai, [], [])
      refute Map.has_key?(data, "safetySettings")
    end

    test "generate a map containing a text, inline image, and image url parts", %{
      vertex_ai: google_ai
    } do
      messages = [
        %LangChain.Message{
          content:
            "You are an expert at providing an image description for assistive technology and SEO benefits.",
          role: :system
        },
        %LangChain.Message{
          content: [
            %LangChain.Message.ContentPart{
              type: :text,
              content: "This is the text."
            },
            %LangChain.Message.ContentPart{
              type: :image,
              content: "/9j/4AAQSkz",
              options: [media: "image/jpeg"]
            },
            %LangChain.Message.ContentPart{
              type: :image_url,
              content: "http://localhost:1234/image.jpg",
              options: [media: "image/jpeg"]
            }
          ],
          role: :user
        }
      ]

      data = ChatVertexAI.for_api(google_ai, messages, [])
      assert %{"contents" => [msg1]} = data

      assert %{
               "parts" => [
                 %{
                   "text" => "This is the text."
                 },
                 %{
                   "inlineData" => %{
                     "mimeType" => "image/jpeg",
                     "data" => "/9j/4AAQSkz"
                   }
                 },
                 %{
                   "fileData" => %{
                     "fileUri" => "http://localhost:1234/image.jpg",
                     "mimeType" => "image/jpeg"
                   }
                 }
               ]
             } = msg1
    end

    test "support file_url", %{vertex_ai: google_ai} do
      message =
        Message.new_user!([
          ContentPart.text!("User prompt"),
          ContentPart.file_url!("example.com/test.pdf", media: "application/pdf")
        ])

      data = ChatVertexAI.for_api(google_ai, [message], [])

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{"text" => "User prompt"},
                     %{
                       "fileData" => %{
                         "fileUri" => "example.com/test.pdf",
                         "mimeType" => "application/pdf"
                       }
                     }
                   ],
                   "role" => :user
                 }
               ]
             } = data
    end

    test "generates a map containing user and assistant messages", %{vertex_ai: vertex_ai} do
      user_message = "Hello Assistant!"
      assistant_message = "Hello User!"

      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_user!(user_message),
            Message.new_assistant!(assistant_message)
          ],
          []
        )

      assert %{"contents" => [msg1, msg2]} = data
      assert %{"role" => :user, "parts" => [%{"text" => ^user_message}]} = msg1
      assert %{"role" => :model, "parts" => [%{"text" => ^assistant_message}]} = msg2
    end

    test "generated a map containing response_mime_type and response_schema", %{params: params} do
      vertex_ai =
        params
        |> Map.merge(%{"json_response" => true, "json_schema" => %{"type" => "object"}})
        |> ChatVertexAI.new!()

      data = ChatVertexAI.for_api(vertex_ai, [], [])

      assert %{
               "generationConfig" => %{
                 "response_mime_type" => "application/json",
                 "response_schema" => %{"type" => "object"}
               }
             } = data
    end

    test "generates a map containing function and function call messages", %{vertex_ai: vertex_ai} do
      message = "Can you do an action for me?"
      arguments = %{"args" => "data"}
      function_result = %{"result" => "data"}

      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_user!(message),
            Message.new_assistant!(%{
              tool_calls: [
                ToolCall.new!(%{
                  call_id: "call_123",
                  name: "userland_action",
                  arguments: Jason.encode!(arguments)
                })
              ]
            }),
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "userland_action",
                  content: Jason.encode!(function_result)
                })
              ]
            })
          ],
          []
        )

      assert %{"contents" => [msg1, msg2, msg3]} = data
      assert %{"role" => :user, "parts" => [%{"text" => ^message}]} = msg1
      assert %{"role" => :model, "parts" => [tool_call]} = msg2
      assert %{"role" => :function, "parts" => [tool_result]} = msg3

      assert %{
               "functionCall" => %{
                 "args" => ^arguments,
                 "name" => "userland_action"
               }
             } = tool_call

      assert %{
               "functionResponse" => %{
                 "name" => "userland_action",
                 "response" => ^function_result
               }
             } = tool_result
    end

    test "preserves media as nested functionResponse parts", %{vertex_ai: vertex_ai} do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: [
                    ContentPart.text!(Jason.encode!(%{"summary" => "See attached chart"})),
                    ContentPart.image!("base64-image-data", media: "image/png")
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "render_chart",
                         "response" => %{"summary" => "See attached chart"},
                         "parts" => [
                           %{
                             "inlineData" => %{
                               "mimeType" => "image/png",
                               "data" => "base64-image-data"
                             }
                           }
                         ]
                       }
                     }
                   ],
                   "role" => :function
                 }
               ]
             } = data
    end

    test "preserves inline file parts as inlineData in functionResponse", %{
      vertex_ai: vertex_ai
    } do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "read_document",
                  content: [
                    ContentPart.text!(Jason.encode!(%{"summary" => "See attached document"})),
                    ContentPart.file!("base64-pdf-data",
                      media: "application/pdf",
                      display_name: "report.pdf"
                    )
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "read_document",
                         "response" => %{"summary" => "See attached document"},
                         "parts" => [
                           %{
                             "inlineData" => %{
                               "mimeType" => "application/pdf",
                               "data" => "base64-pdf-data",
                               "displayName" => "report.pdf"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "preserves display names for inline image tool results", %{vertex_ai: vertex_ai} do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "list_assets",
                  content: [
                    ContentPart.text!(
                      Jason.encode!(%{
                        "assets" => [
                          %{
                            "label" => "preview",
                            "image" => %{"$ref" => "asset_preview.png"}
                          }
                        ]
                      })
                    ),
                    ContentPart.image!("base64-image-data",
                      media: "image/png",
                      display_name: "asset_preview.png"
                    )
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "list_assets",
                         "response" => %{
                           "assets" => [
                             %{
                               "label" => "preview",
                               "image" => %{"$ref" => "asset_preview.png"}
                             }
                           ]
                         },
                         "parts" => [
                           %{
                             "inlineData" => %{
                               "mimeType" => "image/png",
                               "data" => "base64-image-data",
                               "displayName" => "asset_preview.png"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "preserves image URLs as nested functionResponse parts", %{vertex_ai: vertex_ai} do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: [
                    ContentPart.text!(Jason.encode!(%{"summary" => "See attached chart"})),
                    ContentPart.image_url!("https://example.com/chart.png", media: "image/png")
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "parts" => [
                           %{
                             "fileData" => %{
                               "mimeType" => "image/png",
                               "fileUri" => "https://example.com/chart.png"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "preserves display names for image URLs as nested functionResponse parts", %{
      vertex_ai: vertex_ai
    } do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: [
                    ContentPart.text!(Jason.encode!(%{"summary" => "See attached chart"})),
                    ContentPart.image_url!("https://example.com/chart.png",
                      media: "image/png",
                      display_name: "frame_1712345678901.png"
                    )
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "parts" => [
                           %{
                             "fileData" => %{
                               "mimeType" => "image/png",
                               "fileUri" => "https://example.com/chart.png",
                               "displayName" => "frame_1712345678901.png"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "preserves display names for file URLs as nested functionResponse parts", %{
      vertex_ai: vertex_ai
    } do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: [
                    ContentPart.text!(Jason.encode!(%{"summary" => "See attached chart"})),
                    ContentPart.file_url!("https://example.com/chart.png",
                      media: "image/png",
                      display_name: "frame_1712345678901.png"
                    )
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "parts" => [
                           %{
                             "fileData" => %{
                               "mimeType" => "image/png",
                               "fileUri" => "https://example.com/chart.png",
                               "displayName" => "frame_1712345678901.png"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "uses an empty response object when tool result only contains media", %{
      vertex_ai: vertex_ai
    } do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: [
                    ContentPart.image!("base64-image-data", media: "image/png")
                  ]
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "render_chart",
                         "response" => %{},
                         "parts" => [
                           %{
                             "inlineData" => %{
                               "mimeType" => "image/png",
                               "data" => "base64-image-data"
                             }
                           }
                         ]
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "keeps plain text tool results unchanged", %{vertex_ai: vertex_ai} do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "render_chart",
                  content: "plain text result"
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "render_chart",
                         "response" => %{"result" => "plain text result"}
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "decodes JSON string tool results into function responses", %{vertex_ai: vertex_ai} do
      data =
        ChatVertexAI.for_api(
          vertex_ai,
          [
            Message.new_tool_result!(%{
              tool_results: [
                ToolResult.new!(%{
                  tool_call_id: "call_123",
                  name: "resize_plan",
                  content:
                    Jason.encode!(%{
                      "error" => "Unknown element alias: 05ba59c1-1234",
                      "status" => "validation_error"
                    })
                })
              ]
            })
          ],
          []
        )

      assert %{
               "contents" => [
                 %{
                   "parts" => [
                     %{
                       "functionResponse" => %{
                         "name" => "resize_plan",
                         "response" => %{
                           "error" => "Unknown element alias: 05ba59c1-1234",
                           "status" => "validation_error"
                         }
                       }
                     }
                   ]
                 }
               ]
             } = data
    end

    test "generates a map containing a system message", %{vertex_ai: vertex_ai} do
      message = "These are some instructions."

      data = ChatVertexAI.for_api(vertex_ai, [Message.new_system!(message)], [])

      assert %{"system_instruction" => msg1} = data
      assert %{"parts" => %{"text" => ^message}} = msg1
    end

    test "generates a map containing function declarations", %{
      vertex_ai: vertex_ai,
      hello_world: hello_world
    } do
      data = ChatVertexAI.for_api(vertex_ai, [], [hello_world])

      assert %{"contents" => []} = data
      assert %{"tools" => [tool_call]} = data

      # A function with no parameters omits the field entirely. Google rejects
      # an empty OBJECT with "parameters.properties: should be non-empty for
      # OBJECT type".
      assert %{
               "functionDeclarations" => [
                 %{
                   "name" => "hello_world",
                   "description" => "Give a hello world greeting."
                 } = declaration
               ]
             } = tool_call

      refute Map.has_key?(declaration, "parameters")
    end

    test "removes schema keywords Google does not support from declarations", %{
      vertex_ai: vertex_ai
    } do
      # Vertex AI reaches the same Gemini `Schema` type, so a declaration
      # carrying `additionalProperties` fails the whole request with
      # `Unknown name "additionalProperties" ... Cannot find field`.
      {:ok, function} =
        Function.new(%{
          name: "get_weather",
          description: "Get the weather.",
          parameters: [
            FunctionParam.new!(%{name: "city", type: :string, required: true})
          ],
          function: fn _args, _context -> {:ok, "75 degrees"} end
        })

      data = ChatVertexAI.for_api(vertex_ai, [], [function])

      assert %{"tools" => [%{"functionDeclarations" => [declaration]}]} = data

      assert %{
               "name" => "get_weather",
               "description" => "Get the weather.",
               "parameters" => %{
                 "type" => "object",
                 "required" => ["city"],
                 "properties" => %{"city" => %{"type" => "string"}}
               }
             } == declaration
    end

    test "removes unsupported schema keywords from the json response schema", %{
      vertex_ai: vertex_ai
    } do
      vertex_ai = %{
        vertex_ai
        | json_response: true,
          json_schema: %{
            "type" => "object",
            "additionalProperties" => false,
            "properties" => %{"answer" => %{"type" => "string"}}
          }
      }

      assert %{
               "generationConfig" => %{
                 "response_mime_type" => "application/json",
                 "response_schema" => %{
                   "type" => "object",
                   "properties" => %{"answer" => %{"type" => "string"}}
                 }
               }
             } = ChatVertexAI.for_api(vertex_ai, [], [])
    end
  end

  describe "do_process_response/2" do
    test "handles receiving a message", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => "Hello User!"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatVertexAI.do_process_response(model, response)
      assert struct.role == :assistant
      [%ContentPart{type: :text, content: "Hello User!"}] = struct.content
      assert struct.index == 0
      assert struct.status == :complete
    end

    test "error if receiving non-text content", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "bad_role", "parts" => [%{"text" => "Hello user"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [{:error, %LangChainError{} = error}] =
               ChatVertexAI.do_process_response(model, response)

      assert error.type == "changeset"
      assert error.message == "role: is invalid"
    end

    test "handles receiving function calls", %{model: model} do
      args = %{"args" => "data"}

      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"functionCall" => %{"args" => args, "name" => "hello_world"}}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = struct] = ChatVertexAI.do_process_response(model, response)
      assert struct.role == :assistant
      assert struct.index == 0
      [call] = struct.tool_calls
      assert call.name == "hello_world"
      assert call.arguments == args
    end

    test "uses the id returned with a function call", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [
                %{
                  "functionCall" => %{
                    "args" => %{"city" => "Denver"},
                    "id" => "call_4162774",
                    "name" => "get_weather"
                  }
                }
              ]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatVertexAI.do_process_response(model, response)
      assert [%ToolCall{call_id: "call_4162774", name: "get_weather"}] = msg.tool_calls
    end

    test "synthesizes a distinct id per call when the response omits one", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [
                %{"functionCall" => %{"args" => %{"city" => "Denver"}, "name" => "get_weather"}},
                %{"functionCall" => %{"args" => %{"city" => "Moab"}, "name" => "get_weather"}}
              ]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%Message{} = msg] = ChatVertexAI.do_process_response(model, response)

      assert [
               %ToolCall{arguments: %{"city" => "Denver"}} = denver,
               %ToolCall{arguments: %{"city" => "Moab"}} = moab
             ] = msg.tool_calls

      assert is_binary(denver.call_id)
      assert denver.call_id != moab.call_id
    end

    test "keeps parallel function calls separate when each arrives in its own chunk", %{
      model: model
    } do
      chunk = fn city, id ->
        %{
          "candidates" => [
            %{
              "content" => %{
                "role" => "model",
                "parts" => [
                  %{
                    "functionCall" => %{
                      "args" => %{"city" => city},
                      "id" => id,
                      "name" => "get_weather"
                    }
                  }
                ]
              },
              "index" => 0
            }
          ]
        }
      end

      deltas =
        [chunk.("Denver", "call_1375045"), chunk.("Moab", "call_1375049")]
        |> Enum.flat_map(&ChatVertexAI.do_process_response(model, &1, MessageDelta))

      merged = MessageDelta.merge_deltas(deltas)

      assert [
               %ToolCall{call_id: "call_1375045", arguments: %{"city" => "Denver"}},
               %ToolCall{call_id: "call_1375049", arguments: %{"city" => "Moab"}}
             ] = merged.tool_calls
    end

    test "handles receiving MessageDeltas as well", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{
              "role" => "model",
              "parts" => [%{"text" => "This is the first part of a mes"}]
            },
            "finishReason" => "STOP",
            "index" => 0
          }
        ]
      }

      assert [%MessageDelta{} = struct] =
               ChatVertexAI.do_process_response(model, response, MessageDelta)

      assert struct.role == :assistant
      assert struct.content == "This is the first part of a mes"
      assert struct.index == 0
      assert struct.status == :incomplete
    end

    test "handles API error messages", %{model: model} do
      response = %{
        "error" => %{
          "code" => 400,
          "message" => "Invalid request",
          "status" => "INVALID_ARGUMENT"
        }
      }

      assert {:error, error_received} = ChatVertexAI.do_process_response(model, response)
      assert %LangChainError{message: error_string} = error_received
      assert error_string == "Invalid request"
      assert error_received.original == response
    end

    test "handles Jason.DecodeError", %{model: model} do
      response = {:error, %Jason.DecodeError{}}

      assert {:error, %LangChainError{} = error} =
               ChatVertexAI.do_process_response(model, response)

      assert error.type == "invalid_json"
      assert "Received invalid JSON:" <> _ = error.message
    end

    test "handles unexpected response with error", %{model: model} do
      response = %{}

      assert {:error, %LangChainError{} = error} =
               ChatVertexAI.do_process_response(model, response)

      assert error.type == "unexpected_response"
      assert error.message == "Unexpected response"
    end

    test "handles receiving a message with token usage", %{model: model} do
      response = %{
        "candidates" => [
          %{
            "content" => %{"role" => "model", "parts" => [%{"text" => "Hello User!"}]},
            "finishReason" => "STOP",
            "index" => 0
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 10,
          "candidatesTokenCount" => 5,
          "totalTokenCount" => 15
        }
      }

      assert [%Message{} = struct] = ChatVertexAI.do_process_response(model, response)
      assert struct.role == :assistant
      [%ContentPart{type: :text, content: "Hello User!"}] = struct.content
      assert struct.index == 0
      assert struct.status == :complete

      # Verify that token usage is properly included in metadata
      assert %TokenUsage{} = struct.metadata.usage
      assert struct.metadata.usage.input == 10
      assert struct.metadata.usage.output == 5

      assert struct.metadata.usage.raw == %{
               "promptTokenCount" => 10,
               "candidatesTokenCount" => 5,
               "totalTokenCount" => 15
             }
    end
  end

  describe "filter_parts_for_types/2" do
    test "returns a single functionCall type" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert [%{"text" => _}] = ChatVertexAI.filter_parts_for_types(parts, ["text"])

      assert [%{"functionCall" => _}] =
               ChatVertexAI.filter_parts_for_types(parts, ["functionCall"])
    end

    test "returns a set of types" do
      parts = [
        %{"text" => "I think I'll call this function."},
        %{
          "functionCall" => %{
            "args" => %{"args" => "data"},
            "name" => "userland_action"
          }
        }
      ]

      assert parts == ChatVertexAI.filter_parts_for_types(parts, ["text", "functionCall"])
    end
  end

  describe "filter_text_parts/1" do
    test "returns only text parts that are not nil or empty" do
      parts = [
        %{"text" => "I have text"},
        %{"text" => nil},
        %{"text" => ""},
        %{"text" => "I have more text"}
      ]

      assert ChatVertexAI.filter_text_parts(parts) == [
               %{"text" => "I have text"},
               %{"text" => "I have more text"}
             ]
    end
  end

  describe "get_message_contents/1" do
    test "returns basic text as a ContentPart" do
      message = Message.new_user!("Howdy!")

      result = ChatVertexAI.get_message_contents(message)

      assert result == [%{"text" => "Howdy!"}]
    end

    test "supports a list of ContentParts" do
      message =
        Message.new_user!([
          ContentPart.new!(%{type: :text, content: "Hello!"}),
          ContentPart.new!(%{type: :text, content: "What's up?"})
        ])

      result = ChatVertexAI.get_message_contents(message)

      assert result == [
               %{"text" => "Hello!"},
               %{"text" => "What's up?"}
             ]
    end
  end

  describe "serialize_config/2" do
    test "does not include the API key or callbacks" do
      model = ChatVertexAI.new!(%{model: @test_model, endpoint: "http://localhost:1234/"})
      result = ChatVertexAI.serialize_config(model)
      assert result["version"] == 1
      refute Map.has_key?(result, "api_key")
      refute Map.has_key?(result, "callbacks")
    end

    test "creates expected map" do
      model =
        ChatVertexAI.new!(%{
          model: @test_model,
          endpoint: "http://localhost:1234/"
        })

      result = ChatVertexAI.serialize_config(model)

      assert result == %{
               "endpoint" => "http://localhost:1234/",
               "model" => @test_model,
               "module" => "Elixir.LangChain.ChatModels.ChatVertexAI",
               "receive_timeout" => 60000,
               "thinking_config" => nil,
               "stream" => false,
               "temperature" => 0.9,
               "top_k" => 1.0,
               "top_p" => 1.0,
               "version" => 1,
               "json_response" => false,
               "json_schema" => nil,
               "safety_settings" => []
             }
    end

    test "survives a serialize/restore round-trip with safety settings" do
      settings = [
        %{"category" => "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold" => "BLOCK_ONLY_HIGH"}
      ]

      model =
        ChatVertexAI.new!(%{
          model: @test_model,
          endpoint: "http://localhost:1234/",
          safety_settings: settings
        })

      restored =
        model
        |> ChatVertexAI.serialize_config()
        |> ChatVertexAI.restore_from_map()

      assert {:ok, %ChatVertexAI{safety_settings: ^settings}} = restored
    end
  end

  describe "inspect" do
    test "redacts the API key" do
      chain = ChatVertexAI.new!(%{"model" => @test_model, "endpoint" => "http://localhost:1000"})

      changeset = Ecto.Changeset.cast(chain, %{api_key: "1234567890"}, [:api_key])

      refute inspect(changeset) =~ "1234567890"
      assert inspect(changeset) =~ "**redacted**"
    end
  end

  describe "live tests and token usage information" do
    @tag live_call: true, live_vertex_ai: true
    test "basic non-streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatVertexAI{} =
        chat =
        ChatVertexAI.new!(%{
          model: "gemini-2.5-flash",
          temperature: 0,
          endpoint: System.fetch_env!("VERTEX_API_ENDPOINT"),
          stream: false
        })

      chat = %ChatVertexAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatVertexAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert [
               %Message{
                 content: [
                   %Message.ContentPart{
                     type: :text,
                     content: "Colorful Threads",
                     options: []
                   }
                 ],
                 status: :complete,
                 role: :assistant,
                 index: nil,
                 tool_calls: [],
                 metadata: %{
                   usage: %TokenUsage{
                     input: 7,
                     output: 2
                   }
                 }
               }
             ] = result

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 7, output: 2} = usage
    end

    @tag live_call: true, live_vertex_ai: true
    test "streamed response works and fires token usage callback" do
      handlers = %{
        on_llm_token_usage: fn usage ->
          send(self(), {:fired_token_usage, usage})
        end
      }

      %ChatVertexAI{} =
        chat =
        ChatVertexAI.new!(%{
          model: "gemini-2.5-flash",
          temperature: 0,
          endpoint: System.fetch_env!("VERTEX_API_ENDPOINT"),
          stream: true
        })

      chat = %ChatVertexAI{chat | callbacks: [handlers]}

      {:ok, result} =
        ChatVertexAI.call(chat, [
          Message.new_user!("Return the response 'Colorful Threads'.")
        ])

      assert [
               [
                 %MessageDelta{
                   content: "Colorful Threads",
                   status: :complete,
                   index: nil,
                   role: :assistant,
                   tool_calls: nil,
                   metadata: %{
                     usage: %TokenUsage{
                       input: 7,
                       output: 2
                     }
                   }
                 }
               ]
             ] = result

      assert_received {:fired_token_usage, usage}
      assert %TokenUsage{input: 7, output: 2} = usage
    end
  end

  describe "google_search native tool" do
    @tag live_call: true, live_google_ai: true
    test "should include grounding metadata in response" do
      alias LangChain.Chains.LLMChain
      alias LangChain.Message
      alias LangChain.NativeTool

      chat =
        ChatVertexAI.new!(%{
          model: "gemini-2.5-flash",
          temperature: 0,
          endpoint: System.fetch_env!("VERTEX_API_ENDPOINT"),
          stream: true
        })

      {:ok, updated_chain} =
        %{llm: chat, verbose: false, stream: false}
        |> LLMChain.new!()
        |> LLMChain.add_message(Message.new_user!("What is the current Google stock price?"))
        |> LLMChain.add_tools(NativeTool.new!(%{name: "google_search", configuration: %{}}))
        |> LLMChain.run()

      assert %Message{} = updated_chain.last_message
      assert updated_chain.last_message.role == :assistant
      assert Map.has_key?(updated_chain.last_message.metadata, "groundingChunks")
    end
  end

  describe "multimodal function response preview" do
    @tag live_call: true, live_vertex_ai: true
    test "supports a tool result that includes text and an image" do
      alias LangChain.Chains.LLMChain

      test_pid = self()

      image_data =
        File.read!("test/support/images/barn_owl.jpg")
        |> Base.encode64()

      describe_image =
        Function.new!(%{
          name: "describe_image",
          description: "Return a short structured summary and the barn owl image.",
          parameters_schema: %{type: "object", properties: %{}},
          function: fn _args, _context ->
            {:ok,
             [
               ContentPart.text!(
                 Jason.encode!(%{
                   "summary" => "The attached image shows a barn owl perched on a branch."
                 })
               ),
               ContentPart.image!(image_data, media: "image/jpeg")
             ]}
          end
        })

      callbacks = %{
        on_llm_new_message: fn _chain, message ->
          send(test_pid, {:callback_msg, message})
        end,
        on_tool_response_created: fn _chain, tool_message ->
          send(test_pid, {:callback_tool_msg, tool_message})
        end
      }

      chat =
        ChatVertexAI.new!(%{
          model: "gemini-3-flash-preview",
          temperature: 0,
          endpoint: System.fetch_env!("VERTEX_API_ENDPOINT"),
          stream: false
        })

      {:ok, updated_chain} =
        %{llm: chat, verbose: false, stream: false}
        |> LLMChain.new!()
        |> LLMChain.add_message(
          Message.new_user!(
            "Call the describe_image tool exactly once. After receiving its tool result, answer in one short sentence."
          )
        )
        |> LLMChain.add_tools(describe_image)
        |> LLMChain.add_callback(callbacks)
        |> LLMChain.run(mode: :while_needs_response)

      assert [
               %Message{role: :user},
               %Message{
                 role: :assistant,
                 tool_calls: [%ToolCall{name: "describe_image"}]
               },
               %Message{
                 role: :tool,
                 tool_results: [%ToolResult{name: "describe_image"} = tool_result]
               },
               %Message{role: :assistant} = response
             ] = updated_chain.messages

      assert [
               %ContentPart{
                 type: :text,
                 content:
                   "{\"summary\":\"The attached image shows a barn owl perched on a branch.\"}"
               },
               %ContentPart{type: :image, options: [media: "image/jpeg"]}
             ] = tool_result.content

      assert_received {:callback_msg,
                       %Message{
                         role: :assistant,
                         tool_calls: [%ToolCall{name: "describe_image"}]
                       }}

      assert_received {:callback_tool_msg,
                       %Message{
                         role: :tool,
                         tool_results: [%ToolResult{name: "describe_image"}]
                       }}

      assert %Message{role: :assistant} = updated_chain.last_message
      response_text = ContentPart.parts_to_string(response.content) |> String.downcase()
      assert response_text =~ "owl"
    end
  end

  describe "req_config" do
    test "merges req_config into the request (non-streaming)" do
      expect(Req, :post, fn req_struct ->
        # assert headers from req_config
        assert req_struct.headers == %{"x-vertex-ai-llm-request-type" => ["shared"]}

        {:error, RuntimeError.exception("Something went wrong")}
      end)

      model =
        ChatVertexAI.new!(%{
          endpoint: "http://localhost:1234/",
          stream: false,
          model: @test_model,
          req_config: %{headers: [{"X-Vertex-AI-LLM-Request-Type", "shared"}]}
        })

      assert {:error, _} = ChatVertexAI.call(model, "prompt", [])
      verify!()
    end

    test "merges req_config into the request (streaming)" do
      expect(Req, :post, fn req_struct, _opts ->
        # assert headers from req_config
        assert req_struct.headers == %{
                 "x-vertex-ai-llm-request-type" => ["shared"],
                 "accept-encoding" => ["utf-8"]
               }

        {:error, RuntimeError.exception("Something went wrong")}
      end)

      model =
        ChatVertexAI.new!(%{
          endpoint: "http://localhost:1234/",
          stream: true,
          model: @test_model,
          req_config: %{headers: [{"X-Vertex-AI-LLM-Request-Type", "shared"}]}
        })

      assert {:error, _} = ChatVertexAI.call(model, "prompt", [])
      verify!()
    end
  end

  describe "streamed token usage" do
    # Gemini repeats the running totals for the message on every streamed
    # chunk rather than reporting only what that chunk added, so merging the
    # deltas must keep the latest reading instead of summing them.
    defp usage_chunk(text, output_tokens, opts \\ []) do
      candidate =
        %{"content" => %{"role" => "model", "parts" => [%{"text" => text}]}, "index" => 0}
        |> then(fn c ->
          if opts[:last], do: Map.put(c, "finishReason", "STOP"), else: c
        end)

      %{
        "candidates" => [candidate],
        "usageMetadata" => %{
          "promptTokenCount" => 100,
          "candidatesTokenCount" => output_tokens,
          "totalTokenCount" => 100 + output_tokens
        }
      }
    end

    test "a per-chunk running total is not summed across chunks", %{model: model} do
      chunks = [
        usage_chunk("Hel", 1),
        usage_chunk("lo", 5),
        usage_chunk("!", 9, last: true)
      ]

      merged =
        chunks
        |> Enum.flat_map(&ChatVertexAI.do_process_response(model, &1, MessageDelta))
        |> MessageDelta.merge_deltas()

      usage = TokenUsage.get(merged)

      assert usage.input == 100
      assert usage.output == 9
      assert usage.raw["promptTokenCount"] == 100
      assert usage.raw["candidatesTokenCount"] == 9
    end

    test "no reported count exceeds the largest single chunk reporting it", %{model: model} do
      chunks = [
        usage_chunk("Hel", 1),
        usage_chunk("lo", 5),
        usage_chunk("!", 9, last: true)
      ]

      merged =
        chunks
        |> Enum.flat_map(&ChatVertexAI.do_process_response(model, &1, MessageDelta))
        |> MessageDelta.merge_deltas()

      usage = TokenUsage.get(merged)

      assert usage.input <= 100
      assert usage.output <= 9
      assert usage.raw["totalTokenCount"] <= 109
    end

    test "a whole response reports its usage once", %{model: model} do
      response = usage_chunk("Hello User!", 9, last: true)

      assert [%Message{} = message] = ChatVertexAI.do_process_response(model, response, Message)

      assert %TokenUsage{input: 100, output: 9} = TokenUsage.get(message)
    end
  end
end
