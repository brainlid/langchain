defmodule LangChain.UtilsTest do
  use ExUnit.Case

  doctest LangChain.Utils
  alias LangChain.Message
  alias LangChain.Message.ContentPart
  alias LangChain.MessageDelta
  alias LangChain.ChatModels.ChatOpenAI
  alias LangChain.Utils
  alias LangChain.Chains.LLMChain

  defmodule FakeSchema do
    use Ecto.Schema
    import Ecto.Changeset

    embedded_schema do
      field :name, :string
      field :age, :integer
      field :required, :string
      field :multiple, :string
    end

    def changeset(struct, attrs) do
      struct
      |> cast(attrs, [:name, :age, :required, :multiple])
      |> validate_required([:name, :required])
      |> validate_number(:age, greater_than_or_equal_to: 0)
      |> validate_inclusion(:multiple, ["north", "east", "south", "west"])
      |> validate_length(:multiple, max: 10)
    end
  end

  describe "changeset_error_to_string/1" do
    test "returns multiple errors as comma separated" do
      changeset = FakeSchema.changeset(%FakeSchema{}, %{})
      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)
      assert result == "name: can't be blank; required: can't be blank"
    end

    test "returns multiple errors on a field" do
      changeset =
        FakeSchema.changeset(%FakeSchema{}, %{
          name: "Tom",
          required: "value",
          multiple: "too long to be valid"
        })

      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)
      assert result == "multiple: should be at most 10 character(s), is invalid"
    end

    test "combines errors for multiple fields and multiple on single field" do
      changeset =
        FakeSchema.changeset(%FakeSchema{}, %{
          name: nil,
          required: nil,
          multiple: "too long to be valid"
        })

      refute changeset.valid?
      result = Utils.changeset_error_to_string(changeset)

      assert result ==
               "multiple: should be at most 10 character(s), is invalid; name: can't be blank; required: can't be blank"
    end

    test "handles ecto enum type errors" do
      {:error, changeset} = MessageDelta.new(%{role: "invalid"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid"
    end

    test "handles multiple errors on a field" do
      {:error, changeset} = MessageDelta.new(%{role: "invalid"})
      changeset = Ecto.Changeset.add_error(changeset, :role, "is required")
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is required, is invalid"
    end

    test "handles errors on multiple fields" do
      {:error, changeset} = MessageDelta.new(%{role: "invalid", index: "abc"})
      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is invalid; index: is invalid"
    end

    test "handles multiple errors on multiple fields" do
      {:error, changeset} = MessageDelta.new(%{role: "invalid", index: "abc"})

      changeset =
        changeset
        |> Ecto.Changeset.add_error(:index, "is numeric")
        |> Ecto.Changeset.add_error(:role, "is important")

      result = Utils.changeset_error_to_string(changeset)
      assert result == "role: is important, is invalid; index: is numeric, is invalid"
    end
  end

  describe "put_in_list/3" do
    test "adds to empty list" do
      assert [1] == Utils.put_in_list([], 0, 1)
    end

    test "adds to list with values" do
      result =
        [1, 2]
        |> Utils.put_in_list(2, "3")

      assert result == [1, 2, "3"]
    end

    test "replaces existing value at the index" do
      result =
        [1, 2, 3]
        |> Utils.put_in_list(0, "a")
        |> Utils.put_in_list(1, "b")
        |> Utils.put_in_list(2, "c")

      assert result == ["a", "b", "c"]
    end
  end

  describe "to_serializable_map/3" do
    test "converts a chat model to a string keyed map with a version included" do
      model =
        ChatOpenAI.new!(%{
          model: "gpt-4o",
          temperature: 0,
          frequency_penalty: 0.5,
          seed: 123,
          max_tokens: 1234,
          stream_options: %{include_usage: true}
        })

      result =
        Utils.to_serializable_map(model, [
          :model,
          :temperature,
          :frequency_penalty,
          :seed,
          :max_tokens,
          :stream_options
        ])

      assert result == %{
               "model" => "gpt-4o",
               "temperature" => 0.0,
               "frequency_penalty" => 0.5,
               "seed" => 123,
               "max_tokens" => 1234,
               "stream_options" => %{"include_usage" => true},
               "version" => 1,
               "module" => "Elixir.LangChain.ChatModels.ChatOpenAI"
             }
    end
  end

  describe "module_from_name/1" do
    test "returns :ok tuple with module when valid" do
      assert {:ok, DateTime} = Utils.module_from_name("Elixir.DateTime")
    end

    test "returns error when not a module" do
      assert {:error, reason} = Utils.module_from_name("not a module")
      assert reason == "Not an Elixir module: \"not a module\""
    end

    test "returns error when not an existing atom" do
      assert {:error, reason} = Utils.module_from_name("Elixir.Missing.Module")
      assert reason == "ChatModel module \"Elixir.Missing.Module\" not found"
    end
  end

  describe "split_system_message/2" do
    test "returns system message and rest separately" do
      system = Message.new_system!()
      user_msg = Message.new_user!("Hi")
      assert {system, [user_msg]} == Utils.split_system_message([system, user_msg])
    end

    test "return nil when no system message set" do
      user_msg = Message.new_user!("Hi")
      assert {nil, [user_msg]} == Utils.split_system_message([user_msg])
    end

    test "raises exception with multiple system messages" do
      error_message = "Anthropic only supports a single System message"

      assert_raise LangChain.LangChainError,
                   error_message,
                   fn ->
                     system = Message.new_system!()
                     user_msg = Message.new_user!("Hi")
                     Utils.split_system_message([system, user_msg, system], error_message)
                   end
    end

    test "has a default error message when no error message provided" do
      assert_raise LangChain.LangChainError,
                   "Only one system message is allowed",
                   fn ->
                     system = Message.new_system!()
                     user_msg = Message.new_user!("Hi")
                     Utils.split_system_message([system, user_msg, system])
                   end
    end
  end

  describe "replace_system_message!/2" do
    test "returns list with new system message" do
      non_system = [
        Message.new_user!("User 1"),
        Message.new_assistant!("Assistant 1")
      ]

      [new_system | rest] =
        Utils.replace_system_message!(
          [Message.new_system!("System A") | non_system],
          Message.new_system!("System B")
        )

      assert rest == non_system
      assert new_system.role == :system
      assert new_system.content == [ContentPart.text!("System B")]
    end

    test "handles when no existing system message" do
      non_system = [
        Message.new_user!("User 1"),
        Message.new_assistant!("Assistant 1")
      ]

      [new_system | rest] =
        Utils.replace_system_message!(non_system, Message.new_system!("System B"))

      assert rest == non_system
      assert new_system.role == :system
      assert new_system.content == [ContentPart.text!("System B")]
    end
  end

  describe "rewrap_callbacks_for_model/2" do
    test "wraps all LLM callback functions (not chain callbacks)" do
      # split across two callback maps
      callback_1 =
        %{
          on_llm_new_delta: fn %LLMChain{custom_context: context}, arg ->
            "Custom: #{inspect(context)} + #{arg} in on_llm_new_delta"
          end,
          on_llm_new_message: fn %LLMChain{custom_context: context}, arg ->
            "Custom: #{inspect(context)} + #{arg} in on_llm_new_message-1"
          end
        }

      callback_2 =
        %{
          on_llm_new_message: fn %LLMChain{custom_context: context}, arg ->
            # a repeated callback
            "Custom: #{inspect(context)} + #{arg} in on_llm_new_message-2"
          end,
          on_llm_ratelimit_info: fn %LLMChain{custom_context: context}, arg ->
            "Custom: #{inspect(context)} + #{arg} in on_llm_ratelimit_info"
          end,
          on_llm_token_usage: fn %LLMChain{custom_context: context}, arg ->
            "Custom: #{inspect(context)} + #{arg} in on_llm_token_usage"
          end,
          on_message_processed: fn _chain, _arg ->
            :ok
          end
        }

      llm = ChatOpenAI.new!(%{})

      chain =
        %{llm: llm}
        |> LLMChain.new!()
        |> LLMChain.update_custom_context(%{value: 1})
        |> LLMChain.add_callback(callback_1)
        |> LLMChain.add_callback(callback_2)

      updated_llm = Utils.rewrap_callbacks_for_model(llm, chain.callbacks, chain)

      [group_1, group_2] = updated_llm.callbacks

      assert "Custom: %{value: 1} + delta in on_llm_new_delta" ==
               group_1.on_llm_new_delta.("delta")

      assert "Custom: %{value: 1} + msg in on_llm_new_message-1" ==
               group_1.on_llm_new_message.("msg")

      assert "Custom: %{value: 1} + msg in on_llm_new_message-2" ==
               group_2.on_llm_new_message.("msg")

      assert "Custom: %{value: 1} + info in on_llm_ratelimit_info" ==
               group_2.on_llm_ratelimit_info.("info")

      assert "Custom: %{value: 1} + usage in on_llm_token_usage" ==
               group_2.on_llm_token_usage.("usage")

      # not an LLM event. Not included
      assert group_2[:on_message_processed] == nil
    end
  end

  describe "migrate_to_content_parts/1" do
    defmodule FakeContentSchema do
      use Ecto.Schema
      import Ecto.Changeset

      embedded_schema do
        field :content, :any, virtual: true
        field :other_field, :string
      end

      def changeset(struct, attrs) do
        struct
        |> cast(attrs, [:content, :other_field])
      end
    end

    test "converts binary content to list of ContentParts" do
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: "Hello world"})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?

      assert [%ContentPart{type: :text, content: "Hello world"}] =
               Ecto.Changeset.get_change(result, :content)
    end

    test "wraps single ContentPart in a list" do
      content_part = ContentPart.text!("Hello world")
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: content_part})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert [^content_part] = Ecto.Changeset.get_change(result, :content)
    end

    test "leaves list of ContentParts unchanged" do
      content_parts = [
        ContentPart.text!("Hello"),
        ContentPart.text!("world")
      ]

      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: content_parts})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert ^content_parts = Ecto.Changeset.get_change(result, :content)
    end

    test "leaves empty list unchanged" do
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: []})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert [] = Ecto.Changeset.get_change(result, :content)
    end

    test "leaves changeset unchanged when no content change" do
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{other_field: "value"})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert Ecto.Changeset.get_change(result, :content) == nil
    end

    test "leaves changeset unchanged when content is nil" do
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: nil})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert Ecto.Changeset.get_change(result, :content) == nil
    end

    test "handles whitespace content" do
      changeset = FakeContentSchema.changeset(%FakeContentSchema{}, %{content: "   "})

      result = Utils.migrate_to_content_parts(changeset)

      assert result.valid?
      assert nil == Ecto.Changeset.get_change(result, :content)
    end
  end
end
