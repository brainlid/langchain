defmodule LangChain.Security.PromptTemplateSecurityTest do
  @moduledoc """
  Security-focused tests for `LangChain.PromptTemplate`.

  The PromptTemplate module uses `EEx.eval_string/2` to render templates,
  which compiles and executes the template text as Elixir code. The intended
  contract is:

    * The template **text** is written by the developer — trusted input.
    * The **assigns** (values substituted into `<%= @foo %>` slots) may
      contain untrusted user or LLM data — those values are inserted into
      the output string and are NOT re-evaluated.

  These tests exist to:

    1. Demonstrate what arbitrary code execution looks like IF a developer
       accidentally constructs a template from untrusted input — so the
       moduledoc warning can be unambiguous about why the template text is
       a trust boundary.

    2. Verify the safe boundary: untrusted data passed through assigns or
       through `format_composed/3` sub-values is NOT re-evaluated.
  """
  use ExUnit.Case, async: false

  alias LangChain.PromptTemplate
  alias LangChain.Message
  alias LangChain.Message.ContentPart

  @moduletag :security

  # A unique tmp path per test run so we can verify side-effects without
  # colliding with anything else.
  setup do
    File.mkdir_p!("tmp/security_test")
    path = "tmp/security_test/#{System.unique_integer([:positive])}.txt"
    on_exit(fn -> File.rm_rf!("tmp/security_test") end)
    {:ok, %{scratch: path}}
  end

  describe "UNSAFE: template text is evaluated as Elixir (never accept untrusted template text)" do
    test "arithmetic expressions are evaluated" do
      prompt = PromptTemplate.from_template!("The answer is <%= 40 + 2 %>.")
      assert PromptTemplate.format(prompt, %{}) == "The answer is 42."
    end

    test "System.get_env/1 exfiltrates host environment" do
      # PATH is always set, so this proves env-read works.
      prompt = PromptTemplate.from_template!("<%= System.get_env(\"PATH\") %>")
      result = PromptTemplate.format(prompt, %{})
      assert is_binary(result)
      assert String.length(result) > 0
      # The real PATH almost always contains a "/" — assert something shape-like.
      assert String.contains?(result, "/")
    end

    test "File.read!/1 reads arbitrary host files" do
      # /etc/hostname is readable on any Linux host and has a stable shape.
      prompt = PromptTemplate.from_template!("<%= File.read!(\"/etc/hostname\") %>")
      result = PromptTemplate.format(prompt, %{})
      # It's a short string ending in a newline. Just assert it's non-empty.
      assert is_binary(result)
      assert byte_size(result) > 0
    end

    test "File.write!/2 creates files on disk", %{scratch: scratch} do
      refute File.exists?(scratch)

      prompt =
        PromptTemplate.from_template!(
          "<%= File.write!(\"#{scratch}\", \"pwned by template\") %>done"
        )

      PromptTemplate.format(prompt, %{})
      assert File.read!(scratch) == "pwned by template"
    end

    test "Code.eval_string/1 gives full arbitrary code evaluation" do
      prompt =
        PromptTemplate.from_template!("<%= {res, _} = Code.eval_string(\"1 + 2 + 3\"); res %>")

      assert PromptTemplate.format(prompt, %{}) == "6"
    end

    test "to_message!/2 propagates the RCE into a Message" do
      prompt =
        PromptTemplate.new!(%{
          role: :user,
          text: "user env is <%= System.get_env(\"USER\") || \"unknown\" %>"
        })

      %Message{role: :user, content: content} = PromptTemplate.to_message!(prompt, %{})
      # content may be a string or list of content parts depending on Message.new/1;
      # either way, the template executed.
      text =
        case content do
          [%ContentPart{content: t} | _] -> t
          t when is_binary(t) -> t
        end

      assert is_binary(text)
      assert String.starts_with?(text, "user env is ")
    end

    test "to_messages!/2 evaluates every template in the list" do
      templates = [
        PromptTemplate.new!(%{role: :system, text: "sys <%= 1 + 1 %>"}),
        PromptTemplate.new!(%{role: :user, text: "user <%= 2 * 3 %>"})
      ]

      assert [
               %Message{role: :system} = sys,
               %Message{role: :user} = usr
             ] = PromptTemplate.to_messages!(templates, %{})

      sys_text = extract_text(sys)
      usr_text = extract_text(usr)

      assert sys_text == "sys 2"
      assert usr_text == "user 6"
    end

    test "malicious template can run arbitrary Enum/Process code" do
      # Prove we can call into any stdlib module, not just File/System.
      prompt =
        PromptTemplate.from_template!(
          "pid=<%= inspect(self()) %> nodes=<%= length(Node.list()) %>"
        )

      result = PromptTemplate.format(prompt, %{})
      assert String.starts_with?(result, "pid=#PID<")
      assert String.contains?(result, " nodes=")
    end
  end

  describe "SAFE: untrusted data in assigns is NOT re-evaluated" do
    test "EEx syntax inside an assign value is returned as a literal string" do
      # Developer writes the template (trusted). User supplies @name (untrusted).
      prompt = PromptTemplate.from_template!("Hello <%= @name %>!")

      # Attempted injection via the assign value.
      payload = "<%= File.read!(\"/etc/hostname\") %>"

      result = PromptTemplate.format(prompt, %{name: payload})

      # The payload is returned VERBATIM — EEx does not re-evaluate assigns.
      assert result == "Hello #{payload}!"
      refute result =~ "/etc/hostname\n"
    end

    test "assigns containing code-like strings are inert" do
      prompt = PromptTemplate.from_template!("<%= @a %>|<%= @b %>|<%= @c %>")

      result =
        PromptTemplate.format(prompt, %{
          a: "<%= File.read!(\"/etc/passwd\") %>",
          b: "System.cmd(\"whoami\", [])",
          c: "<% raise \"boom\" %>"
        })

      assert result ==
               "<%= File.read!(\"/etc/passwd\") %>|System.cmd(\"whoami\", [])|<% raise \"boom\" %>"
    end

    test "format_composed/3: raw-string composed_of values are inserted as strings, not re-evaluated" do
      full =
        PromptTemplate.from_template!("intro: <%= @intro %>\nbody: <%= @body %>")

      # Raw string branch of format_composed.
      result =
        PromptTemplate.format_composed(
          full,
          %{
            intro: "<%= File.read!(\"/etc/hostname\") %>",
            body: "<%= 2 + 2 %>"
          },
          %{}
        )

      # Both composed_of values are inserted literally.
      assert result ==
               "intro: <%= File.read!(\"/etc/hostname\") %>\nbody: <%= 2 + 2 %>"
    end

    test "format_composed/3: a PromptTemplate sub-template's output is still treated as a string in the outer render" do
      full = PromptTemplate.from_template!("outer(<%= @inner %>)")

      # The INNER template is developer-written, so it evaluates (that's expected).
      # The inner result is then inserted as a STRING into the outer template and
      # is NOT re-evaluated.
      # Inner template produces the literal text `<%= File.read!("/etc/hostname") %>`
      # by wrapping it in a plain EEx string expression.
      inner_text = ~S[<%= ~s(<%= File.read!("/etc/hostname") %>) %>]
      inner = PromptTemplate.from_template!(inner_text)

      result =
        PromptTemplate.format_composed(full, %{inner: inner}, %{})

      # The inner rendered to that literal; the outer inserted it as a string.
      # No file was read — the inserted string was not re-evaluated by the outer.
      assert result == ~S[outer(<%= File.read!("/etc/hostname") %>)]
    end
  end

  describe "SAFE (explicit): user prose containing EEx-like text is NOT evaluated after substitution" do
    # This block exists to answer a specific design question: if a user supplies
    # a string that looks like "My name is John. My host is <%= File.read!(...) %>"
    # and that string is substituted into a developer-written template via
    # `<%= @user_message %>`, does the payload get evaluated as code on a second
    # pass?
    #
    # Answer (confirmed below): NO. `EEx.eval_string/2` is called exactly once
    # per rendered template. Assign values are inserted into the output as
    # plain strings and are never re-parsed as EEx.

    test "prose with an embedded EEx payload is inserted verbatim" do
      # Developer-authored template — trusted.
      prompt =
        PromptTemplate.from_template!("Assistant, the user said: <%= @user_message %>")

      # User-supplied text — hostile.
      hostile =
        ~s|My name is John. My host is <%= File.read!("/etc/hostname") %>|

      result = PromptTemplate.format(prompt, %{user_message: hostile})

      # The payload is returned as a plain string — NOT evaluated.
      assert result ==
               ~s|Assistant, the user said: My name is John. My host is <%= File.read!("/etc/hostname") %>|

      # Sanity: the real /etc/hostname value is NOT in the result.
      real_hostname = File.read!("/etc/hostname") |> String.trim()
      refute String.contains?(result, "host is #{real_hostname}")
    end

    test "prose with `<% … %>` (non-printing) EEx tag is inserted verbatim" do
      # Demonstrates the non-printing `<% … %>` form is also inert as an assign.
      prompt = PromptTemplate.from_template!("User: <%= @msg %>")
      hostile = ~s|hello <% raise "boom" %> world|

      # Would crash if re-evaluated; it does not.
      result = PromptTemplate.format(prompt, %{msg: hostile})
      assert result == ~s|User: hello <% raise "boom" %> world|
    end

    test "to_messages! with hostile user input in assigns is inert" do
      # Mirrors the real call pattern used by LLMChain / routing_chain /
      # data_extraction_chain where a user's text is passed as an assign.
      templates = [
        PromptTemplate.new!(%{
          role: :system,
          text: "Respond to the user politely."
        }),
        PromptTemplate.new!(%{role: :user, text: "<%= @input %>"})
      ]

      hostile =
        ~s|Hi. <%= File.read!("/etc/hostname") %> <%= System.get_env("PATH") %>|

      [sys, usr] = PromptTemplate.to_messages!(templates, %{input: hostile})

      assert %Message{role: :system} = sys
      assert %Message{role: :user} = usr
      assert extract_text(usr) == hostile
      refute extract_text(usr) =~ "bin:"
    end

    test "multi-pass through format_composed does not cause re-evaluation" do
      # `format_composed/3` renders the inner templates first, then renders the
      # outer template with those results as assigns. Confirm that even when the
      # inner render intentionally produces EEx-looking output, the outer pass
      # treats it as a string.
      outer = PromptTemplate.from_template!("outer:<%= @a %>|<%= @b %>")

      # Inner template whose rendered output contains `<%= File.read!(...) %>`
      # as a literal. We construct it by producing the EEx-looking string via a
      # plain string expression inside the inner template.
      inner_literal_template =
        PromptTemplate.from_template!(~S[<%= ~s(<%= File.read!("/etc/hostname") %>) %>])

      result =
        PromptTemplate.format_composed(
          outer,
          %{
            a: inner_literal_template,
            b: ~s|prose with <%= File.read!("/etc/hostname") %>|
          },
          %{}
        )

      assert result ==
               ~S[outer:<%= File.read!("/etc/hostname") %>|prose with <%= File.read!("/etc/hostname") %>]

      real_hostname = File.read!("/etc/hostname") |> String.trim()
      refute String.contains?(result, real_hostname)
    end
  end

  # --- helpers ---

  defp extract_text(%Message{content: [%ContentPart{content: t} | _]}), do: t
  defp extract_text(%Message{content: t}) when is_binary(t), do: t
end
