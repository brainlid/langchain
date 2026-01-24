# LangChain Elixir

This file provides guidance for Coding Agents working in this repository.

## Project Overview

Elixir LangChain is a toolchain framework for integrating Large Language Models (LLMs) into Elixir applications.
It provides a chain-based architecture to connect processes, services, tools, and functionality with various AI models including OpenAI, Anthropic Claude, Google Gemini, and others.
It draws inspiration from LangChain for Python and JavaScript, but follows it's own path that aims for integration with Elixir ecosystem.

## Common Development Commands

### Setup and Dependencies
```bash
# Install dependencies
mix deps.get

# Set up environment variables for API keys
cp .envrc_template .envrc
# Edit .envrc with API keys, then:
source .envrc
```

### Testing
```bash
# Run unit tests only (makes no external API calls)
mix test

# Run specific live API tests (billable and should be confirmed first)
mix test --include live_call
mix test --include live_open_ai
mix test --include live_anthropic
mix test --include live_grok

# Run a single test file
mix test test/path/to/test_file.exs

# Run a specific test by line number
mix test test/path/to/test_file.exs:42
```

#### Writing Tests

Prefer to make test assertions using pattern matching rather than checking list lengths and extracting items by index.

Avoid:
```elixir
  # Should have patched the dangling tool call
  assert length(processed_state.messages) == 3
  tool_message = Enum.at(processed_state.messages, 1)
  assert tool_message.role == :tool
```

Favor:
```elixir
  # Should have patched the dangling tool call
  assert [_msg1, %ToolMessage{role: :tool} = tool_msg, _] = processed_state.messages
```

Or favor this format when more than 1 assertion will be made:
```elixir
  # Should have patched the dangling tool call
  assert [_msg1, tool_msg, _] = processed_state.messages
  assert tool_msg.role == :tool
```

### Code Quality

Always run this before committing:

```bash
# Performs a set of pre-commit steps of compile checks, formatting, running test, etc.
mix precommit
```

## High-Level Architecture

### Core Components

1. **Chat Models** (`lib/chat_models/`)
   - `ChatModel` behavior defines the interface for all LLM implementations
   - Each provider (OpenAI, Anthropic, etc.) implements this behavior
   - Supports streaming, function calling, and multi-modal inputs

2. **Chains** (`lib/chains/`)
   - `LLMChain`: Primary abstraction for core orchestration of conversations and other LLM workflows
   - `DataExtractionChain`: Structured data extraction from text
   - `RoutingChain`: Dynamic routing based on input
   - Chains compose multiple operations and maintain conversation state

3. **Messages** (`lib/message.ex` and `lib/message/`)
   - `Message`: Core structure with roles (system, user, assistant, tool)
   - `ContentPart`: Handles multi-modal content (text, images, files)
   - `ToolCall` and `ToolResult`: Function invocation and results

4. **Functions** (`lib/function.ex`)
   - Integrates custom Elixir functions with LLMs
   - JSON Schema-based parameter validation
   - Context-aware execution with async support

### Key Patterns

- **Behavior-based design**: All chat models implement the `ChatModel` behavior
- **Ecto schemas**: Used for data validation and type casting throughout
- **Streaming support**: Built-in streaming capabilities for real-time responses
- **Error handling**: Consistent error tuples `{:ok, result}` / `{:error, reason}`

### Adding New Features

When adding a new LLM provider:
1. Create a new module in `lib/chat_models/`
2. Implement the `ChatModel` behavior
3. Add corresponding tests in `test/chat_models/`
4. Update documentation with supported features

When adding new chain types:
1. Create in `lib/chains/` following existing patterns
2. Use `Ecto.Schema` for configuration
3. Implement `run/2` function for execution
4. Add comprehensive tests including async scenarios

## Testing Guidelines

- Tests mirror the source structure (e.g., `lib/chains/llm_chain.ex` → `test/chains/llm_chain_test.exs`)
- Use `@tag :live_call` for tests requiring actual API calls
- Mock external dependencies using `Mimic` for unit tests
- Always test both sync and async execution paths when applicable

## Important Notes

- **API Keys**: Never commit API keys. Use environment variables via `.envrc`
- **Live Tests**: Be cautious with live tests as they incur API costs
- **Multi-modal**: When working with messages, use `ContentPart` structures
- **Callbacks**: Chains support extensive callback system for monitoring and extending behavior


## Project guidelines

- Use `mix precommit` alias when you are done with all changes and fix any pending issues
- Use the already included and available `:req` (`Req`) library for HTTP requests, **avoid** `:httpoison`, `:tesla`, and `:httpc`. Req is included by default and is the preferred HTTP client for Phoenix apps

### Phoenix v1.8 guidelines

- **Always** begin your LiveView templates with `<Layouts.app flash={@flash} ...>` which wraps all inner content
- The `MyAppWeb.Layouts` module is aliased in the `my_app_web.ex` file, so you can use it without needing to alias it again
- Anytime you run into errors with no `current_scope` assign:
  - You failed to follow the Authenticated Routes guidelines, or you failed to pass `current_scope` to `<Layouts.app>`
  - **Always** fix the `current_scope` error by moving your routes to the proper `live_session` and ensure you pass `current_scope` as needed
- Phoenix v1.8 moved the `<.flash_group>` component to the `Layouts` module. You are **forbidden** from calling `<.flash_group>` outside of the `layouts.ex` module
- Out of the box, `core_components.ex` imports an `<.icon name="hero-x-mark" class="w-5 h-5"/>` component for for hero icons. **Always** use the `<.icon>` component for icons, **never** use `Heroicons` modules or similar
- **Always** use the imported `<.input>` component for form inputs from `core_components.ex` when available. `<.input>` is imported and using it will will save steps and prevent errors
- If you override the default input classes (`<.input class="myclass px-2 py-1 rounded-lg">)`) class with your own values, no default classes are inherited, so your
custom classes must fully style the input

### JS and CSS guidelines

- **Use Tailwind CSS classes and custom CSS rules** to create polished, responsive, and visually stunning interfaces.
- Tailwindcss v4 **no longer needs a tailwind.config.js** and uses a new import syntax in `app.css`:

      @import "tailwindcss" source(none);
      @source "../css";
      @source "../js";
      @source "../../lib/my_app_web";

- **Always use and maintain this import syntax** in the app.css file for projects generated with `phx.new`
- **Never** use `@apply` when writing raw css
- **Always** manually write your own tailwind-based components instead of using daisyUI for a unique, world-class design
- Out of the box **only the app.js and app.css bundles are supported**
  - You cannot reference an external vendor'd script `src` or link `href` in the layouts
  - You must import the vendor deps into app.js and app.css to use them
  - **Never write inline <script>custom js</script> tags within templates**

### UI/UX & design guidelines

- **Produce world-class UI designs** with a focus on usability, aesthetics, and modern design principles
- Implement **subtle micro-interactions** (e.g., button hover effects, and smooth transitions)
- Ensure **clean typography, spacing, and layout balance** for a refined, premium look
- Focus on **delightful details** like hover effects, loading states, and smooth page transitions

<!-- usage-rules-start -->

<!-- phoenix:elixir-start -->
## Elixir guidelines

- Elixir lists **do not support index based access via the access syntax**

  **Never do this (invalid)**:

      i = 0
      mylist = ["blue", "green"]
      mylist[i]

  Instead, **always** use `Enum.at`, pattern matching, or `List` for index based list access, ie:

      i = 0
      mylist = ["blue", "green"]
      Enum.at(mylist, i)

- Elixir variables are immutable, but can be rebound, so for block expressions like `if`, `case`, `cond`, etc
  you *must* bind the result of the expression to a variable if you want to use it and you CANNOT rebind the result inside the expression, ie:

      # INVALID: we are rebinding inside the `if` and the result never gets assigned
      if connected?(socket) do
        socket = assign(socket, :val, val)
      end

      # VALID: we rebind the result of the `if` to a new variable
      socket =
        if connected?(socket) do
          assign(socket, :val, val)
        end

- **Never** nest multiple modules in the same file as it can cause cyclic dependencies and compilation errors
- **Never** use map access syntax (`changeset[:field]`) on structs as they do not implement the Access behaviour by default. For regular structs, you **must** access the fields directly, such as `my_struct.field` or use higher level APIs that are available on the struct if they exist, `Ecto.Changeset.get_field/2` for changesets
- Elixir's standard library has everything necessary for date and time manipulation. Familiarize yourself with the common `Time`, `Date`, `DateTime`, and `Calendar` interfaces by accessing their documentation as necessary. **Never** install additional dependencies unless asked or for date/time parsing (which you can use the `date_time_parser` package)
- Don't use `String.to_atom/1` on user input (memory leak risk)
- Predicate function names should not start with `is_` and should end in a question mark. Names like `is_thing` should be reserved for guards
- Elixir's builtin OTP primitives like `DynamicSupervisor` and `Registry`, require names in the child spec, such as `{DynamicSupervisor, name: MyApp.MyDynamicSup}`, then you can use `DynamicSupervisor.start_child(MyApp.MyDynamicSup, child_spec)`
- Use `Task.async_stream(collection, callback, options)` for concurrent enumeration with back-pressure. The majority of times you will want to pass `timeout: :infinity` as option

## Mix guidelines

- Read the docs and options before using tasks (by using `mix help task_name`)
- To debug test failures, run tests in a specific file with `mix test test/my_test.exs` or run all previously failed tests with `mix test --failed`
- `mix deps.clean --all` is **almost never needed**. **Avoid** using it unless you have good reason
<!-- phoenix:elixir-end -->

<!-- phoenix:ecto-start -->
## Ecto Guidelines

- **Always** preload Ecto associations in queries when they'll be accessed in templates, ie a message that needs to reference the `message.user.email`
- Remember `import Ecto.Query` and other supporting modules when you write `seeds.exs`
- `Ecto.Schema` fields always use the `:string` type, even for `:text`, columns, ie: `field :name, :string`
- `Ecto.Changeset.validate_number/2` **DOES NOT SUPPORT the `:allow_nil` option**. By default, Ecto validations only run if a change for the given field exists and the change value is not nil, so such as option is never needed
- You **must** use `Ecto.Changeset.get_field(changeset, :field)` to access changeset fields
- Fields which are set programatically, such as `user_id`, must not be listed in `cast` calls or similar for security purposes. Instead they must be explicitly set when creating the struct
<!-- phoenix:ecto-end -->

<!-- usage-rules-end -->