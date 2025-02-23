defmodule LangChain.TextSplitter.LanguageSeparators do
  @python [
    # First, try to split along class definitions
    "\nclass ",
    "\ndef ",
    "\n\tdef ",
    # Now split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def python() do
    @python
  end

  @go [
    # Split along function definitions
    "\nfunc ",
    "\nvar ",
    "\nconst ",
    "\ntype ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nswitch ",
    "\ncase ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def go() do
    @go
  end

  @java [
    # Split along class definitions
    "\nclass ",
    # Split along method definitions
    "\npublic ",
    "\nprotected ",
    "\nprivate ",
    "\nstatic ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nswitch ",
    "\ncase ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def java() do
    @java
  end

  @kotlin [
    # Split along class definitions
    "\nclass ",
    # Split along method definitions
    "\npublic ",
    "\nprotected ",
    "\nprivate ",
    "\ninternal ",
    "\ncompanion ",
    "\nfun ",
    "\nval ",
    "\nvar ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nwhen ",
    "\ncase ",
    "\nelse ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def kotlin() do
    @kotlin
  end

  @js [
    # Split along function definitions
    "\nfunction ",
    "\nconst ",
    "\nlet ",
    "\nvar ",
    "\nclass ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nswitch ",
    "\ncase ",
    "\ndefault ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def js() do
    @js
  end

  @ts [
    "\nenum ",
    "\ninterface ",
    "\nnamespace ",
    "\ntype ",
    # Split along class definitions
    "\nclass ",
    # Split along function definitions
    "\nfunction ",
    "\nconst ",
    "\nlet ",
    "\nvar ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nswitch ",
    "\ncase ",
    "\ndefault ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def ts() do
    @ts
  end

  @php [
    # Split along function definitions
    "\nfunction ",
    # Split along class definitions
    "\nclass ",
    # Split along control flow statements
    "\nif ",
    "\nforeach ",
    "\nwhile ",
    "\ndo ",
    "\nswitch ",
    "\ncase ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def php() do
    @php
  end

  @proto [
    # Split along message definitions
    "\nmessage ",
    # Split along service definitions
    "\nservice ",
    # Split along enum definitions
    "\nenum ",
    # Split along option definitions
    "\noption ",
    # Split along import statements
    "\nimport ",
    # Split along syntax declarations
    "\nsyntax ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def proto() do
    @proto
  end

  @rst [
    # Split along section titles
    "\n=+\n",
    "\n-+\n",
    "\n\\*+\n",
    # Split along directive markers
    "\n\n.. *\n\n",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def rst() do
    @rst
  end

  @ruby [
    # Split along method definitions
    "\ndef ",
    "\nclass ",
    # Split along control flow statements
    "\nif ",
    "\nunless ",
    "\nwhile ",
    "\nfor ",
    "\ndo ",
    "\nbegin ",
    "\nrescue ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def ruby() do
    @ruby
  end

  @elixir [
    # Split along method function and module definition
    "\ndef ",
    "\ndefp ",
    "\ndefmodule ",
    "\ndefprotocol ",
    "\ndefmacro ",
    "\ndefmacrop ",
    # Split along control flow statements
    "\nif ",
    "\nunless ",
    "\nwhile ",
    "\ncase ",
    "\ncond ",
    "\nwith ",
    "\nfor ",
    "\ndo ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def elixir() do
    @elixir
  end

  @rust [
    # Split along function definitions
    "\nfn ",
    "\nconst ",
    "\nlet ",
    # Split along control flow statements
    "\nif ",
    "\nwhile ",
    "\nfor ",
    "\nloop ",
    "\nmatch ",
    "\nconst ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def rust() do
    @rust
  end

  @scala [
    # Split along class definitions
    "\nclass ",
    "\nobject ",
    # Split along method definitions
    "\ndef ",
    "\nval ",
    "\nvar ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nmatch ",
    "\ncase ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def scala() do
    @scala
  end

  @swift [
    # Split along function definitions
    "\nfunc ",
    # Split along class definitions
    "\nclass ",
    "\nstruct ",
    "\nenum ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\ndo ",
    "\nswitch ",
    "\ncase ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def swift() do
    @swift
  end

  @markdown [
    # First, try to split along Markdown headings (starting with level 2)
    "\n#\{1,6\} ",
    # Note the alternative syntax for headings (below) is not handled here
    # Heading level 2
    # ---------------
    # End of code block
    "```\n",
    # Horizontal lines
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    # Note that this splitter doesn't handle horizontal lines defined
    # by *three or more* of ***, ---, or ___, but this is not handled
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def markdown() do
    @markdown
  end

  @latex [
    # First, try to split along Latex sections
    "\n\\\\chapter{",
    "\n\\\\section{",
    "\n\\\\subsection{",
    "\n\\\\subsubsection{",
    # Now split by environments
    "\n\\\\begin{enumerate}",
    "\n\\\\begin{itemize}",
    "\n\\\\begin{description}",
    "\n\\\\begin{list}",
    "\n\\\\begin{quote}",
    "\n\\\\begin{quotation}",
    "\n\\\\begin{verse}",
    "\n\\\\begin{verbatim}",
    # Now split by math environments
    "\n\\\\begin{align}",
    "$$",
    "$",
    # Now split by the normal type of lines
    " ",
    ""
  ]
  def latex() do
    @latex
  end

  @html [
    # First, try to split along HTML tags
    "<body",
    "<div",
    "<p",
    "<br",
    "<li",
    "<h1",
    "<h2",
    "<h3",
    "<h4",
    "<h5",
    "<h6",
    "<span",
    "<table",
    "<tr",
    "<td",
    "<th",
    "<ul",
    "<ol",
    "<header",
    "<footer",
    "<nav",
    # Head
    "<head",
    "<style",
    "<script",
    "<meta",
    "<title",
    ""
  ]
  def html() do
    @html
  end

  @csharp [
    "\ninterface ",
    "\nenum ",
    "\nimplements ",
    "\ndelegate ",
    "\nevent ",
    # Split along class definitions
    "\nclass ",
    "\nabstract ",
    # Split along method definitions
    "\npublic ",
    "\nprotected ",
    "\nprivate ",
    "\nstatic ",
    "\nreturn ",
    # Split along control flow statements
    "\nif ",
    "\ncontinue ",
    "\nfor ",
    "\nforeach ",
    "\nwhile ",
    "\nswitch ",
    "\nbreak ",
    "\ncase ",
    "\nelse ",
    # Split by exceptions
    "\ntry ",
    "\nthrow ",
    "\nfinally ",
    "\ncatch ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def csharp() do
    @csharp
  end

  @sol [
    # Split along compiler information definitions
    "\npragma ",
    "\nusing ",
    # Split along contract definitions
    "\ncontract ",
    "\ninterface ",
    "\nlibrary ",
    # Split along method definitions
    "\nconstructor ",
    "\ntype ",
    "\nfunction ",
    "\nevent ",
    "\nmodifier ",
    "\nerror ",
    "\nstruct ",
    "\nenum ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\ndo while ",
    "\nassembly ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def sol() do
    @sol
  end

  @cobol [
    # Split along divisions
    "\nIDENTIFICATION DIVISION.",
    "\nENVIRONMENT DIVISION.",
    "\nDATA DIVISION.",
    "\nPROCEDURE DIVISION.",
    # Split along sections within DATA DIVISION
    "\nWORKING-STORAGE SECTION.",
    "\nLINKAGE SECTION.",
    "\nFILE SECTION.",
    # Split along sections within PROCEDURE DIVISION
    "\nINPUT-OUTPUT SECTION.",
    # Split along paragraphs and common statements
    "\nOPEN ",
    "\nCLOSE ",
    "\nREAD ",
    "\nWRITE ",
    "\nIF ",
    "\nELSE ",
    "\nMOVE ",
    "\nPERFORM ",
    "\nUNTIL ",
    "\nVARYING ",
    "\nACCEPT ",
    "\nDISPLAY ",
    "\nSTOP RUN.",
    # Split by the normal type of lines
    "\n",
    " ",
    ""
  ]
  def cobol() do
    @cobol
  end

  @lua [
    # Split along variable and table definitions
    "\nlocal ",
    # Split along function definitions
    "\nfunction ",
    # Split along control flow statements
    "\nif ",
    "\nfor ",
    "\nwhile ",
    "\nrepeat ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def lua() do
    @lua
  end

  @haskell [
    # Split along function definitions
    "\nmain :: ",
    "\nmain = ",
    "\nlet ",
    "\nin ",
    "\ndo ",
    "\nwhere ",
    "\n:: ",
    "\n= ",
    # Split along type declarations
    "\ndata ",
    "\nnewtype ",
    "\ntype ",
    "\n:: ",
    # Split along module declarations
    "\nmodule ",
    # Split along import statements
    "\nimport ",
    "\nqualified ",
    "\nimport qualified ",
    # Split along typeclass declarations
    "\nclass ",
    "\ninstance ",
    # Split along case expressions
    "\ncase ",
    # Split along guards in function definitions
    "\n| ",
    # Split along record field declarations
    "\ndata ",
    "\n= {",
    "\n, ",
    # Split by the normal type of lines
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def haskell() do
    @haskell
  end

  @powershell [
    # Split along function definitions
    "\nfunction ",
    # Split along parameter declarations (escape parentheses)
    "\nparam ",
    # Split along control flow statements
    "\nif ",
    "\nforeach ",
    "\nfor ",
    "\nwhile ",
    "\nswitch ",
    # Split along class definitions (for PowerShell 5.0 and above)
    "\nclass ",
    # Split along try-catch-finally blocks
    "\ntry ",
    "\ncatch ",
    "\nfinally ",
    # Split by normal lines and empty spaces
    "\n\n",
    "\n",
    " ",
    ""
  ]
  def powershell() do
    @powershell
  end
end
