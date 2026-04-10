# Data Nodes

Data nodes handle text rendering, variable management, code execution, branching
logic, and tool integration -- all without invoking an LLM unless explicitly needed.

> See also: [Memory Nodes](memory-nodes.md) for persistent storage across nodes.

---

## TextNode

| | |
|---|---|
| **Type enum** | `TextNode1` |
| **Class** | `TextNode` |
| **Version** | 1.0 |

Renders a Jinja2 template against the current thought metadata and appends the
result to the thought text. Use it whenever you need to compose dynamic messages
from variables produced by earlier nodes.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | `"This is a default text."` | Jinja2 template string to render |

### How it works

1. Reads the `text` metadata field (a Jinja2 template).
2. Renders the template with the full thought `metadata` dict as context.
3. Appends the rendered string to the thought.

### Jinja2 template examples

```jinja
Hello, {{ user_name }}! Your order #{{ order_id }} is confirmed.

{% if score > 80 %}Great job!{% else %}Keep trying.{% endif %}

{% for item in items %}- {{ item.name }}: ${{ item.price }}
{% endfor %}
```

### Common use cases

- Format LLM output into a user-facing message.
- Build prompts dynamically from collected variables.

---

## StaticNode1

| | |
|---|---|
| **Type enum** | `StaticAssistant` |
| **Class** | `StaticNode1` |
| **Version** | 1.0 |

Outputs a fixed text string without any template processing or LLM call. The
text is taken verbatim from the node metadata and appended to the thought.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `static_text` | `str` | `"This is a default static text."` | Literal text to output |

### How it works

1. Reads the `static_text` metadata field.
2. Appends the raw string to the thought -- no rendering, no LLM.

### Common use cases

- Display fixed instructions or disclaimers.
- Provide a deterministic fallback message.

---

## CodeNode

| | |
|---|---|
| **Type enum** | `Code1` |
| **Class** | `CodeNode` |
| **Version** | 1.0.0 |

Declares a block of custom Python code executed by the runtime. The node itself
is a no-op during `think()`; the engine handles execution externally.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `code` | `str` | `""` | Python source code to execute |
| `filename` | `str` | `""` | Optional filename for the code block |

### How it works

1. Stores `code` and `filename` in metadata.
2. `think()` is a no-op -- the runtime picks up the code and executes it.
3. Does not accept incoming or outgoing edges (`accepts_incoming_edges=False`,
   `accepts_outgoing_edges=False`).

### Common use cases

- Custom data transformation or validation logic.
- Define helper functions for downstream expression evaluators.

---

## VarNode

| | |
|---|---|
| **Type enum** | `VarNode1` |
| **Class** | `VarNode` |
| **Version** | 1.0 |

Evaluates a Python expression and stores the result under a named key in
thought metadata. The primary way to create derived variables without an LLM.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `""` | Key under which the result is stored in metadata |
| `expression` | `str` | `""` | Python expression to evaluate |

### How it works

1. Reads `name` and `expression` from metadata.
2. Evaluates using the `_expression_evaluator` (or restricted `eval()` fallback).
3. Stores the result in thought metadata under the given `name`.

### Expression examples

```python
"price * quantity"                        # arithmetic
"user_name.upper()"                       # string method
"'VIP' if total > 1000 else 'Regular'"   # conditional
"[x for x in items if x['in_stock']]"    # list comprehension
```

### Common use cases

- Compute derived values (totals, scores, flags).
- Transform data between nodes without an LLM.
- Set boolean flags for downstream [StaticDecision1](#staticdecision1) nodes.

---

## TextToVariableNode

| | |
|---|---|
| **Type enum** | `TextToVariableNode1` |
| **Class** | `TextToVariableNode` |
| **Version** | 1.0 |

Reads the current thought text and stores it as a named variable in thought
metadata, bridging text output and variable-based processing.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `variable_name` | `str` | `"custom_variable"` | Metadata key to store the thought text under |

### How it works

1. Reads `variable_name` from metadata.
2. Takes `ctx.thought.text` (the current thought content).
3. Stores it in metadata as `{variable_name: thought_text}`.

### Common use cases

- Capture LLM-generated text for use in expressions or templates.
- Feed text output into a [VarNode](#varnode) or [WriteMemoryNode](memory-nodes.md#writememorynode) downstream.

---

## ProgramRunner1

| | |
|---|---|
| **Type enum** | `ProgramRunner1` |
| **Class** | `ProgramRunner1` |
| **Version** | 1.0 |

Executes a registered tool (program) by version ID with given parameters. The
result is appended to the thought text.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `program_version_id` | `str \| None` | `None` | Identifier of the tool version to run |
| `parameters` | `dict` | `{}` | Parameters to pass to the tool |

### How it works

1. Retrieves `_program_executor` from context.
2. Calls `program_executor(program_id, parameters, ctx)`.
3. Appends the result (if any) to the thought text.

### Common use cases

- Invoke external APIs or tools from within a flow.
- Chain with [StaticProgramParameters1](#staticprogramparameters1) for fully
  deterministic tool calls.

---

## StaticDecision1

| | |
|---|---|
| **Type enum** | `StaticDecision1` |
| **Class** | `StaticDecision1` |
| **Version** | 1.0 |

Evaluates a Python expression and routes execution to one of two outgoing edges
based on truthiness. Provides deterministic branching without an LLM.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `expression` | `str` | `""` | Python expression that resolves to a truthy or falsy value |

### How it works

1. Evaluates `expression` against thought metadata (via `_expression_evaluator`
   or restricted `eval()`).
2. Looks at predecessor edges: `main_direction=True` is the "true" branch,
   `main_direction=False` is the "false" branch.
3. Sets `next_assistant_node_id` to the chosen branch; only that branch continues
   (`traverse_out=SpawnPickedNode`).

### Expression examples

```python
"score >= 80"                           # numeric comparison
"status == 'approved'"                  # string check
"'admin' in user_roles"                 # membership test
"age >= 18 and country == 'US'"         # compound condition
```

### Common use cases

- Route users based on computed flags or thresholds.
- Implement if/else logic without LLM overhead.

---

## StaticMerge1

| | |
|---|---|
| **Type enum** | `StaticMerge1` |
| **Class** | `StaticMerge1` |
| **Version** | 1.0 |

Waits for all incoming branches to complete (`traverse_in=AwaitAll`), then
outputs static text. The static counterpart to an LLM-based merge node.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `static_text` | `str` | `""` | Text to output after all branches arrive |

### How it works

1. Execution pauses until every incoming edge has delivered a thought.
2. Reads `static_text` from metadata and appends it to a new thought.

### Common use cases

- Rejoin parallel branches with a fixed status message.
- Synchronize multiple paths before continuing.

---

## StaticProgramParameters1

| | |
|---|---|
| **Type enum** | `StaticProgramParameters1` |
| **Class** | `StaticProgramParameters1` |
| **Version** | 1.0 |

Injects static tool parameters into thought metadata so a downstream
[ProgramRunner1](#programrunner1) can consume them. No LLM is involved.

### Configuration

| Field | Type | Default | Description |
|---|---|---|---|
| `parameters` | `dict` | `{}` | Key-value pairs to inject into metadata |
| `program_name` | `str` | `""` | Name of the target program (informational) |

### How it works

1. Reads `parameters` from metadata.
2. Merges the dict directly into thought metadata.

### Common use cases

- Pre-configure tool inputs with known values.
- Pair with [ProgramRunner1](#programrunner1) for fully static tool execution.
