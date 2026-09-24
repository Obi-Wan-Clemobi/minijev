# State Machine Flow Builder for minijev

*Design specification - 2026-09-24*

## 1. Overview

Build a visual flow builder that lets users chain minijev decisions into state machines. Users drag nodes onto a canvas, connect them with transitions, and execute multi-stage decision workflows where each stage feeds its result into the next.

**Core use case:** Tool selection workflow
1. "What type of tool is needed?" → "search tool"
2. "Which search tool?" → "grep"
3. Return the full decision chain with probabilities

The system maintains structured decision history throughout execution, giving full traceability of the reasoning path.

## 2. Goals

1. **Visual flow editor**: Drag-and-drop interface for building decision workflows
2. **State machine executor**: Run flows with structured state accumulation
3. **Full feature set**: Conditional branching, parallel execution, templates, export/import
4. **Minijev integration**: All decisions use minijev primitives (Noul, Choice, Score)
5. **Execution visualization**: Show the path taken through the flow with decision history

## 3. Non-Goals

- Visual programming (this is decision chaining, not general computation)
- Real-time collaboration (single-user editing only)
- Version control UI (flows are JSON files - use git)
- Production deployment features (auth, rate limiting, multi-tenant)

## 4. Architecture

### 4.1 Three-Layer System

```
┌─────────────────────────────────────────┐
│  Flow Editor (Next.js + React Flow)     │
│  - Visual canvas for building flows     │
│  - Node palette and property editor     │
│  - Execution visualization              │
└─────────────┬───────────────────────────┘
              │ HTTP
┌─────────────┴───────────────────────────┐
│  API Layer (FastAPI)                    │
│  - POST /flows/execute                  │
│  - GET/POST/DELETE /flows               │
└─────────────┬───────────────────────────┘
              │
┌─────────────┴───────────────────────────┐
│  Flow Executor (Python)                 │
│  - State machine runner                 │
│  - Calls minijev for each state         │
│  - Accumulates decision history         │
└─────────────────────────────────────────┘
```

### 4.2 File Structure

```
poc/
├── flows/
│   ├── executor.py          # State machine execution engine
│   ├── schema.py            # Flow definition validation (Pydantic)
│   ├── templates/           # Pre-built flow templates
│   │   ├── tool-selection.json
│   │   └── customer-triage.json
│   └── definitions/         # User-saved flows (gitignored)
└── server.py                # Add /flows endpoints

web/
├── app/
│   └── flows/
│       ├── page.tsx         # Main flow builder page
│       ├── FlowCanvas.tsx   # React Flow canvas component
│       ├── NodePalette.tsx  # Drag-to-add node types
│       ├── NodeEditor.tsx   # Side panel for editing nodes
│       ├── ExecutionViewer.tsx  # Show execution trace
│       └── nodes/           # Custom node components
│           ├── NoulNode.tsx
│           ├── ChoiceNode.tsx
│           └── ScoreNode.tsx
```

## 5. Flow Definition Format

Flows are stored as JSON with this schema:

```json
{
  "id": "tool-selector-v1",
  "name": "Tool Chain Selection",
  "description": "Select the appropriate tool type, then specific tool",
  "version": "1.0",
  "initial_state": "select_tool_type",
  "states": {
    "select_tool_type": {
      "id": "select_tool_type",
      "type": "choice",
      "instructions": "What type of tool is needed for this task?",
      "criteria": {
        "search": "Grep, ripgrep, find patterns in files",
        "file": "Read, write, move, delete files",
        "network": "HTTP requests, APIs, downloads"
      },
      "position": {"x": 100, "y": 100},
      "transitions": [
        {
          "from_answer": "search",
          "target": "select_search_tool",
          "condition": {"confidence_gte": 0.7}
        },
        {
          "from_answer": "search",
          "target": "clarify_search",
          "condition": {"confidence_lt": 0.7}
        },
        {
          "from_answer": "file",
          "target": "select_file_tool"
        },
        {
          "from_answer": "network",
          "target": "DONE"
        }
      ]
    },
    "select_search_tool": {
      "id": "select_search_tool",
      "type": "choice",
      "instructions": "Which search tool should we use?",
      "criteria": {
        "grep": "Standard Unix grep",
        "ripgrep": "Faster modern alternative"
      },
      "position": {"x": 400, "y": 50},
      "transitions": [
        {"from_answer": "grep", "target": "DONE"},
        {"from_answer": "ripgrep", "target": "DONE"}
      ]
    },
    "clarify_search": {
      "id": "clarify_search",
      "type": "noul",
      "instructions": "Does the task require searching inside file contents (vs just finding file names)?",
      "criteria": {
        "true": "Search file contents",
        "false": "Search file names"
      },
      "position": {"x": 400, "y": 200},
      "transitions": [
        {"from_answer": true, "target": "select_search_tool"},
        {"from_answer": false, "target": "select_file_tool"}
      ]
    }
  }
}
```

### 5.1 Special States

- **`DONE`**: Terminal state - execution stops successfully
- **`ERROR`**: Terminal state - execution failed (not implemented yet)

### 5.2 Conditional Transitions

Transitions can have optional conditions based on confidence:

```json
{
  "from_answer": "search",
  "target": "next_state",
  "condition": {
    "confidence_gte": 0.7,
    "confidence_lt": 0.9
  }
}
```

Multiple transitions from the same answer are evaluated in order. The first matching condition wins.

### 5.3 Parallel Execution (Future)

For asking multiple independent questions at once:

```json
{
  "id": "parallel_triage",
  "type": "parallel",
  "instructions": "Analyze this ticket on multiple dimensions",
  "questions": {
    "urgency": {
      "type": "noul",
      "instructions": "Is this urgent?"
    },
    "sentiment": {
      "type": "score",
      "instructions": "Customer sentiment?",
      "criteria": ["happy", "neutral", "frustrated", "angry"]
    }
  },
  "transitions": [
    {
      "condition": {
        "all_of": [
          {"urgency": true},
          {"sentiment_gte": 2}
        ]
      },
      "target": "escalate"
    },
    {"target": "normal_queue"}
  ]
}
```

This uses minijev's existing batch capability - all questions see the same state and are answered in one minijev call.

## 6. State Accumulation Model

The executor maintains structured history as decisions are made:

```python
{
  "original_query": "find all TODO comments in Python files",
  "flow_id": "tool-selector-v1",
  "execution_id": "exec_abc123",
  "started_at": "2026-09-24T10:30:00Z",
  "decisions": [
    {
      "state_id": "select_tool_type",
      "question_type": "choice",
      "answer": "search",
      "probabilities": {
        "search": 0.95,
        "file": 0.04,
        "network": 0.01
      },
      "confidence": 0.93,
      "timestamp": "2026-09-24T10:30:01Z"
    },
    {
      "state_id": "select_search_tool",
      "question_type": "choice",
      "answer": "grep",
      "probabilities": {
        "grep": 0.82,
        "ripgrep": 0.18
      },
      "confidence": 0.64,
      "timestamp": "2026-09-24T10:30:02Z"
    }
  ],
  "final_answer": {
    "tool_type": "search",
    "specific_tool": "grep"
  },
  "status": "completed",
  "completed_at": "2026-09-24T10:30:02Z"
}
```

### 6.1 State Passed to Minijev

At each stage, minijev receives:

```json
{
  "state": {
    "original_query": "find all TODO comments in Python files",
    "decisions": [
      {
        "stage": "tool_type",
        "choice": "search",
        "confidence": 0.93
      }
    ]
  },
  "questions": {
    "next_choice": {
      "type": "choice",
      "instructions": "Which search tool should we use?",
      "criteria": {...}
    }
  }
}
```

The state object is serialized as JSON (minijev already handles this per DESIGN.md §4).

## 7. Execution Model

### 7.1 Sequential Execution Algorithm

```python
def execute_flow(flow: FlowDefinition, query: str) -> ExecutionTrace:
    state = {
        "original_query": query,
        "decisions": []
    }
    
    current_state_id = flow.initial_state
    max_depth = 20  # Safety limit
    depth = 0
    
    while current_state_id != "DONE" and depth < max_depth:
        state_def = flow.states[current_state_id]
        
        # Call minijev with accumulated state
        response = minijev.ask({
            "state": state,
            "questions": {
                "next": build_question(state_def)
            }
        })
        
        answer = response.answers["next"]
        
        # Append to history
        state["decisions"].append({
            "state_id": current_state_id,
            "question_type": state_def.type,
            "answer": extract_answer(answer),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
            "timestamp": now()
        })
        
        # Find next state
        current_state_id = find_transition(
            state_def,
            extract_answer(answer),
            answer.get("confidence", 1.0)
        )
        
        depth += 1
    
    if depth >= max_depth:
        raise MaxDepthExceeded("Flow exceeded maximum depth")
    
    return ExecutionTrace(state)
```

### 7.2 Error Handling

- **No matching transition**: Raise `NoTransitionFound` exception
- **Max depth exceeded**: Raise `MaxDepthExceeded` exception  
- **Minijev API error**: Propagate with context (which state failed)
- **Invalid flow definition**: Validate before execution (Pydantic schema)

## 8. React Flow Integration

### 8.1 Custom Node Types

Three node types corresponding to minijev primitives:

**NoulNode (Yes/No Question)**
```tsx
// Two output handles: "true" and "false"
// Properties: instructions, criteria (optional true/false descriptions)
```

**ChoiceNode (Multiple Choice)**
```tsx
// One output handle per option (dynamic based on criteria)
// Properties: instructions, criteria (key -> description map)
```

**ScoreNode (Ordinal Scale)**
```tsx
// Multiple output handles for score ranges
// Properties: instructions, criteria (ordered array)
// Transitions can specify: score_gte, score_lt, score_eq
```

### 8.2 Node Editor Panel

Side panel shows when a node is selected:
- Question type (readonly after creation)
- Instructions (textarea)
- Criteria editor:
  - Choice: key-value pairs (add/remove options)
  - Score: ordered list (drag to reorder levels)
  - Noul: optional true/false descriptions
- Validation errors (missing fields, invalid criteria)

### 8.3 Edge Types

Two visual styles:
- **Solid edge**: Unconditional transition (always follows this answer)
- **Dashed edge**: Conditional transition (shows condition in label)

Edge labels show:
- Answer value (e.g., "search")
- Condition if present (e.g., "conf ≥ 0.7")

### 8.4 Execution Visualization

During/after execution:
- Highlight the path taken through the graph
- Show decision values on edges
- Display confidence scores as node badges
- Animate the flow (slow-motion replay)

## 9. API Endpoints

### 9.1 Execute Flow

```
POST /flows/execute
Content-Type: application/json

{
  "flow": {...},  // Complete flow definition
  "query": "find all TODO comments in Python files"
}

Response 200:
{
  "execution_id": "exec_abc123",
  "status": "completed",
  "trace": {...},  // Full execution trace from §6
  "duration_ms": 1234
}
```

### 9.2 Flow Management

```
GET /flows
Response: [
  {"id": "tool-selector-v1", "name": "Tool Chain Selection", "created_at": "..."},
  ...
]

POST /flows
Body: {"flow": {...}}
Response: {"id": "flow_xyz", "created_at": "..."}

GET /flows/{id}
Response: {"flow": {...}}

DELETE /flows/{id}
Response: 204 No Content
```

### 9.3 Templates

```
GET /flows/templates
Response: [
  {
    "id": "tool-selection",
    "name": "Tool Chain Selection",
    "description": "Two-stage tool selection (type, then specific tool)",
    "flow": {...}
  },
  {
    "id": "customer-triage",
    "name": "Customer Support Triage",
    "description": "Urgency + team + sentiment analysis",
    "flow": {...}
  }
]
```

Templates are stored in `poc/flows/templates/*.json` and loaded at startup.

## 10. UI Features

### 10.1 Flow Builder Page (`/flows`)

**Layout:**
- Left sidebar: Node palette (drag Noul/Choice/Score onto canvas)
- Center: React Flow canvas (zoom, pan, minimap)
- Right sidebar: Node editor (appears when node selected)
- Top toolbar: Save, Load, New, Templates, Run

**Canvas Controls:**
- Drag nodes from palette
- Click to select nodes/edges
- Click edge to edit transition conditions
- Delete key removes selected items
- Ctrl+Z/Ctrl+Y for undo/redo

### 10.2 Execution View

**Run Flow Dialog:**
1. User clicks "Run" button
2. Modal appears: "Enter your query"
3. User types: "find all TODO comments in Python files"
4. Click "Execute"
5. Modal shows animated execution:
   - Current state highlighted
   - Decision appears as it's made
   - Transitions light up as they're followed
6. Final result shows full trace

**Execution History:**
- List of past runs (in-memory, cleared on refresh)
- Click a run to replay its visualization
- Export trace as JSON

### 10.3 Templates Library

Modal with cards:
- Template name and description
- Preview image (static screenshot of flow)
- "Use Template" button (loads into canvas)
- "View Details" expands to show all states

Users can save their flows as templates (personal templates directory).

### 10.4 Export/Import

**Export:**
- "Export" button downloads flow as JSON
- Filename: `{flow-name}-{timestamp}.json`

**Import:**
- "Load" button opens file picker
- Validates JSON schema
- Shows validation errors if invalid
- Loads valid flows onto canvas

## 11. Validation

### 11.1 Flow Validation (Runtime)

Before execution, validate:
- `initial_state` exists in `states`
- Every transition target exists (or is "DONE")
- No unreachable states (warning, not error)
- No cycles without exit conditions (warning)
- All Choice options have transitions
- Score transition conditions are valid ranges

### 11.2 Node Validation (Editor)

Real-time validation in editor:
- Instructions not empty
- Choice has at least 2 options
- Score has 2-10 levels
- No duplicate option keys
- All outputs connected (warning)

Show validation badges on nodes (red = error, yellow = warning).

## 12. Testing Strategy

### 12.1 Unit Tests

**Executor (`poc/flows/executor.py`):**
- Execute simple linear flow
- Execute branching flow with conditions
- State accumulation correctness
- Error cases (no transition, max depth, invalid flow)

**Schema validation (`poc/flows/schema.py`):**
- Valid flow definitions pass
- Invalid definitions fail with clear errors

### 12.2 Integration Tests

- Full execution through API endpoint
- Save/load flows
- Template loading

### 12.3 Frontend Tests

**Component tests:**
- Node rendering (all three types)
- Edge rendering (conditional vs unconditional)
- Node editor updates node properties

**E2E tests (Playwright):**
- Create a flow from scratch
- Run a flow and view results
- Load a template
- Export and re-import a flow

## 13. Implementation Phases

Given the full-featured scope, break into incremental milestones:

### Phase 1: Core Executor (Backend)
- Flow schema definition (Pydantic models)
- State machine executor (no conditions yet)
- API endpoints (execute, save, load)
- Unit tests

**Deliverable:** Can execute flows via API, no UI yet

### Phase 2: Basic Visual Editor (Frontend)
- React Flow canvas setup
- Three node types (basic rendering)
- Drag-drop from palette
- Node editor panel (basic)
- Save/load flows

**Deliverable:** Can build flows visually, execute via API

### Phase 3: Execution Visualization
- Run dialog with query input
- Animated execution viewer
- Decision history display
- Highlight path taken

**Deliverable:** Full visual feedback during execution

### Phase 4: Advanced Features
- Conditional transitions (confidence-based)
- Condition editor UI
- Templates library
- Export/import
- Validation badges
- Undo/redo

**Deliverable:** Production-ready flow builder

### Phase 5: Parallel Execution (Optional)
- Parallel node type
- Complex condition syntax
- Condition builder UI

**Deliverable:** Can ask multiple questions in one stage

## 14. Open Questions & Future Work

### 14.1 Variables and Data Extraction

Currently decisions are recorded but not easily referenced. Future:
```json
{
  "state_id": "extract_urgency",
  "output_variable": "is_urgent",
  "transitions": [
    {"from_answer": true, "target": "high_priority_flow"}
  ]
}
```

Then later states can reference `$is_urgent` in their instructions.

### 14.2 Loops and Iteration

Current design is DAG (no cycles). Could add:
- Loop nodes that repeat a subflow N times
- While loops with exit conditions
- Requires more sophisticated cycle detection

### 14.3 Sub-flows

Compose flows from smaller flows:
```json
{
  "type": "subflow",
  "flow_id": "determine_priority",
  "transitions": [
    {"from_result": "high", "target": "..."}
  ]
}
```

### 14.4 Integration with Existing Pages

The flow builder is a new standalone page. Future integration:
- Run flows from the main playground page
- Use flows as the "question" in existing UI
- Embed small flows inline (e.g., in the comparison page)

## 15. Dependencies

**New Dependencies:**
- `reactflow` (React Flow library) - MIT license
- `@xyflow/react` (React Flow v12+)
- `zustand` (state management for React Flow) - MIT license

**Existing Dependencies (no changes):**
- FastAPI, Pydantic (backend)
- Next.js, React, TailwindCSS (frontend)
- minijev POC (`poc/minijev_poc.py`)

## 16. Success Criteria

The implementation is complete when:

1. ✅ User can create a tool-selection flow in the visual editor
2. ✅ Flow can be saved and reloaded
3. ✅ Executing the flow with a query returns correct structured trace
4. ✅ Execution visualization shows the path taken
5. ✅ Conditional transitions work (confidence-based branching)
6. ✅ At least 2 templates are included
7. ✅ Export/import works
8. ✅ All unit and integration tests pass
9. ✅ The tool-selection example from §1 works end-to-end

## 17. Design Decisions

### Why React Flow instead of custom canvas?
React Flow provides zoom, pan, selection, undo/redo, minimap, and edge routing for free. Building these from scratch would take weeks and introduce bugs. The library is well-maintained and widely used (50k+ GitHub stars).

### Why store flows as JSON files instead of a database?
- Simple deployment (no DB to manage)
- Git-friendly (version control flows)
- Easy export/import
- Sufficient for single-user POC
- Can migrate to DB later if needed

### Why execute server-side instead of client-side?
- Minijev runs in Python on the server
- Execution traces need to be reproducible (same flow + query = same result)
- Easier to add auth/rate-limiting later
- Client just visualizes, server is source of truth

### Why structured state accumulation instead of text?
- Enables conditional logic (if confidence < 0.7, branch)
- Clean for later referencing decisions (variables)
- Better debugging (can inspect exact probabilities)
- Serializes clearly for logging
- Text concatenation loses structure

## 18. Non-Functional Requirements

**Performance:**
- Flow execution should complete in ~500ms per state (limited by minijev speed)
- Canvas should handle 50+ nodes without lag
- Save/load should be instant (<100ms)

**Usability:**
- First flow built in <5 minutes (with template)
- Clear error messages for validation failures
- Intuitive node connections (drag from output to input)

**Maintainability:**
- Clear separation: executor (pure Python), API (FastAPI), UI (React)
- Comprehensive tests for executor logic
- Schema validation prevents corrupt flows

**Compatibility:**
- Works in Chrome, Firefox, Safari (latest versions)
- No mobile UI required (desktop-only editor)
- Flow JSON schema versioned (forward-compatible)
