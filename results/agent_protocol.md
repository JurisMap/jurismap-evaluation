# Agent protocol: defense generation (and chat) in JurisMap v1 and v2

Phase 2, task 3 (plan C1). Read from the code at `v1-expert-eval` (backend `3c417ff`, Streamlit frontend `203376f`) and `v2.2-eval-frozen` (backend `4dce1ab`, React frontend `e9a2483`). ADK behaviour was checked in the source of the pinned `google-adk==1.3.0`. File references are `repo:path`. The graphs are `figures/agent_graph_v1.mmd` and `figures/agent_graph_v2.mmd` (SVG alongside). The settings are in `reproducibility_table.md`. No prompt is quoted here; see `prompt_specs.md`.

## 1. Inputs common to both versions

A draft is requested with `POST /API_generate_defense` after three optional uploads, which are stored in the Flask session:

| Session variable | Filled by | Content |
| --- | --- | --- |
| `defense_template` | `/API_upload_defense_template` | Text of the uploaded DOCX, PDF or TXT (PDF text by PyMuPDF; OCR by the Gemini model when a PDF yields < 50 characters) |
| `case_file_content` | `/API_upload_case` | Text of the uploaded notice (same parser). In v2 the key is scoped per user, and the upload is also compressed, embedded and projected onto the map. |
| `case_description` | `/API_upload_case_description` | Free text typed by the user |
| `context_cases` | request field `selected_cases_ids` (v2 eval: `case_hashes`) | The selected decisions, each reduced to the 17 context fields (session, parties, rapporteur, lawyer, ementa, penalty, legal basis, acórdão, participants, **full report, full vote**, outcome) |

Both frontends cap the selection at **10 decisions**. v1 (Streamlit) refuses to call the API when the dashboard filters leave more than 10. v2 (React) lets the user click up to 10 decisions on the map.

## 2. Version 1: LangGraph fixed sequence

`backend:agents/router.py` builds a `StateGraph(MessagesState)` on every request.

### 2.1 State

The only graph state is `MessagesState.messages`, an append-only list of messages. Session variables are passed to the node functions as a closure argument; they are not graph state.

- Initial state: one `HumanMessage` with the request's `human_message`. The Streamlit client always sends "Elabore a defesa administrativa para o caso fornecido pelo usuário."
- **Every agent appends its output as a `HumanMessage`** with `name=<agent>`, not as an AI message. This matters twice:
  - The legal searcher takes "the last human message" as its search topic, which is therefore the case interpreter's output, not the user's request.
  - The supervisor labels every message "Humano" when it builds its transcript, including the other agents' outputs.

### 2.2 Order, reads and writes

| # | Node | Model | Reads | Writes |
| --- | --- | --- | --- | --- |
| 1 | `defense_structure_interpreter` | `CHATBOT_MODEL` (gemini-2.0-flash) | Its prompt with `{defense_template}` interpolated, **plus the template appended again** (or `agents/defense_structure.json` when no template was uploaded); all messages | Structure of the defense (sections and content) |
| 2 | `case_interpreter` | gemini-2.0-flash | Prompt with `{case_description}`, `{case_file_content}` and `{context_cases}` (the list of 17-field dicts, rendered as its Python repr); all messages | Case description and candidate precedents |
| 3 | `legal_searcher` | gemini-2.0-flash, 2 calls in one chat session | Call 1: the last `HumanMessage` (the case analysis) → up to 3 quoted queries (fallback: the whole message as one query). Tavily per query (advanced, ≤ 10 results, restricted to planalto.gov.br and the CRSNSP legislation page). Call 2: prompt + formatted search results + all messages | Extracts of statutes |
| 4 | `defense_creator` | `CHATBOT_REASONING_MODEL` (gemini-2.5-flash-preview-04-17) | Prompt (no placeholders) + all messages + "CASE INFORMATION" (`case_description`, `case_file_content`) | Draft defense |
| 5 | `supervisor` (review) | gemini-2.5-flash-preview-04-17 | Review prompt + the whole message history as a transcript + the review task | **Final draft** |

- Precedents reach nodes 3–5 only through the case interpreter's text; only node 2 sees the decisions themselves.
- Each node also returns `Command(goto=<next>)`, which agrees with the static edges.

### 2.3 Termination, errors and retries

- **Termination:** the static edge `supervisor → END`. There are no loops or conditional edges.
  - `prompts/supervisor_system_prompt.yaml` describes a routing supervisor (FINISH / REPEAT) but is **never loaded**. `utils/constants.py` defines the path, and nothing uses it.
- **Prompt sections dropped by the loader:** v1's `utils/system_prompt.py` renders only the sections `your_role`, `members`, `goal`, `rules`, `examples`, `other_notes` and `answer`.
  - The review prompt's `input` and `output_format` sections are therefore not sent. `output_format` is the instruction to answer only with the final text.
  - All other v1 prompts use only rendered sections. This is v1 as the experts used it; it is recorded here, not fixed (DECISIONS 2026-09-27, item 1).
- **Retries:** none in the code; the SDK defaults apply.
- **Errors in nodes 1–4:** an exception propagates out of `graph.stream`. `run_agents` then yields one `system` message ("Ocorreu um erro durante a execução: …") and the stream ends without a draft.
- **Errors in the supervisor:** caught inside the node. Its error text becomes the last message, and a `system` message follows.
- **Legal-search errors:** a Tavily error, including a missing API key, is caught in `execute_search` and turned into one result titled "Search Error". The pipeline **continues silently** without legislation.
- **Timeouts:** none in the code. gunicorn runs with `--timeout 0`. The Cloud Run request timeout used in May–June 2025 is not recorded in the repository.

### 2.4 Streaming

- `graph.stream(..., stream_mode="values")` yields the state after each node. The backend sends the last message as one server-sent event `{"role": <node name>, "content": …}`: five agent messages, then a `system` message only on errors.
- Streamlit shows every message as a chat bubble after stripping code fences, so **the user sees every intermediate output**. The last bubble (`supervisor`) is the draft.

### 2.5 Chat (v1)

`/API_chatbot` makes one call to `CHATBOT_MODEL`. The prompt contains:

- the system prompt, then a data dictionary;
- the top `min(10, n)` selected decisions, ranked by cosine similarity between the question's embedding (task type not set) and the full-report embeddings, each with the 17 fields;
- the whole conversation history;
- the new question.

## 3. Version 2: Google ADK (supervisor, parallel research, creator, merger)

Code: `backend:agents_adk/router_adk.py`, `workflow_coordinator.py`, `common.py` and one module per agent. Every generation role uses `gemini-3.7-flash` on the `global` endpoint.

### 3.1 Request flows

| Request | Flow |
| --- | --- |
| `phase` absent or `"generation"` (legacy, default) | `run_adk_agents`: supervisor check. If `clarification_needed`, the stream ends with the questions. If `ready`, generation runs in the same stream. |
| `phase="supervisor"` or `"clarification"` (React clarification dialog) | `run_supervisor_check`; if the supervisor answers `ready`, `run_defense_generation` follows in the same stream |
| `skip_supervisor=true` (EVAL_MODE=1 only) | `run_defense_generation` directly; used for the ablation runs |
| `ablation_mode="single_prompt"` | Supervisor check (unless skipped), then one model call (§ 3.6) |

- **The supervisor has no memory across requests.** Every request builds a new coordinator and a new in-memory ADK session with a fresh `uuid4` session id.
  - A clarification answer therefore reaches the supervisor only as that request's `human_message`, alongside the session variables. The questions it asked before are not in its context.
- **Unparsable supervisor answers:** the code parses the text between the first `{` and the last `}`. If there is no valid JSON with a `status`, the answer is **treated as `ready`**.

### 3.2 ADK state and what each agent sees

- **Instructions:** built once per request by rendering each agent's YAML prompt with the session variables, every section in file order.
  - `{case_analysis_result}`, `{structure_analysis_result}` and `{legal_search_result}` stay as placeholders.
  - An `InstructionProvider` fills them from the ADK session state when the agent runs, which also avoids ADK's own `{…}` templating of user text.
- **Initial session state:** the serialisable session variables plus the ablation overrides (§ 3.5).
- **Outputs:** the research agents and the creator write their text to state via `output_key`. The merger has no `output_key`.
- **Conversation contents (ADK 1.3.0, `flows/llm_flows/contents.py`):** an agent also receives earlier session events.
  - Other agents' outputs are converted into user turns ("For context: [Agent] said: …").
  - Events are filtered by branch. `ParallelAgent` runs each sub-agent in its own branch (`ParallelResearchTeam.<name>`), so **the three research agents do not see each other**; each sees only the trigger message.
  - The creator and the merger run without a branch and see every earlier event.

Which inputs each agent receives (✓ = interpolated in its instruction; H = only through the conversation history; blank = not at all):

| Agent | `case_description` | `case_file_content` (uploaded notice) | `defense_template` | `context_cases` (≤ 10 × 17 fields) | Research results | Creator's draft |
| --- | --- | --- | --- | --- | --- | --- |
| SupervisorAgent | ✓ | ✓ | ✓ | ✓ | | |
| CaseInterpreter | ✓ | ✓ | | ✓ | | |
| LegalSearcher | ✓ | | | ✓ | | |
| DefenseStructureInterpreter | | | ✓ | | | |
| DefenseCreator | ✓ | H (via the case analysis) | ✓ | ✓ | ✓ and H (twice) | |
| MergerAgent | ✓ | H (via the case analysis) | ✓ | ✓ | H | H |

Consequences for the evaluation:

1. **With an uploaded notice and no typed description (the smoke test, and Phase 8 as planned), the LegalSearcher never sees the notice.** It searches legislation from the precedents alone.
   - Its prompt says it will receive the case interpreter's analysis. It cannot, because both run in parallel.
   - v1's legal searcher did receive the case analysis (§ 2.2).
   - The creator and merger see the notice only as the CaseInterpreter summarised it.
   - Decided (DECISIONS 2026-09-27): Phase 8 supplies the notices as uploads only, as users do, so v2 is measured as it runs.
2. **The 10 precedents, with full reports and votes, are placed in five instructions** (supervisor, case interpreter, legal searcher, creator, merger), and the creator gets the research results twice. This is why a full draft costs about 261k prompt tokens (smoke test).
3. **The merger's prompt names its inputs as state keys** (`case_analysis_result`, …, `defense_creation_result`) without interpolating them; one name is misspelt (`legal_research_result`). It gets them only through the conversation history, which works, but the prompt describes a different mechanism.

### 3.3 Order and termination

- **Order:** SupervisorAgent → `SequentialAgent`:
  1. `ParallelAgent` (CaseInterpreter, LegalSearcher, DefenseStructureInterpreter, run concurrently);
  2. DefenseCreator;
  3. MergerAgent.
- **Research completion order varies** between runs. The smoke-test logs record, for example, DefenseStructureInterpreter → CaseInterpreter → LegalSearcher in `full`, and CaseInterpreter → DefenseStructureInterpreter → LegalSearcher in `no_precedents`.
- **Termination:** the sequential agent ends after the merger. There are no loops, no agent transfer, and no tools other than Google Search grounding on the LegalSearcher.
- **The draft** is the MergerAgent's text, after the fact guard (§ 3.4).
  - If that text is empty (for example, blocked twice for recitation), the stream ends with an `error` message and no draft ("A defesa final não foi gerada pelo modelo…"), and the eval log records the error.

### 3.4 Fact guard (v2.2)

- `utils/fact_guard.py` checks the merger's draft deterministically against the sources:
  - the case description and uploaded notice;
  - the template;
  - the precedents (as JSON);
  - the LegalSearcher's text (the fixed notice in `no_legal_search`; nothing in `single_prompt`, whose search results are not available as text).
- **Unsupported case identifiers** (process numbers, CPF/CNPJ, numbered documents, SEI numbers) are replaced by `[A PREENCHER: …]`.
- **Unsupported dates, amounts and legal-act citations** are flagged. Core statutes such as DL 73/1966 need no source.
- The result is streamed as a separate "Verificação de Fatos" message and logged (`fact_check`).

### 3.5 Ablations (EVAL_MODE flags; default `full` = production)

| Mode | Change |
| --- | --- |
| `no_legal_search` | LegalSearcher not created; `legal_search_result` preset to a fixed notice ("Pesquisa de legislação não realizada nesta execução.") |
| `no_precedents` | `context_cases` and `selected_cases_ids` emptied for the generation agents (the supervisor still sees the original inputs) |
| `no_template` | DefenseStructureInterpreter not created and no template passed; `structure_analysis_result` preset to the structure prompt's `default_template` |
| `single_prompt` | One `gemini-3.7-flash` call. The system instruction is a header, then the five generation agents' rendered prompts in pipeline order (their `{…_result}` placeholders replaced by "the result of task N"), then a footer. Google Search grounding is on; the fact guard runs on the output. |

### 3.6 Retries, timeouts and errors

- **HTTP level (google-genai):**
  - 6 attempts on 408/429/500/502/503/504, backoff 2–60 s, 240 s per call.
  - The primary service-account key gets 6 attempts first. On failure the call is repeated with the project's own credentials, and the primary key is skipped for 60 s.
- **Blocked responses:** a response with no content and an error code (RECITATION, SAFETY) is **retried once**. This covers ADK agents (`FallbackGemini`) and direct calls (single prompt, chat, OCR, summaries). Retried responses are logged as `agent_blocked_retry`.
  - A second block passes the empty response on. For the merger, that means the error message of § 3.3.
  - For a research agent or the creator it means an empty result in state, which the pipeline does not detect.
- **Exceptions:**
  - Inside the ADK run: caught by the coordinator, which yields `{"role": "system", "status": "error", …}`, and the stream ends.
  - Event-processing errors are logged and skipped.
- **Request timeouts:**
  - **Production:** Cloud Run `--timeout 300` and gunicorn `--timeout 300`. The smoke test's full draft through the supervisor took **307 s**, so production can hit this limit.
  - **Evaluation server:** Flask `app.run` (`jm_eval/server.py`), with **no request timeout**. Phases 8–9 should report the share of runs over 300 s as would-be production timeouts.

### 3.7 Streaming

- `Runner.run_async` runs with the default run configuration (no token streaming), so each model response is **one** event.
- The backend sends one server-sent event per event: `{"role": <display name>, "agent_id": …, "content": …}`.
- After the merger's event come the fact-check message and a final `system` message (`status: completed` or `error`).
- The supervisor's messages carry `is_partial`, `is_final`, `status` and `questions`.
- The React client shows the agents' progress and the final draft.
- With `EVAL_MODE=1`, `utils/eval_log.py` writes one JSON line per request with:
  - the flags;
  - every LLM call: role or agent, model, token usage, latency, finish reason, credentials used;
  - each agent event's text, plus the Google Search queries from the grounding metadata;
  - the fact-check report.

### 3.8 Chat (v2)

- **Default (`full_text`):** as v1 (§ 2.5), with these differences:
  - Gemini 3.7 Flash;
  - question embedded with `SEMANTIC_SIMILARITY` at 3,072 dimensions;
  - decisions ranked on the compressed-report embeddings;
  - context fields as extracted (no `*_curated` copies);
  - one extra system-prompt rule.
- **Evaluation-only `hierarchical_summary`:**
  - each of up to 30 retrieved decisions is first summarised to at most 300 tokens (8 calls in parallel, cached per server process);
  - the answer is written over the summaries plus 12 metadata fields.

## 4. Protocol differences that matter for Phases 6–9

| Topic | v1 | v2 |
| --- | --- | --- |
| Who sees the precedents | Case interpreter only | Supervisor, case interpreter, legal searcher, creator, merger |
| Who sees the uploaded notice | Case interpreter and creator (appended); the others through earlier outputs | Supervisor and case interpreter; creator and merger through the case analysis; **legal searcher: no** |
| Legal-search topic | The case analysis (sequential) | The typed description and the precedents only (parallel) |
| Silent degradations | Tavily error → no legislation, no signal | Second block of a research or creator response → empty input to the next agent, no signal |
| Visible failure | Exception → system message | Empty final draft → error message; exception → error message |
| Human in the loop | None | Supervisor may ask questions (skipped in the ablation runs) |
| Request time limit | None in code | 300 s in production; none on the evaluation server |
