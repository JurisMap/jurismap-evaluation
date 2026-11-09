# Prompt specifications: JurisMap v1 and v2

Phase 2, task 4 (plan C2). The prompts are commercially sensitive, so this file **describes** them and quotes at most two short excerpts per prompt (≤ 25 words each, marked «…», with an English translation). Verbatim copies stay in `data/private/prompts_verbatim/` (git-ignored). `scripts/21_prompt_privacy.py` writes them there and checks this folder: every «…» excerpt must occur in a source prompt and be ≤ 25 words, and no other run of prompt text of 10 or more words may appear anywhere in `results/`.

Versions: v1 = `v1-expert-eval` (backend `3c417ff`, scrapers `2734e6b`); v2 = `v2.2-eval-frozen` (backend `4dce1ab`, scrapers `69385a3`). All prompts are in Portuguese unless marked (EN). Placeholders are written `{name}`.

How prompts are assembled:

- **v1:** `utils/system_prompt.py` renders only the sections `your_role`, `members`, `goal`, `rules`, `examples`, `other_notes` and `answer`, under fixed Portuguese headings. Other sections are dropped.
- **v2:** the same loader renders every section in file order (the consolidated ADK prompts), and keeps the agents' results as placeholders to be filled from state.

See `agent_protocol.md` for what is appended at run time.

## Version 1

### v1.1 Chat system prompt (`prompts/system_prompt.yaml`)

- **Role:** legal assistant for insurance law and regulation in Brazil. It answers questions about the CRSNSP decisions supplied as context.
- **Inputs:** none interpolated. The endpoint appends a data dictionary, the top 10 decisions (17 fields each), the conversation history and the question.
- **Constraints:**
  - answer only about the supplied cases, and say when information is not in them;
  - identify appeals by their title;
  - neutral, factual tone; no advice; no personal data;
  - refuse illegal or inappropriate requests;
  - never name data fields;
  - Brazilian Portuguese, third person, no bold or bullets.
- **Examples:** two appropriate and two inappropriate answers.
- **Output:** free text.
- **Excerpt:** «Quando o usuário perguntar sobre informações que não estão nos casos, informe que você não tem essas informações.»
  - EN: "When the user asks about information that is not in the cases, say that you do not have this information."

### v1.2 Defense structure interpreter (`prompts/defense_structure_interpreter_prompt.yaml`)

- **Role:** interprets the structure of administrative defenses.
- **Inputs:** `{defense_template}`. The code also appends the template again, and the message history.
- **Constraints:**
  - follow the output format of the example;
  - list the section names and the expected content of each, leaving out the facts of any concrete case in the template;
  - plain string, not JSON; no bold or bullets.
  - Note: the prompt refers to an example that it does not contain.
- **Output:** list of sections with their content.
- **Excerpt:** «A sua proposta de estrutura deve conter apenas os nomes das seções e o conteúdo que deve existir em cada uma…»
  - EN: "Your proposed structure must contain only the names of the sections and the content each should have…"

### v1.3 Case interpreter (`prompts/case_interpreter_prompt.yaml`)

- **Role:** analyses the user's case and assesses similar decisions as possible precedents.
- **Inputs:** `{case_description}`, `{case_file_content}`, `{context_cases}`; the message history.
- **Constraints:**
  - describe the case in detail (date, place, alleged infraction, parties);
  - point out applicable precedents among the similar cases;
  - list the legislation cited in the case and in the precedents;
  - third person, no bold or bullets.
- **Output:** a report in two parts: case description, then legal precedents.
- **Excerpt:** «Seu relatório deve ser estruturado em duas partes: descrição do caso e precedentes jurídicos.»
  - EN: "Your report must be structured in two parts: description of the case and legal precedents."

### v1.4 Legal searcher (`prompts/legal_searcher_prompt.yaml`, plus an inline query prompt in `agents/LegalSearcherAgent.py`)

- **Role:** legislation researcher with a case summary available.
- **Inputs:** none interpolated. The code adds the Tavily results and the message history.
- **Constraints:**
  - give the text of relevant articles, items and paragraphs, not whole laws;
  - consider the legislation cited in the case and the date of the facts;
  - do not interpret the case or write the defense;
  - objective, no explanations.
- **Inline query prompt (EN):** asks for up to 3 search queries on Brazilian law, as a JSON array, from the last message.
- **Output:** extracts of legislation.
- **Excerpt:** «Considere em sua pesquisa a data do(s) fato(s), se houver, para determinar a legislação válida no período.»
  - EN: "In your research, consider the date of the fact(s), if any, to determine the legislation in force at the time."
- **Excerpt (EN, inline):** «Extract up to 3 clear search queries related to Brazilian law»

### v1.5 Defense creator (`prompts/defense_creator_prompt.yaml`)

- **Role:** writes administrative defenses to be judged by the CRSNSP, with access to similar decisions and legislation.
- **Inputs:** none interpolated. The code adds the message history and the case description and file content.
- **Constraints:**
  - follow the supplied structure section by section;
  - include the case's context and quotations of legislation;
  - formal style;
  - facts must be the user's case, never the precedents';
  - use placeholders when information is missing;
  - use the precedents as supporting references.
- **Output:** only the defense, in Markdown.
- **Excerpt:** «Se não tiver informações suficientes, como nomes, datas, locais, etc., utilize placeholders '<NOME>', '<DATA>', '<LOCAL>'.»
  - EN: "If you do not have enough information, such as names, dates, places, etc., use placeholders '<NAME>', '<DATE>', '<PLACE>'."

### v1.6 Supervisor review (`prompts/supervisor_review_prompt.yaml`)

- **Role:** senior legal supervisor who reviews, refines and, if needed, enriches the draft.
- **Inputs:** none interpolated. The code adds the whole message history as a transcript and a closing task line.
- **Constraints (rendered):**
  - check the draft against the case analysis, the legislation found and the structure;
  - fix language and errors;
  - polish lightly when the draft is sound, rewrite when it has major flaws;
  - justify arguments with facts from the context;
  - consider the date of the facts;
  - add nothing that the history does not support.
- **Not rendered by v1's loader:** the sections describing the input and the output format (answer only with the final defense).
- **Output:** final defense (in practice).
- **Excerpt:** «Não introduza informações completamente novas que não possam ser inferidas do histórico da conversa.»
  - EN: "Do not introduce completely new information that cannot be inferred from the conversation history."

### v1.7 Routing supervisor (`prompts/supervisor_system_prompt.yaml`): not used

A LangGraph supervisor that would choose the next agent and answer FINISH or REPEAT. The fixed-sequence graph never loads it.

- **Excerpt:** «Convoque cada assistente uma vez e apenas uma vez.»
  - EN: "Call each assistant once and only once."

### v1.8 OCR (inline in `utils/file_parsers.py`, EN; same text in v2)

Asks the model to extract all the text of a non-searchable PDF or of an image.

- **Excerpt:** «Please extract and return all the text content from this document.»

## Version 2

### v2.1 Chat system prompt (`prompts/system_prompt.yaml`)

v1.1 plus one rule: questions about defense strategies are answered only from the cases, saying whether the strategies worked.

- **Excerpt:** «Quando o usuário perguntar sobre estratégias de defesa, responda apenas com o que está nos casos e diga se as estratégias foram eficazes ou não.»
  - EN: "When the user asks about defense strategies, answer only with what is in the cases and say whether the strategies were effective."

### v2.2 Chat case summary (`prompts/chat_case_summary.yaml`; evaluation-only `hierarchical_summary` mode)

- **Role:** CRSNSP legal assistant.
- **Inputs:** `{case_text}`, one decision's 17 fields.
- **Constraints:**
  - at most 300 tokens, in Portuguese;
  - keep parties, rapporteur, session, infraction and cited provisions, penalty, main arguments, grounds of the vote and outcome exactly;
  - add nothing; omit what is absent.
- **Output:** summary.
- **Excerpt:** «Não acrescente informações que não estejam no texto.»
  - EN: "Do not add information that is not in the text."

### v2.3 SupervisorAgent (`prompts/supervisor_consolidated.yaml`)

- **Role:** the user-facing "Assistente de Defesa". It checks that the case information suffices before generation starts.
- **Inputs:** `{case_description}`, `{defense_template}`, `{context_cases}`, `{case_file_content}`.
- **Constraints:**
  - a checklist of essential information (case, parties, process number, object of the appeal, arguments) and desirable information;
  - when to ask, how to ask (at most 2–3 questions), when to proceed;
  - it is the only agent that talks to the user.
- **Output:** JSON, either `{"status": "clarification_needed", "questions": [...]}` or `{"status": "ready", "summary": "..."}`, with one example of each.
- **Excerpt:** «Faça no máximo 2-3 perguntas por vez para não sobrecarregar»
  - EN: "Ask at most 2–3 questions at a time so as not to overload [the user]."

### v2.4 CaseInterpreter (`prompts/case_interpreter_consolidated.yaml`)

- **Role:** as v1.3.
- **Inputs:** `{case_description}`, `{case_file_content}`, `{context_cases}`.
- **Constraints:** v1.3's rules, plus:
  - say that the case is needed if neither description nor file is given;
  - do not mix the user's case with the similar cases;
  - mark which information comes from which;
  - Markdown with headings.
- **Output:** Markdown template:
  - the case (parties, date and place, alleged infraction, legislation cited);
  - per precedent: situation, decision, legal basis, applicability;
  - legislation to research.
- **Excerpt:** «Não confunda a descrição do caso do usuário com os casos similares apresentados.»
  - EN: "Do not confuse the description of the user's case with the similar cases presented."

### v2.5 LegalSearcher (`prompts/legal_searcher_consolidated.yaml`; Google Search grounding)

- **Role:** legislation researcher with an online search tool.
- **Inputs:** `{case_description}`, `{context_cases}`. It does **not** receive `{case_file_content}` (see `agent_protocol.md`).
- **Constraints:**
  - v1.4's research rules;
  - a methodology: identify statutes, search online, structure the results;
  - search priorities: insurance laws, SUSEP/CNSP rules, Civil Code, Consumer Code, CRSNSP case law;
  - check validity at the date of the facts;
  - include source links; no legal interpretation.
- **Output:** Markdown template: legislation identified; relevant legal texts by law and article (quoted in code blocks, with applicability); complementary regulations; synthesis.
- **Excerpt:** «Use a ferramenta de busca disponível para encontrar o texto completo das legislações identificadas.»
  - EN: "Use the available search tool to find the full text of the legislation identified."

### v2.6 DefenseStructureInterpreter (`prompts/defense_structure_consolidated.yaml`)

- **Role:** structuring of CRSNSP administrative defenses.
- **Inputs:** `{defense_template}`.
- **Constraints:**
  - analyse the template's formal structure, check compliance with CRSNSP procedure, find gaps, and give filling guidelines per section;
  - use the default structure when there is no template;
  - quality standards; Markdown.
- **Output:** Markdown: sections found, compliance, gaps, guidelines.
- **Default structure:** six sections, from header to closing; also used by the `no_template` ablation.
- **Excerpt:** «Se o template não estiver disponível, utilize o template padrão fornecido em default_template.»
  - EN: "If the template is not available, use the default template provided in default_template."

### v2.7 DefenseCreator (`prompts/defense_creator_consolidated.yaml`)

- **Role:** experienced writer of CRSNSP administrative defenses.
- **Inputs:** `{case_description}`, `{defense_template}`, `{context_cases}`, `{case_analysis_result}`, `{structure_analysis_result}`, `{legal_search_result}`.
- **Constraints:**
  - rules on document quality, legal integration, formatting (placeholders when data is missing) and consistency with the user's case;
  - integration guidelines per research agent; quality checks;
  - **case-facts rule (v2.2):** process and document numbers, names, CPF/CNPJ, addresses, dates and fine amounts only from the user's case, never invented and never copied from precedents; otherwise keep the template's marker or write `[A PREENCHER: …]`.
- **Output:** the defense in Markdown, following the interpreted structure.
- **Excerpt:** «Nunca invente esses dados e nunca os copie dos casos similares (precedentes).»
  - EN: "Never invent these data and never copy them from the similar cases (precedents)."

### v2.8 MergerAgent (`prompts/merger_consolidated.yaml`)

- **Role:** senior insurance lawyer and final reviewer, who produces the definitive defense.
- **Inputs:** `{case_description}`, `{defense_template}`, `{context_cases}`. The four agent results are named in the text but reach the agent only through the conversation history.
- **Constraints:**
  - integration priorities: the creator's draft as the base, then the legal research, the case analysis, the structure;
  - quality-assurance steps; how to handle missing information (placeholders) and conflicting precedents;
  - the same case-facts rule as v2.7.
- **Output:** only the final defense, in Markdown, with a formal header and a closing with date, lawyer and OAB number.
- **Excerpt:** «Use a minuta do Defense Creator como estrutura base principal»
  - EN: "Use the Defense Creator's draft as the main base structure."

### v2.9 Single-prompt ablation (`prompts/single_prompt_defense.yaml`)

- **Role:** a wrapper for the `single_prompt` ablation.
- **Contents:** a header, then the five generation agents' rendered prompts as numbered tasks, then a footer. Each `{…_result}` placeholder is replaced by a reference to the task that would produce it.
- **Constraints:** do tasks 1–4 internally; write only the result of task 5.
- **Output:** the final defense.
- **Excerpt:** «Realize mentalmente as tarefas 1 a 4, usando o resultado de cada uma nas seguintes, e escreva apenas o resultado final da tarefa 5.»
  - EN: "Carry out tasks 1 to 4 mentally, using each result in the following ones, and write only the final result of task 5."

## Data pipeline prompts

### Compression (EN; `prompts/compress_legal_text_prompt.yaml` in backend and scrapers, identical; v2 only)

- **Role:** a legal expert who compresses text so that it can be reconstructed.
- **Inputs:** `{text}`.
- **Constraints, six conditions:**
  - remove only unimportant words; keep word order; do not change words; add nothing;
  - remove numbering, URLs, names, signature metadata, case numbers, companies, dates, amounts and IDs;
  - keep citations to legislation;
  - compress aggressively.
- **Output:** compressed text only. It is applied to the corpus reports and to uploaded cases.
- **Excerpt:** «Do not remove citations to legislation.»

### Extraction (`crsnsp_scraper/prompts/extraction_prompt.yaml`; v2 corpus; its texts are used by both versions' evaluations, ground rule 9)

- **Role:** specialist in CRSNSP decisions who extracts structured fields.
- **Inputs:** `{existing_names}` (names already in the corpus), `{document_text}`.
- **Constraints:**
  - names complete and spelled as in the document, reusing a known spelling when one exists;
  - outcome from a closed list of six labels ("Desconhecido" only when undeterminable);
  - rules for text fields, empty fields and JSON-safe formatting.
- **Output:** one JSON object.
- **Excerpt:** «Se um nome já existe na lista de "NOMES CONHECIDOS" abaixo, use a MESMA GRAFIA»
  - EN: "If a name already exists in the list of 'KNOWN NAMES' below, use the SAME SPELLING."

### v1 curation (scrapers `data_pipeline/prompts/*.yml`, backend `utils/curateNames.py`)

These produced v1's `*_curated` fields; which of the two produced the v1 snapshot is an open question (reproducibility table).

- Prompts for name normalisation (batch, single name, uniqueness), outcome labelling (batch, single) and inference of missing fields (lawyer, appellant, respondent, rapporteur, generic).
- The GPT-4o script asks for a JSON map from each original name to a curated name.

They are not part of the evaluated systems' runtime, so they are listed but not specified further.
