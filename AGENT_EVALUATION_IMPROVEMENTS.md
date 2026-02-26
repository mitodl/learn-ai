# AI Agent Architecture & Evaluation Improvement Plan

This document summarizes the current AI agent architecture and proposes a set of actionable steps to enhance the evaluation framework. These steps are designed to be easily converted into GitHub issues.

## AI Agent Architecture Overview

The application employs a sophisticated multi-agent architecture built on Django, `langchain`, and `langgraph`. This modular design supports multiple, specialized AI chatbots, each tailored to a specific task.

- **Core Framework**: The system uses `langgraph` to define agents as stateful graphs, enabling complex reasoning loops. The use of `litellm` provides a flexible backend to connect with various LLM providers.
- **Agent Types**: The primary agents are "ReAct" (Reasoning and Acting) style agents that perform Retrieval-Augmented Generation (RAG):
  - `ResourceRecommendationBot`: Searches for learning resources.
  - `SyllabusBot`: Generates course syllabi from content files.
  - `VideoGPTBot`: Engages in Q&A over video transcripts.
  - `TutorBot`: A specialized agent for assisting with educational problem sets.
- **State and Memory**: Conversational state is managed by `langgraph` and persisted to a PostgreSQL database via a custom `DjangoCheckpoint`, providing agents with long-term memory.
- **Tools and RAG**: Agents use a variety of tools to fetch contextual data from vector stores (`chromadb`), APIs, and content files before generating a response.

---

## Recommendations for Evaluation and Accuracy Checks

The following items outline concrete steps to build upon the existing evaluation framework in `ai_chatbots/evaluation/`.

### 1. Integrate Advanced RAG Metrics

- **Title**: `feat(evaluation): Integrate Advanced RAG Metrics into Evaluation Suite`
- **Description**: The current evaluation framework relies on comparing against an `expected_output`, which can be brittle. To get a more nuanced understanding of agent performance, we should integrate advanced RAG metrics using the `deepeval` library that is already a project dependency. This will allow us to measure the distinct components of our RAG pipeline (retrieval, generation) independently.
- **Acceptance Criteria**:
  - [ ] Update the evaluation orchestrator (`ai_chatbots/evaluation/orchestrator.py`) to compute and record scores for the following metrics:
    - `Faithfulness`: Does the generated answer contradict the retrieved context?
    - `Answer Relevancy`: Is the answer relevant to the user's query?
    - `Contextual Precision`: How relevant is the retrieved context to the query?
    - `Contextual Recall`: Was all the necessary information retrieved?
  - [ ] Update the reporting module (`ai_chatbots/evaluation/reporting.py`) to display these new scores in the evaluation summary.
  - [ ] Ensure `expected_tools` from test cases are also validated to measure tool-calling accuracy.

### 2. Establish a Version-Controlled "Golden Dataset"

- **Title**: `chore(evaluation): Establish a Version-Controlled Golden Dataset`
- **Description**: Our test cases are currently loaded from various JSON files. To improve rigor and maintainability, we should consolidate these into a formal, version-controlled "golden dataset." This dataset will serve as the single source of truth for benchmarking agent performance and tracking regressions.
- **Acceptance Criteria**:
  - [ ] Create a new top-level directory, e.g., `evaluation_datasets/`.
  - [ ] Move existing test case JSON files into this new directory, organizing them by agent (e.g., `evaluation_datasets/recommendation/tests.json`).
  - [ ] Refactor the `load_test_cases` methods in `ai_chatbots/evaluation/evaluators.py` to load data from this new, centralized location.
  - [ ] Add a section to `README.md` or a `CONTRIBUTING.md` file documenting the structure of the golden dataset and the process for adding new, high-quality test cases.

### 3. Automate Evaluations in CI/CD

- **Title**: `ci: Automate Agent Evaluation Suite in GitHub Actions`
- **Description**: To catch regressions and monitor performance continuously, the agent evaluation suite should be automated and integrated into our CI/CD pipeline. This will provide rapid feedback on how code changes, prompt updates, or new models affect agent behavior.
- **Acceptance Criteria**:
  - [ ] Create a new GitHub Actions workflow (e.g., `.github/workflows/evaluate_agents.yml`).
  - [ ] Configure the workflow to run a small, critical subset of the golden dataset on pull requests targeting the `main` branch.
  - [ ] Configure the workflow to run the _full_ evaluation suite on a nightly schedule.
  - [ ] Ensure the evaluation orchestrator can be triggered via a command-line script (e.g., a Django management command).
  - [ ] The workflow should publish the evaluation report as a build artifact for analysis.

### 4. Implement a Human-in-the-Loop (HITL) Feedback System

- **Title**: `feat(chat): Implement User Feedback Mechanism (HITL)`
- **Description**: The ultimate measure of agent quality is user satisfaction. We should implement a simple feedback mechanism in the UI to capture real-world performance data. This data is invaluable for identifying failure modes and improving our evaluation datasets.
- **Acceptance Criteria**:
  - [ ] Create a new Django model to store user feedback, linking it to the conversation `thread_id` and `checkpoint_pk`. The model should store a rating (e.g., positive/negative) and an optional text comment.
  - [ ] Add a new API endpoint to receive and persist this feedback from the frontend.
  - [ ] Update the `frontend-demo` application to include UI elements (e.g., thumbs up/down buttons) on each agent message for submitting feedback.
  - [ ] Create a Django admin view or management command to easily review feedback and identify poor-quality conversations that should be turned into new test cases for the golden dataset.

### 5. Document the Process for Prompt and Model Evolution

- **Title**: `docs(engineering): Document Process for Prompt and Model Evaluation`
- **Description**: To ensure the new evaluation tools are used effectively, we must document the process for systematically testing changes to prompts and benchmarking new LLMs. This will help the team make data-driven decisions.
- **Acceptance Criteria**:
  - [ ] Create a new markdown document (e.g., `docs/ai_evaluation_process.md`).
  - [ ] The document must outline the step-by-step process for testing a new prompt template against the golden dataset.
  - [ ] The document must describe how to benchmark a new LLM (e.g., a new model from OpenAI or Anthropic via `litellm`) against the current one using the evaluation suite.
  - [ ] Link to this new document from the main project `README.md`.
