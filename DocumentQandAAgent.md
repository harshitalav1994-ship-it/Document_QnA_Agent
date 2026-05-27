# AI-Powered Document Q&A Agent API

## User Story

**As a AI engineer**
I want to submit document text and natural language questions to an API that uses an LLM-powered agent to answer questions about the document content
**So that** I can extract intelligent, context-aware insights from unstructured text 

---

## Acceptance Criteria

### 1. Document Ingestion

- The API accepts document text via a `POST` endpoint in the request body
- Maximum input size is clearly documented and enforced

### 2. Question Answering (Agentic)

- Users can submit natural language questions about the uploaded document text via a `POST` endpoint
- The API uses an **agentic framework** (e.g., LangChain, LangGraph, CrewAI, or similar) to orchestrate retrieval and answer generation
- If needed, the agent can demonstrate tool-use capability (e.g., a retrieval tool the agent decides when to invoke)
- The agent should handle questions that cannot be answered from the provided document gracefully

### 3. Evaluation

- Implement a basic evaluation mechanism (e.g., using Ragas, DeepEval, or similar)
- Evaluate **faithfulness** (is the answer grounded in the retrieved context?) against at least 3 test cases you define
- Runnable via a single command or script

### 4. Observability

- Integrate basic **LLM tracing** (e.g., LangSmith, LangFuse, or callback-based logging)
- Track: request/response content, token usage, and latency per request

### 5. Response Format

- The API returns results in **JSON format**, including:
  - The generated answer
  - The source chunks used to generate the answer

---

## Technical Notes

- Implement in **Python** using RESTful principles (e.g., FastAPI)
- Use any LLM of your choice (document your choice in the README)
- Include **unit tests** for input validation and at least one test with a mocked LLM
- If time allows, **Containerise** with Docker (runnable via `docker compose up` or equivalent)
- Provide a `README.md` with setup instructions, how to run, and brief notes on trade-offs or what you'd improve with more time
- Give due consideration as you would for a production application (error handling, no hardcoded secrets, clean code structure)

---

## What We're Looking For

| Area | Focus |
|------|-------|
| **AI/LLM Integration** | Effective use of an LLM with agentic behaviour and retrieval |
| **Evaluation** | Demonstrates awareness of how to validate LLM output quality |
| **Observability** | Basic tracing/monitoring in place |
| **Software Engineering** | Clean code, error handling, and testing |
| **Deployment** | Containerised and easy to run |

---

*Estimated effort: ~2-3 hours. Focus on demonstrating production-readiness thinking over feature completeness. If you run short on time, document what you would do next in your README.*
