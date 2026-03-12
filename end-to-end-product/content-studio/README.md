Content Studio is a multi-agent AI system that converts a content idea into a production-ready short-form video package.

Python: >=3.11
Framework: FastAPI
Orchestration: LangGraph
LLM: Claude / OpenAI
Deployment: Docker + Cloud Run


Model orchestration strategy

Claude Sonnet:
- Director agent
- Script generation

Claude Haiku:
- Research synthesis
- Storyboard generation
- Title generation
- QA validation


The Director agent dynamically plans which specialist agents to execute
based on the user's content request.

LangGraph's Send routing mechanism is used to orchestrate dynamic
agent execution.

The system implements a self-revision loop where generated scripts are evaluated
by a QA agent and automatically improved by a revision agent before final packaging.

START
 ↓
Director (Claude Sonnet)
 ↓
Router
 ↓
Dynamic Agents
 ├ Research
 ├ Script
 ├ Storyboard
 ├ Title
 └ Music
 ↓
QA (Haiku)
 ↓
Revision (Sonnet)
 ↓
Packaging
 ↓
END