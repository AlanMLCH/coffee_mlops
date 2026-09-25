"""The agent and its three tools: SQL over the layers, the prediction API, and the
retriever over the corpus.

Each tool is a plain function with a pydantic schema first - measured, then wrapped for
LangGraph and, last, for MCP - and each keeps its own guardrails, whoever calls it.
"""
