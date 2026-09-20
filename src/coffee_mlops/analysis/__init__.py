"""Analysis pipeline: the layers turned into tables, figures and a dashboard.

Reads what the other pipelines leave on disk and never writes back into them. Like the
CLI and the orchestrator, it may know about the other packages; they must not know it.
"""
