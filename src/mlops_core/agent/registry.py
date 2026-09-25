"""The agent's prompts in MLflow's prompt registry.

The templates live in the code, versioned by git; the registry is where each trace says
which one it ran. A template is registered when its content changes - never twice for
the same content - as `mlops-agent-<name>`, tagged with the content's hash and written
with MLflow's `{{variable}}` placeholders. The agent formats its prompts from the code;
the registry is the record, so a trace always points at the exact words sent.
"""

import hashlib
import re

import mlflow

from mlops_core.agent.prompts import PROMPTS, version

PREFIX = "mlops-agent-"


def register_prompts(tracking_uri: str) -> dict[str, str]:
    """Register every prompt whose content changed; return each one's registry URI."""
    mlflow.set_tracking_uri(tracking_uri)
    uris = {}
    for name, (template, reply) in PROMPTS.items():
        digest = (
            version(template, reply) if reply else hashlib.sha256(template.encode()).hexdigest()[:8]
        )
        registered = mlflow.genai.load_prompt(PREFIX + name, allow_missing=True)
        if registered is None or registered.tags.get("sha") != digest:
            registered = mlflow.genai.register_prompt(
                PREFIX + name,
                re.sub(r"\{(\w+)\}", r"{{\1}}", template),
                commit_message=f"content {digest}",
                tags={"sha": digest},
                response_format=reply,
            )
        uris[name] = registered.uri
    return uris
