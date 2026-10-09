"""Serve the reviewer over A2A on localhost.

    python -m pipeline.serve_reviewer                # 127.0.0.1:9999
    python -m pipeline.serve_reviewer --port 9000

Set REVIEWER_A2A_URL to this address in the pipeline's .env, and the review
gate sends each change's evidence here instead of reviewing in-process. The
service uses its own model connection and has no access to any repository.
"""

import argparse

import uvicorn

from pipeline.a2a_reviewer import build_app
from pipeline.config import get_context_size, get_model_name, get_review_settings


def main(host, port):
    """Build the application with the real model and serve it."""
    # Imported here so the servers-only code paths do not need Ollama's client installed.
    from langchain_ollama import ChatOllama

    url = f"http://{host}:{port}"
    model = ChatOllama(model=get_model_name(), temperature=0, num_ctx=get_context_size())
    app = build_app(model, get_review_settings(), url)
    print(f"Reviewer service on {url}, model {get_model_name()}; agent card at {url}/.well-known/agent-card.json")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    # The first line of the module docstring doubles as the command's help text.
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind; localhost only by default")
    parser.add_argument("--port", type=int, default=9999, help="port to listen on")
    arguments = parser.parse_args()
    main(arguments.host, arguments.port)
