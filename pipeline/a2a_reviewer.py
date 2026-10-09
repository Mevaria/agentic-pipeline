"""The reviewer over A2A: a service that judges evidence it is sent, and the client the gate uses to reach it.

The pipeline keeps gathering the evidence itself, since that needs the MCP
servers and the target checkout. What crosses the A2A boundary is the pure
part of a review: the evidence goes out as one JSON data part, the model pass
and the verdict rules run on the service, and the model's findings and
summary come back as one JSON artifact. The pipeline then adds the findings
code decides, scan, formatting and tests, and derives approve or request
changes exactly as it does for a local review.

The service binds to localhost with no authentication; it is a local
demonstration of the protocol, not a deployment. It never touches a
repository and has no tool beyond submit_review.
"""

import inspect

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_artifact, new_data_message, new_task_from_user_message, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role, SendMessageRequest, TaskState
from starlette.applications import Starlette

from pipeline.reviewer import build_reviewer, recursion_limit
from pipeline.tooling import innermost_error, scrub_secrets

# The one skill the service advertises; the client refuses a card without it.
SKILL_ID = "review_change"
# The evidence keys a request must carry, the same ones gather_evidence produces.
EVIDENCE_KEYS = ("task", "branch", "base_branch", "diff", "changed_files", "test_record", "scan", "format_report", "checklist")
# Name of the artifact that carries the verdict back.
VERDICT_ARTIFACT = "verdict"


def agent_card(url):
    """Return the agent card the service serves: one skill that takes evidence as JSON and returns a verdict as JSON."""
    skill = AgentSkill(
        id=SKILL_ID,
        name="Review a change",
        description=(
            "Judges one change against a review checklist. Send one JSON data part with the evidence: "
            f"{', '.join(EVIDENCE_KEYS)}. Returns one JSON artifact named 'verdict' with the model's findings "
            "(item, severity, file, line, message), a summary, and no_verdict. Scope, input validation, secrets "
            "and information exposure are judged; tests, scans and formatting are not."
        ),
        tags=["code-review", "security"],
        input_modes=["application/json"],
        output_modes=["application/json"],
    )
    return AgentCard(
        name="Agentic pipeline reviewer",
        description="Security and scope reviewer of the agentic pipeline, judging evidence gathered by the pipeline.",
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")],
    )


class ReviewerExecutor(AgentExecutor):
    """Runs the reviewer graph over the evidence in a request and returns the model's verdict as an artifact."""

    def __init__(self, model, settings, log=print):
        """Keep the model and the review settings; the graph is built per request so each is independent."""
        self.model = model
        self.settings = settings
        self.log = log

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Validate the evidence, run the review, and finish the task with the verdict artifact or a failure."""
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        data_parts = get_data_parts(context.message.parts) if context.message else []
        evidence = data_parts[0] if data_parts and isinstance(data_parts[0], dict) else None
        missing = [key for key in EVIDENCE_KEYS if evidence is None or key not in evidence]
        if missing:
            await updater.update_status(state=TaskState.TASK_STATE_FAILED,
                                        message=new_text_message(f"The request must carry evidence with: {', '.join(missing)}"))
            return
        await updater.update_status(state=TaskState.TASK_STATE_WORKING, message=new_text_message("Reviewing the change"))
        try:
            # No tools: the service has no repository, so the graph starts from the evidence it was given.
            graph = build_reviewer(None, [], self.model, self.settings, self.log, provided_evidence=True)
            state = await graph.ainvoke({"evidence": evidence, "branch": evidence.get("branch", "")},
                                        {"recursion_limit": recursion_limit(self.settings)})
        except Exception as error:
            await updater.update_status(state=TaskState.TASK_STATE_FAILED,
                                        message=new_text_message(f"The review failed: {scrub_secrets(innermost_error(error))[:300]}"))
            return
        verdict = state["model_verdict"]
        self.log(f"[a2a] verdict for {evidence.get('branch')}: {len(verdict['findings'])} finding(s), no_verdict={verdict['no_verdict']}")
        await updater.add_artifact(name=VERDICT_ARTIFACT, parts=[new_data_artifact(VERDICT_ARTIFACT, verdict, media_type="application/json").parts[0]])
        await updater.update_status(state=TaskState.TASK_STATE_COMPLETED, message=new_text_message("Review complete"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Reviews are short and not cancellable."""
        raise NotImplementedError("Cancel is not supported.")


def build_app(model, settings, url, log=print):
    """Return the Starlette application serving the agent card and the JSON-RPC endpoint."""
    card = agent_card(url)
    handler = DefaultRequestHandler(agent_executor=ReviewerExecutor(model, settings, log), task_store=InMemoryTaskStore(), agent_card=card)
    routes = [*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")]
    return Starlette(routes=routes)


def verdict_from_artifacts(artifacts):
    """Return the verdict dict from the artifacts that came back, or None when none holds a usable verdict."""
    for artifact in artifacts:
        for data in get_data_parts(artifact.parts):
            if isinstance(data, dict) and isinstance(data.get("findings"), list) and "no_verdict" in data:
                findings = []
                for finding in data["findings"]:
                    entry = dict(finding)
                    # JSON numbers travel as floats through the protocol's Struct; a line number must come back as an int.
                    if isinstance(entry.get("line"), float) and entry["line"].is_integer():
                        entry["line"] = int(entry["line"])
                    findings.append(entry)
                return {"findings": findings, "summary": str(data.get("summary", "")),
                        "no_verdict": bool(data["no_verdict"]), "missing_passes": list(data.get("missing_passes", []))}
    return None


async def remote_model_verdict(url, evidence, httpx_client=None, log=print):
    """Send the evidence to the reviewer service and return the model's verdict, failing closed on any problem.

    httpx_client lets a test reach an in-process app; the gate leaves it None.
    """
    failed = {"findings": [], "summary": "", "no_verdict": True, "missing_passes": ["review"]}
    client = None
    try:
        http = httpx_client or httpx.AsyncClient(timeout=600)
        card = await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()
        if not any(skill.id == SKILL_ID for skill in card.skills):
            log(f"[a2a] the agent at {url} does not offer the {SKILL_ID} skill")
            return failed
        # create_client is a coroutine in this SDK release.
        client = await create_client(agent=card, client_config=ClientConfig(streaming=False, httpx_client=http))
        request = SendMessageRequest(message=new_data_message(evidence, media_type="application/json", role=Role.ROLE_USER))
        task = None
        artifacts = []
        state = None
        # The client yields StreamResponse wrappers whose payload is a task, a message, a status update or an
        # artifact update; without streaming the one response carries the finished task, but every form is read.
        async for event in client.send_message(request):
            payload = event.WhichOneof("payload") if hasattr(event, "WhichOneof") else None
            if payload == "task":
                task = event.task
            elif payload == "artifact_update":
                artifacts.append(event.artifact_update.artifact)
            elif payload == "status_update":
                state = event.status_update.status.state
        if task is not None:
            artifacts.extend(task.artifacts)
            state = task.status.state
        verdict = verdict_from_artifacts(artifacts)
        if verdict is None:
            log(f"[a2a] no verdict came back from {url} (task state {state})")
            return failed
        log(f"[a2a] verdict received from {url}: {len(verdict['findings'])} finding(s)")
        return verdict
    except Exception as error:
        log(f"[a2a] the reviewer service at {url} could not be used: {scrub_secrets(innermost_error(error))[:300]}")
        return failed
    finally:
        # close() is synchronous in this SDK release but documented as awaitable elsewhere; both are handled.
        if client is not None:
            closing = client.close()
            if inspect.isawaitable(closing):
                await closing
        if httpx_client is None and "http" in locals():
            await http.aclose()
