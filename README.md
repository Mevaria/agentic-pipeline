# Agentic pipeline

A multi-agent development pipeline that turns a plain-English change request, a bug report or a labelled GitHub issue into a reviewed pull request on a target repository, using a small local model. Four stages run in order: a spec and test author writes the tests first, an implementer makes them pass, a reviewer judges the change against the target's conventions, and a reporter opens the pull request. Everything that needs no judgement, from branching to the final verdict, is done by code rather than by a model. Nothing is ever merged by the system and nothing is pushed to the base branch.

The design, each agent's rules, the safety rules in the servers, memory, the A2A reviewer and the full settings table are in [docs/design.md](docs/design.md).

Contents: [How a run works](#how-a-run-works) · [Quick start](#quick-start) · [Prerequisites](#prerequisites) · [Setting up a target](#setting-up-a-target) · [Setup](#setup) · [Usage](#usage) · [What a run leaves behind](#what-a-run-leaves-behind) · [Tests](#tests) · [Known limitations](#known-limitations)

## How a run works

| Stage | Done by | What happens |
|---|---|---|
| preflight | code | The target must be clean; the base branch is checked out |
| recall | code | With memory on, similar past runs are shown to the author; an open pull request for a near-identical request is flagged |
| author | model + code | Reads the code, writes a scoped spec and tests that must fail before any change. Code checks that on the base branch and sends any test that passes or breaks back to the author; or the author asks one clarifying question and the run stops |
| implementer | model + code | On a new branch, edits code until the tests pass (ReAct loop, Reflexion between attempts, a retry cap). Code runs the deciding tests, restores any test the model touched, formats, and commits |
| review gate | model + code | Code gathers the diff, the changed files, a fresh test run, the security scan and the checklist; the model judges scope, input validation, secrets, information exposure and whether the tests match the spec; code derives the verdict. Blocking findings go back to the implementer, up to a cap |
| reporter | code + one model call | Pushes the branch, opens the pull request with a templated body and a model-written summary paragraph, notifies you, records the run |
| cleanup | code | Always: uncommitted changes are saved as a patch, the target returns to the base branch with a clean tree |

A request can also come from a GitHub issue the repository owner labelled `agent-ready`; its text enters as quoted data and the pull request links back to the issue. After the pull request exists, a `/revise` comment from the owner sends the feedback back through the author and the implementer on the same branch; closing with `/revise` restarts on a fresh branch; merging is recorded as acceptance.

## Quick start

With the prerequisites installed and a target of your own (both below):

```
ollama pull gemma4:e4b
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then set the three required values
python -m pipeline.smoke_test --no-model
python -m pipeline.run_pipeline "Add a route DELETE /api/books/<id> that removes the book and returns 204."
```

The run ends with a pull request link on your copy of the target, or with a question or a blocked outcome, and the target is left clean on its base branch.

## Prerequisites

| Requirement | Why |
|---|---|
| Python 3.14 | Everything was verified on 3.14 on Windows, where the frozen Chroma environment was produced. Only the early server tests also ran on 3.12 on Linux |
| Node.js | Runs the reference Filesystem MCP server through `npx` |
| Ollama 0.20.0 or later, with `gemma4:e4b` pulled | Serves the local model; Gemma 4 needs this version or newer |
| Git, signed in to GitHub | Branches are pushed with your git sign-in; no token is needed for pushing |
| GitHub MCP server binary | Download your platform's archive from the official releases, verify it against the published checksums, unzip it outside the repositories. No Docker or Go is needed |
| A fine-grained GitHub token | Scoped to your copy of the target only: Pull requests read and write, Issues read, Metadata read. No Contents permission, no Issues write |

## Setting up a target

The pipeline needs a GitHub repository it may push branches to and open pull requests on. You cannot push to the original target, so make your own copy of [github.com/Mevaria/reading-list](https://github.com/Mevaria/reading-list), a small Flask app with a pytest suite and a `CONVENTIONS.md` holding the review checklist:

1. Fork it on GitHub, or create a new repository and push a copy of its contents.
2. Clone your copy locally; that clone is `TARGET_REPO_PATH`.
3. Create a fine-grained token scoped to your copy with the permissions above.
4. Optional, to mirror the original: a ruleset on `main` that forbids deletion and force pushes and requires every change to go through a pull request. The pipeline never touches `main` either way.

Any other repository works if it has `pytest.ini`, a `tests` folder, a `CONVENTIONS.md` with a review checklist, and an origin on GitHub.

## Setup

1. Pull the model: `ollama pull gemma4:e4b`.

2. Create one virtual environment. The custom server runs the target's tests with this interpreter, so the target's dependencies are in `requirements.txt` too:

   ```
   python -m venv .venv
   .venv\Scripts\activate            # Windows
   source .venv/bin/activate         # macOS and Linux
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set the three required values: `TARGET_REPO_PATH`, your clone of the target; `GITHUB_MCP_SERVER`, the path to the binary; `GITHUB_PERSONAL_ACCESS_TOKEN`, the token, scrubbed from every log and record. Everything else has a working default; the full table is in [docs/design.md](docs/design.md#settings).

4. Download the Filesystem server once, so its first launch does not time out. Wait for "running on stdio", then press Ctrl+C:

   ```
   npx -y @modelcontextprotocol/server-filesystem C:\path\to\target\repo
   ```

5. Check that the servers connect and that the model can call tools:

   ```
   python -m pipeline.smoke_test --no-model   # servers only
   python -m pipeline.smoke_test              # servers and model, three small tasks
   ```

6. Optional, memory. The Chroma MCP server needs its own virtual environment, installed from the frozen file with `--no-deps` (the design document says why):

   ```
   python -m venv .venv-chroma
   .venv-chroma\Scripts\pip install --no-deps -r requirements-chroma.txt     # Windows
   .venv-chroma/bin/pip install --no-deps -r requirements-chroma.txt         # macOS and Linux
   ```

   On macOS and Linux also set `CHROMA_MCP_SERVER=.venv-chroma/bin/chroma-mcp` in `.env`. The first run downloads the embedding model once, 79 MB.

7. Optional, the reviewer as an A2A service: run `python -m pipeline.serve_reviewer` in a second terminal and set `REVIEWER_A2A_URL=http://127.0.0.1:9999` in `.env`. Every review then goes through the service, and the gate fails closed if it cannot be reached.

## Usage

**One request to one pull request:**

```
python -m pipeline.run_pipeline "Add a route DELETE /api/books/<id> that removes the book and returns 204."
python -m pipeline.run_pipeline --file request.txt
```

A run prints each stage and ends `READY_TO_REVIEW` with the pull request link, `NEEDS_CLARIFICATION` with the author's question, `BLOCKED` when a stage hit its cap, or `CRASHED` with the stage and a one-line error. Either way the target is left clean on its base branch, with one record and one tool log under `runs/records`.

**Labelled issues.** Apply the `agent-ready` label to an open issue on your copy; only triage access can, so the label is the owner's approval. The issue text reaches the author as quoted data, never as instructions, and the pull request body says `Fixes #N`:

```
python -m pipeline.run_issues              # the oldest labelled issue without a run record
python -m pipeline.run_issues --all        # every such issue, one run each
python -m pipeline.run_issues --issue 5    # issue 5 again: your explicit re-approval after re-labelling it
```

**Feedback on a pull request.** Post a comment starting with `/revise`, then:

```
python -m pipeline.run_feedback <pull request number>
```

An open pull request is revised on its branch and a note is posted on it; a closed one restarts on a fresh branch; a merged one is recorded as accepted. Each stage also has its own runner (`run_author`, `run_implementer`, `run_review`, `run_report`), described in the design document.

## What a run leaves behind

Everything under `runs/` is gitignored run output: `records/` with one JSON record and one tool log per run, `tasks/` with each run's generated spec and tests, `notifications.log` with one line per outcome, and `chroma/` with the memory database. Records patched by hand carry a `manual_corrections` field saying what changed and why, so evidence never looks like the pipeline produced something it did not.

`python -m scripts.run_metrics` reads every record and tool log and prints one table per run of trajectory metrics measured from the files: tool calls in total and by who asked for them, failed and refused calls, revisions, attempts, review rounds, fail-first checks, outcome and duration.

## Tests

Tests cover the dev tools server, the three agent graphs, the review gate, the reporter, the orchestrator end to end, the feedback loop, the issue intake, memory, the A2A reviewer and the metrics script, plus a check that every module, class and function has a docstring. The graph tests use scripted stand-ins for the model; the MCP servers, git, pytest, Bandit and ruff are real, on temporary repositories. Tests needing the GitHub binary and token or the Chroma environment skip, with a message, where those are missing.

```
pytest
```

`python -m scripts.check_docstrings` lists missing docstrings without running the rest of the suite.

## Known limitations

- **The model is not deterministic at temperature 0.** The same finding was blocking in one run and non-blocking in the next.
- **Memory is near-duplicate detection.** A loose rewording can be further from its own run than a different feature on the same app, so only clear matches are shown.
- **The fail-first check proves sensitivity, not correctness.** In one live run the author bent a not-found test from 404 to 500 to make it red and the implementer satisfied it; the author's guidance and the reviewer's `tests_match_spec` item now address this. Full account in the design document.
- **Line numbers from the model are approximate.** They are kept only when the diff added that line.
- **Pull requests are opened under the developer's identity**, so GitHub disables approving them; the `/revise` comment stands in.
- **An issue can change between the owner's label and the pipeline's first read.** No GitHub server tool exposes label or edit times; the record stores the exact text acted on, and the owner reviews the pull request before any merge.
- **Ollama can crash on a cold model load when host memory is nearly exhausted.** Warm the model with any small request first.
- **The A2A service has no authentication** and listens on localhost only.
- **Windows paths** are used in the examples; the code is path-agnostic, but only the early server tests ran outside Windows.
