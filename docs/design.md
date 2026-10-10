# Agentic pipeline: design

The design behind the pipeline described in the [README](../README.md): each agent's graph and rules, the MCP servers and their configuration, the safety rules in code, the orchestrator, the feedback loop, the issue intake, memory and the A2A reviewer, the full settings table, and what changed from the submitted design while building.

Contents: [MCP servers](#mcp-servers) · [Tool allowlists](#tool-allowlists) · [Spec and test author](#spec-and-test-author) · [Implementer](#implementer) · [Reviewer and review gate](#reviewer-and-review-gate) · [Reporter](#reporter) · [Orchestrator](#orchestrator) · [Pull request feedback](#pull-request-feedback) · [Issue intake](#issue-intake) · [Memory](#memory) · [Reviewer over A2A](#reviewer-over-a2a) · [Settings](#settings) · [Notes](#notes) · [Design changes from the submitted design](#design-changes-from-the-submitted-design)

The target used throughout is `reading_list`, a small Flask app with a JSON API, one HTML page, a pytest suite and a `CONVENTIONS.md` that holds the coding rules, the git rules and the review checklist. The pipeline is not specific to it: it works on any repository that has `pytest.ini`, a `tests` folder and an origin on GitHub, and reads the review checklist from the target's `CONVENTIONS.md` when that file has a `## Review checklist` section, falling back to a built-in four-item checklist (input validation, secrets, information exposure, scope) when it does not. Everything that needs no judgement, from branching to the final verdict, is done by code rather than by a model. Nothing is ever merged by the system and nothing is pushed to the base branch.

## MCP servers

The pipeline connects to up to five MCP servers over stdio. Launch settings are in `pipeline/config.py`.

| Server | Source | Used for |
|---|---|---|
| Filesystem | `@modelcontextprotocol/server-filesystem` (reference server) | Reading and writing files, restricted to the target repository |
| Git | `mcp-server-git` (reference server) | Status, branches, checkouts and commits, called by pipeline code |
| Dev tools | `servers/dev_tools_server.py` (this project) | Tests, security scan, fail-first check, formatting, task folders, push, restore and notifications |
| GitHub | `github-mcp-server` (official server, release binary) | Pull requests, comments and labelled issues; started only by the runners that need it |
| Chroma | `chroma-mcp` (official server), optional | Run memory; started only when installed |

Each server gets one long-lived session for the whole run rather than the client's default of a new process per tool call, and every server gets the full environment: without it, `npx` took over a minute to start and the Git server could not read the git identity.

### GitHub server

The server is started with `GITHUB_TOOLS=create_pull_request,list_pull_requests,pull_request_read,add_issue_comment,list_issues,issue_read` and no toolsets. Verified live against v2.0.2: given a tools list alone, the server exposes exactly those six tools and does not load its default toolsets, so `merge_pull_request` and `issue_write` are never offered to anything in the pipeline. A tool name the server does not know makes it refuse to start, so a typo in the list fails loudly rather than silently widening it. Naming any toolset as well would add that group on top, which is why `GITHUB_TOOLSETS` is deliberately left unset. `tests/test_github_server.py` starts the real binary from this configuration and asserts the exact tool list; it is skipped on a machine without the binary and token.

### Dev tools server

| Tool | Purpose |
|---|---|
| `run_tests` | Runs the target repository's pytest suite |
| `run_security_scan` | Runs Bandit on the code and pip-audit on `requirements.txt`, and decides whether findings block |
| `check_tests_fail` | Runs newly written tests in a throwaway worktree of `BASE_BRANCH` and reports whether every one fails on an assertion |
| `write_task_spec` | Saves a generated task's type, short description and spec as `task.json` under `TASKS_PATH/<run id>` |
| `write_task_test` | Saves one generated test file into that task folder, refusing a name the target repository's tests folder already has |
| `format_code` | Runs `ruff format` and ruff's whitespace rules on the Python files the change touched outside `tests/`, either fixing them or only reporting |
| `diff_against_base` | The checked-out branch's diff against its merge base with `BASE_BRANCH` |
| `read_file_on_base` | A file's content as it is on `BASE_BRANCH`, whatever the checked-out branch did to it |
| `remote_repository` | The GitHub owner and repository of the origin remote, so pull requests go to the right place |
| `uncommitted_patch` | The working tree's uncommitted changes, new files included, as one patch; kept as evidence before a restore |
| `restore_repository` | Discards uncommitted changes, returns to `BASE_BRANCH`, and deletes the run's branch only if it has no commits. Refuses any path other than the configured target, and never deletes the base or a protected branch |
| `git_push` | Pushes the current branch, refusing protected branches |
| `notify_user` | Records the outcome of a run for the developer: ready to review, blocked, or needs clarification |

Safety rules are enforced in the server code, not in prompts: the repository path comes from configuration, protected branches cannot be pushed or deleted, and the blocking decision is computed from fixed rules.

The scan blocks a change when any of these hold:

- Bandit reports a code finding at or above `BLOCKING_SEVERITY`.
- pip-audit reports a known vulnerability and the change edits `requirements.txt`.
- The scan itself fails, including when the change cannot be compared against `BASE_BRANCH`.

Dependency vulnerabilities in a change that does not touch `requirements.txt` are reported as non-blocking, so unrelated work is not held up by a problem it did not cause.

#### Fail-first check

`check_tests_fail` is how the pipeline confirms that tests written before an implementation are testing the right thing. Each new test gets one of three statuses:

| Status | Meaning |
|---|---|
| red | Failed on an assertion, or on `pytest.raises` reporting that nothing was raised. This is what a correct new test looks like |
| passes | Passed before any change was made, so it does not test the requested behaviour |
| broken | Failed to import, failed in a fixture, was skipped, raised something other than an assertion, or the file defined no tests |

The check reports `all_red` only when every test is red. Classification is by exception type, not by message text: a small pytest plugin in `servers/pytest_outcomes.py` is loaded for that one run and records what each test raised. The tests run in a git worktree of the base branch at `FAIL_FIRST_WORKTREE` (default `runs/fail_first_worktree`), which is removed afterwards even if the check fails, and any worktree left at that path by a crashed run is removed first. The target repository's own checkout is never touched.

The check proves a test is sensitive to the current code, not that its expected values are right. A test asserting the wrong value also fails on an assertion and is reported as red. This happened in a live run: a not-found test passed on the base branch because a route that does not exist yet already answers 404, and the author made it red by changing the expected status to 500, which the implementer then satisfied and the reviewer, judging scope and security only, approved. Two layers now address it: the author's revision guidance says to assert what the spec requires and never to bend an expected value, and the reviewer has a `tests_match_spec` item that blocks a test contradicting the spec. Both depend on the model following them. Measured afterwards: the revision of that pull request wrote a not-found test that was red on the base branch because the JSON body was missing, and a single review round of the original 500 commit with the new item raised `tests_match_spec` as a blocking finding on the test file and blocked the change. The records of those runs are the evidence.

#### Generated task folders

`write_task_spec` and `write_task_test` write a task folder in the same layout as the hand-written ones under `tasks/`, so the implementer can run it unchanged. Generated folders are run output, not source, so they go under `TASKS_PATH` (default `runs/tasks`, which is gitignored), one folder per run named by `RUN_ID`, which the pipeline sets before it starts the servers. The test tool accepts only a bare `test_*.py` name and refuses one that already exists in the target repository's `tests` folder: the implementer copies generated tests into that folder and then protects every file there, so a clash would overwrite an existing suite and protect the overwritten version as if it were original.

## Tool allowlists

Each agent receives only the tools listed for it in `AGENT_TOOLS` in `pipeline/config.py`. Steps that need no judgement, such as creating the branch and committing, are done by pipeline code rather than offered to a model.

| Agent | Tools |
|---|---|
| Spec and test author | `list_directory`, `read_text_file`, `write_task_spec`, `write_task_test`, `request_clarification` |
| Implementer | `list_directory`, `read_text_file`, `write_file`, `run_tests` |
| Reviewer | `submit_review` only |
| Reporter | none; code calls `git_push`, `notify_user`, `remote_repository`, `list_pull_requests`, `create_pull_request` and, after a revision, `add_issue_comment` |
| Issue intake | none; code calls `list_issues` and `issue_read` |

The author can read the target repository but never write to it: its only writes go into its own task folder. The fail-first check is not on its list, because code runs it at a fixed point. `request_clarification` and `submit_review` are defined in the pipeline rather than on a server, since they have no side effect and only signal their graphs; they are logged to the run's tool log like every MCP call. The implementer's list is deliberately short: `directory_tree` flooded the model with `.git` contents, and `edit_file` needs exact text matches a small model fumbles.

## Spec and test author

`pipeline/author.py` is a LangGraph graph that turns a plain-English change request or bug report into a task folder the implementer can run. It is the only agent that takes input. Tests are written before any implementation exists, and a bug report starts with a reproduction test.

| Node | Done by | What it does |
|---|---|---|
| prepare | Code | Refuses a target with uncommitted changes, checks out `BASE_BRANCH` so the author reads the same code the fail-first check runs against, and starts the conversation with the request |
| author | Model | ReAct loop with the author's tools: reads the code and existing tests, then writes the spec and the test files |
| tools | Code | Executes the tool calls the model asked for; an accepted `request_clarification` ends the attempt here, so the model gets no further turn |
| check | Code | Runs the fail-first check on the written tests; a run in which the author called `request_clarification` is sent to clarify instead |
| revise | Code | Starts a fresh attempt with the request, the spec and tests so far, and one line per test saying why it was not red, up to `MAX_AUTHOR_REVISIONS` times. For a test that passed, the guidance says to make it fail by asserting something the spec requires that the current code does not do, such as the JSON error body, and never to change an expected value so that it contradicts the spec |
| finish | Code | Every test is red, so the task folder is ready |
| clarify | Code | Ends the run with the author's question and its reason, so the developer can rerun with a clearer request |
| blocked | Code | The revision cap was reached without every test being red |
| not_reproducible | Code | A `fix` task whose every test still passed at the cap: the reported bug could not be reproduced on the current code |

```
python -m pipeline.run_author "Add a route DELETE /api/books/<id> that removes the book and returns 204."
python -m pipeline.run_author --file request.txt
```

When the request cannot be specified as given, the author calls `request_clarification` with one question and one of five reasons. The attempt ends as soon as the tool accepts the call, anything already written to the task folder is removed so nothing that looks usable is left behind, and the run ends with the reason and question printed. Any other reason is rejected by the tool and the author must choose again. A final message without a spec and without that call is fed back as a failed attempt.

| Reason | Meaning |
|---|---|
| `behaviour_not_testable` | The request gives no observable outcome to test |
| `contradicts_existing` | The request conflicts with existing code or tests |
| `unknown_reference` | The request refers to something that does not exist in the code |
| `too_broad` | The request is too large for one change |
| `duplicate_of_open_pr` | A past run surfaced by memory already implements a near-identical request and its pull request is still open; the author asks whether to revise that pull request instead |

## Implementer

`pipeline/implementer.py` is a LangGraph graph that changes code until the given tests pass, or stops as blocked.

| Node | Done by | What it does |
|---|---|---|
| prepare | Code | Branches from `BASE_BRANCH` as `<type>/<short-description>`, adds the given tests, and records every test file as protected. In continue mode it checks out an existing branch instead |
| agent | Model | ReAct loop with the implementer's four tools, up to `MAX_TOOL_STEPS` tool calls per attempt |
| tools | Code | Executes the tool calls the model asked for |
| verify | Code | Restores any protected test file the model changed, then runs the tests itself |
| reflect | Model | Explains why the attempt failed; the lesson is carried into the next attempt (Reflexion) |
| finish | Code | Formats the changed files with `format_code`, re-runs the tests if anything changed, then commits with a Conventional Commits message the model writes, falling back to a default if it is invalid |
| blocked | Code | Stops after `MAX_ATTEMPTS` failed attempts, leaving the changes uncommitted for the orchestrator to save as a patch |

Branching, the test run that decides pass or fail, formatting and committing are done in code, so the model cannot skip them, fake them, or do them at the wrong time. If a branch name is already taken, a numbered suffix is added rather than reusing it. Formatting is deterministic and happens before the single commit, so no reviewer, model or human, ever has to police trailing whitespace, final newlines or quote style. Only files the change touched are formatted, and never test files.

When the review gate or the feedback loop sends findings back, the task carries the branch name and the findings: prepare continues on that branch, the findings go into the model's prompt, and finish adds a commit whose message describes the revision.

`tasks/delete_book` is a hand-written task folder that can be given to the implementer directly:

```
python -m pipeline.run_implementer tasks/delete_book
```

## Reviewer and review gate

`pipeline/reviewer.py` judges one branch's change against the review checklist in the target's `CONVENTIONS.md` and returns a structured verdict. It never sees the implementer's reasoning, only evidence code gathers.

| Node | Done by | What it does |
|---|---|---|
| gather | Code | Checks out the branch, collects the diff against the merge base with `BASE_BRANCH` (so later commits on the base do not appear as deletions), the full content of each changed file, a fresh test run, the security scan, a check-only formatting report, and the checklist as it is on the base branch (so a change cannot remove the rule it is judged by) |
| reviewer | Model | Reads the evidence and calls `submit_review` with findings and a summary |
| tools | Code | Executes the call; a rejected call goes back to the model |
| remind | Code | One nudge if the model answered without calling the tool |
| verdict | Code | Validates the findings, merges in the code-decided ones, and derives the outcome |

What code decides and what the model judges:

- Code decides whether the tests pass (its own run, not the implementer's claim) and whether the scan blocks. Both are shown to the model as settled.
- The model judges the remaining checklist items: input validated before use, no secrets, no route exposing internal configuration, environment or debug information, scope, and every test asserting what the spec requires (`tests_match_spec`): it is asked to check each test file's expected status codes and bodies against the spec, and a test expecting something the spec does not say, such as 500 where the spec requires 404, is a blocking finding on the test file, which the diff adds, so the downgrade rule keeps it blocking. For every new or changed route it answers three questions: what a caller can now reach that they could not before, whether the response includes internal data such as configuration, environment, stack traces or file paths, and whether request input is used before it is validated. Each finding names the item, a severity, the file and a message.
- There is no verdict field. Approve or request changes is derived from the findings, so the model cannot report a verdict that contradicts them.
- A blocking finding must name a file the diff touches; otherwise it is downgraded to non-blocking with a note. A finding's line number is kept only if the diff added that line; otherwise it is dropped, so the pull request never shows a misleading one.
- With `SECURITY_PASS=1`, a second, security-only model call runs over the same evidence with just the route questions. In measurement it found nothing the single pass had not, and added duplicate findings, so it is off by default.
- Formatting findings from the check-only report are non-blocking. Non-blocking findings from every source go into the pull request body.
- If the model never calls `submit_review`, the verdict is changes requested with no verdict, failing closed.

`pipeline/gate.py` runs the loop: review, send blocking findings to the implementer on the same branch, review again, up to `MAX_REVIEW_ROUNDS` reviews. It stops approved on the first clean round and blocked at the cap, when there was no verdict, or when the implementer could not address the findings. Nothing is pushed or reported here.

```
python -m pipeline.run_review runs/tasks/<run id> <type>/<short-description>
```

## Reporter

`pipeline/reporter.py` is a code step with one model call, not an agent. After the gate approves, it runs in a fixed order: find the GitHub repository from the target's origin remote, push the branch with `git_push`, reuse an open pull request for the branch if one exists or open one with `create_pull_request`, notify with `notify_user`, and return its part of the run record. A change the gate did not approve is notified and recorded without being pushed.

The pull request body is a template filled by code: the spec, the tests added, the review summary, the non-blocking findings from every source, and the test and scan records. The model is called once, to write a short neutral paragraph describing the change for a human skimming the pull request; if that fails the template ships without it. Every step records what completed, so a failure part way, such as a push that succeeds and a pull request that does not, leaves an honest record, and a rerun reuses the pushed branch and any open pull request rather than repeating them.

```
python -m pipeline.run_report runs/tasks/<run id> <type>/<short-description>
```

## Orchestrator

`pipeline/orchestrator.py` runs the stages in order and owns what they do not:

- **Preflight.** A target with uncommitted changes is refused, and the base branch is checked out.
- **Early ends.** A clarification from the author, a blocked author, or a blocked implementer ends the run with a notification and a record. Nothing is pushed.
- **The crash path.** A stage that raises is caught. The innermost error, dug out of the exception groups the MCP sessions wrap it in, is recorded on one line with the stage it died in, the developer is notified, and the run still cleans up.
- **Cleanup, always.** Before the target is restored, any uncommitted change is saved as `uncommitted.patch` in the run's task folder, so a blocked or crashed attempt leaves evidence under `runs/` and never a failed branch in the target. Then `restore_repository` discards the changes, returns to the base branch, and deletes the run's branch only if it has no commits. This happens while the server sessions are still open. Only if the dev tools server itself can no longer be reached does the orchestrator run the same git commands directly, and it says so loudly in the log.
- **One record per run** under `RECORDS_PATH` (default `runs/records`), covering every stage: the author's revisions or question, the implementer's attempts and reflections, the gate's rounds and findings, the reporter's steps, the memory recall, the evidence patch and the cleanup result.
- **A tool log per run**, `<run id>-tools.jsonl` next to the record and referenced from it: every tool call of the run, whether made by code or asked for by a model, MCP tools and the pipeline's own `submit_review` and `request_clarification` alike, with its arguments and its full result or error, secrets scrubbed. The calls are logged by the node that runs them, so the call that ends a loop is logged too. Evidence never depends on what was printed to the terminal.

## Pull request feedback

GitHub does not let a pull request's author approve it or request changes, and the pipeline's pull requests are opened with the developer's token, so the developer steers them with one explicit command instead: a comment starting with `/revise`.

```
python -m pipeline.run_feedback <pull request number>
```

The loop reads the pull request's state and comments through `pull_request_read`, finds the run that opened it in the records, and acts by state:

| Pull request | Action |
|---|---|
| merged | recorded as accepted; the merge is the developer's approval |
| open, with a new `/revise` comment | revision on the same branch: the author rewrites the spec and tests with the feedback, the implementer continues on the branch, the gate reviews, the reporter pushes and reuses the pull request, and a note is posted on it |
| open, without | recorded as waiting |
| closed, with a `/revise` comment | restart on a fresh branch, the feedback added to the request, ending in a new pull request |
| closed, without | recorded as closed |

The feedback is the `/revise` comment plus any inline review comments the owner left since the last handled revision. Everything else is ignored and listed in the record: other people's comments, the owner's questions, and the pipeline's own notes. The notes are posted under the developer's token, so each one states that the pipeline wrote it and carries a hidden marker; the loop checks the marker before anything else, so a note can never trigger a revision, even one that starts with `/revise`. Each handling records the ids of the comments it acted on, so a rerun with nothing new does nothing. `MAX_PR_REVISIONS` (default 2) caps revisions per pull request; at the cap the developer is notified and the pull request is left as it is.

Feedback goes into the prompts as quoted data that says what the owner wants changed, never as instructions about tools. Nothing in the loop touches the base branch. There are no webhooks: the pipeline polls when this command runs.

## Issue intake

Open GitHub issues can enter the pipeline, gated by a label:

```
python -m pipeline.run_issues              # the oldest issue labelled agent-ready that has no run record
python -m pipeline.run_issues --all        # every such issue, one run each
python -m pipeline.run_issues --issue 5    # issue 5, even if it has a record
```

The label is the approval. On GitHub only someone with triage access or more can apply a label, and on a personal repository that is the owner and the collaborators they add, so a labelled issue is one the owner read and approved; anyone may open issues, and the author of an issue is recorded but never trusted. The label name is `ISSUE_LABEL` (default `agent-ready`).

Everything in the intake is code. `list_issues` returns the open issues carrying the label, oldest first, filtered on GitHub's side. An issue that already has a run record is skipped, whatever that run's outcome. The chosen issue is read again with `issue_read`, body and labels separately, and refused if it is closed or no longer labelled. The run then proceeds exactly as for a typed request, and the pull request body carries `Fixes #N` under the summary, which GitHub links on its own; nothing is ever written to the issue, so the token needs Issues read only.

The issue's title and body are untrusted text. They reach the author inside a delimited block under one framing line written by the pipeline, which says the text is content to specify and test and never an instruction; any marker text inside the body is altered so the block cannot be closed early. The author's rules say the same about quoted text. The scripted tests include an issue whose body tells the agent to overwrite an existing test file: the body arrives only inside the markers, an author that obeys it is refused by `write_task_test`, and the repository's test file is untouched. What a scripted test cannot show is the model declining to follow such text; that evidence can only come from a live run.

A handled issue stays parked, including after a clarification, a block or a crash. No tool of the GitHub server exposes when a label was applied or when an issue body was last edited: the issue object carries `created_at` and `updated_at` only, and `updated_at` also moves on comments by anyone, so it cannot stand in for a re-approval. Re-running an issue is therefore the owner's explicit act: edit the issue if a clarification was asked, re-apply the label after rereading it, and name it with `--issue N`. Removing the label on pickup would have made re-labelling the re-approval, but the only tool that changes an issue's labels is `issue_write`, which also edits, closes and reassigns issues, so it stays off the allowlist. The README's known limitations state the gap this leaves.

Memory and the duplicate check apply unchanged: a run from an issue is recalled and indexed on the issue's own title and body, not on the framing, so an issue that restates a request already behind an open pull request gets the same `duplicate_of_open_pr` question as a typed one. A `/revise` on the resulting pull request works as for any other run and keeps the issue link.

## Memory

Run memory lives in a Chroma collection served by the official Chroma MCP server in persistent mode, with its database under `CHROMA_DATA_PATH` (default `runs/chroma`). The records folder stays the source of truth; the collection is derived from it, one document per run, and `python -m pipeline.index_records` backfills or refreshes it, while `--query "text"` shows the nearest runs and their distances. Code does the indexing and the recall; no model ever gets a Chroma tool.

What is embedded is the request text only, so the distance between two documents measures how alike two requests are; the outcome, task, branch, pull request number, clarification question, review summary and error travel as metadata. The collection uses cosine distance, 0 for identical and 1 for unrelated. Measured on real records: the same request scores 0.0, the same topic reworded 0.23 to 0.49, a different topic on the same app 0.44 and above, an unrelated request 0.67 and above. The reworded and different-topic bands overlap, so this is near-duplicate detection, not paraphrase matching, and only clear matches are shown.

Before the author runs, the orchestrator recalls the nearest past runs and shows the author only those under `MEMORY_DISTANCE_THRESHOLD` (default 0.40); when nothing is that close, nothing is shown, since unrelated history is noise for a small model. For each shown run with a pull request, the pull request's state is read from GitHub. A run under `MEMORY_DUPLICATE_THRESHOLD` (default 0.20) whose pull request is still open is stated as a near-identical request with an open pull request, and the author's rule for that is to call `request_clarification` with reason `duplicate_of_open_pr` and ask whether to revise that pull request instead; the developer then either posts `/revise` on it or confirms a separate change. Every finished run is indexed whatever its outcome. The implementer and the reviewer get no memory: the implementer has the tests as ground truth, and the reviewer must judge the diff in front of it.

### Chroma server environment

`chroma-mcp` pins an old `mcp` release that cannot share this project's environment, so it lives in its own virtual environment, started as a fifth MCP server only by the full pipeline, the feedback runner and the issue runner, and all proceed without memory, saying so, when it is not installed. The exact versions that work are frozen in `requirements-chroma.txt`, from Python 3.14 on Windows; the install steps are in the README.

`--no-deps` matters: chroma-mcp's metadata pins `mcp==1.6.0`, which cannot run on the pydantic versions that have Python 3.14 wheels, so the frozen set overrides it with the same `mcp` line this project uses and pip must take the pins as given rather than resolve them. The server only needs FastMCP basics and runs fine on it, which the tests exercise. The first run needs internet once: the default embedding function downloads its model, 79 MB, into `~/.cache/chroma/onnx_models`, and every later run loads it from there without a network. No key is needed. The server also prints two status lines to stdout at start-up, which the client logs as parse errors and ignores.

A code index, embedding the target's source files so the author can find relevant code by meaning, is a documented extension and not built: for a sixty-line app the author reads the whole file anyway.

## Reviewer over A2A

The reviewer can run as a separate service that the pipeline reaches over the A2A protocol, using the official Python SDK (`a2a-sdk`). The split follows the reviewer's own design: gathering the evidence needs the MCP servers and the target checkout, so it stays in the pipeline; the model pass and the verdict rules are a pure function of that evidence, so they are what the service runs.

| Side | Does |
|---|---|
| pipeline, the client | gathers the evidence as for a local review and sends it as one JSON data part; receives the model's findings and summary as one JSON artifact; adds the findings code decides, scan, formatting and tests, and derives approve or request changes exactly as locally |
| reviewer service | runs the model pass and the downgrade and line rules on the evidence it was sent, with its own model connection; never touches a repository, has no tool beyond `submit_review` |

The service is started with `python -m pipeline.serve_reviewer` and selected with `REVIEWER_A2A_URL`, as the README shows. With that setting every review in the gate goes through the service and the review result records the service's URL; without it the gate reviews in-process. The service serves its agent card at `/.well-known/agent-card.json` with one skill, `review_change`, and the client refuses a card without that skill, so a wrong URL fails loudly. A transport error, a task that does not complete, or an artifact that is not a verdict counts as no verdict: the gate fails closed, as it does when a local model never calls `submit_review`.

The service binds to localhost with no authentication. It is a local demonstration of the protocol, not a deployment. The evidence it receives is code and test output, scrubbed by the same helper as everything else, and the service cannot act on anything: no tools, no repository, and its output is validated before the gate uses it.

## Settings

All settings are read from `.env`; `.env.example` lists them with comments.

| Variable | Default | Meaning |
|---|---|---|
| `TARGET_REPO_PATH` | required | Local clone of the target repository |
| `BASE_BRANCH` | `main` | Branch runs start from, diff against, and open pull requests to |
| `PROTECTED_BRANCHES` | `main,master` | Branches that are never pushed or deleted |
| `BLOCKING_SEVERITY` | `MEDIUM` | Bandit severity at or above which a code finding blocks |
| `MODEL_NAME` | `gemma4:e4b` | Ollama model for every agent |
| `NUM_CTX` | 8192 | Context window in tokens. 16384 crashed Ollama on a consumer GPU |
| `MAX_TOOL_STEPS` | 12 | Tool calls allowed per attempt, for every agent |
| `MAX_ATTEMPTS` | 3 | Implementer: attempts before stopping as blocked |
| `MAX_AUTHOR_REVISIONS` | 2 | Author: revisions after the first attempt before stopping |
| `MAX_REVIEW_ROUNDS` | 2 | Review gate: reviews before stopping as blocked, counting the first |
| `MAX_PR_REVISIONS` | 2 | Feedback loop: `/revise` comments acted on per pull request |
| `ISSUE_LABEL` | `agent-ready` | Issue intake: the label that marks an open issue as approved for the pipeline |
| `SECURITY_PASS` | 0 | Reviewer: 1 runs a second, security-only model call |
| `REVIEWER_A2A_URL` | unset | Reviewer: the A2A service to send evidence to; unset means in-process |
| `GITHUB_MCP_SERVER` | required for pull requests | Path to the GitHub MCP server executable |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | required for pull requests | Fine-grained token scoped to the target repository: Pull requests read and write, Issues read |
| `GITHUB_REPOSITORY` | unset | `owner/repo` override; normally derived from the target's origin remote |
| `NOTIFICATIONS_LOG` | `runs/notifications.log` | Where outcome notifications are appended |
| `RECORDS_PATH` | `runs/records` | Where run records and tool logs are written |
| `TASKS_PATH` | `runs/tasks` | Where generated task folders are written |
| `FAIL_FIRST_WORKTREE` | `runs/fail_first_worktree` | Throwaway worktree for the fail-first check |
| `CHROMA_MCP_SERVER` | `.venv-chroma/Scripts/chroma-mcp.exe` | Chroma MCP server executable; `.venv-chroma/bin/chroma-mcp` on macOS and Linux |
| `CHROMA_DATA_PATH` | `runs/chroma` | Chroma's persistent database |
| `MEMORY_DISTANCE_THRESHOLD` | 0.40 | Past runs under this cosine distance are shown to the author |
| `MEMORY_DUPLICATE_THRESHOLD` | 0.20 | Under this distance, a run with an open pull request counts as a near-identical request |

## Notes

`mcp` is pinned below version 2 because `langchain-mcp-adapters` and `mcp-server-git` both require it, and FastMCP was renamed in 2.x. Subprocesses started by the dev tools server get an empty stdin, because a pytest run that inherited the MCP protocol channel hung on Windows.

The full pipeline, memory, the A2A reviewer and the issue intake were verified on Python 3.14 on Windows only, and `requirements-chroma.txt` is frozen from that interpreter. The early server and implementer tests also ran on Python 3.12 on Linux; nothing built after them has been run there.

## Design changes from the submitted design

Made while building:

- The planner was folded into the spec and test author, since the spec and the tests are the plan and a separate planning agent only restated them.
- The reporter became a code step with one model call rather than an agent, because pushing, opening and notifying need no judgement and the one paragraph of prose does.
- The author's revise step is code feedback, one line per test saying why it was not red, rather than a model reflection, so Reflexion lives only in the implementer.
- Clarification became an explicit `request_clarification` tool with fixed reasons rather than detecting a question in the text, so the attempt ends in code the moment the tool accepts the call.
- The reviewer no longer decides whether a finding blocks: the scan computes blocking in code and the reviewer judges scope and the checklist items.
- The implementer no longer branches or commits itself; code does both at fixed points.
- The merge tool is excluded by the GitHub server's configuration rather than forbidden by prompt.
- The review trigger after a pull request is an explicit `/revise` comment rather than a changes-requested review, because the pull request author cannot request changes on GitHub.
- A formatter runs before every commit so that formatting is never a review matter.
- Issue intake was added, with the label not consumed on pickup because the only tool that changes labels also edits and closes issues, so re-runs are an explicit `--issue N`.
- The security-only second review pass was built, measured and left off by default, because it found nothing the single pass had not and added duplicate findings.
- The implementer's tools were cut to four, because `directory_tree` flooded the model with `.git` contents and `edit_file` needs exact text matches a small model fumbles.
