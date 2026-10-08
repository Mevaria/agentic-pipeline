"""The review gate: review a branch, send blocking findings back to the implementer, repeat up to a cap.

Round 1 reviews the branch as the implementer left it. If the verdict has
blocking findings, they are handed to the implementer, which continues on the
same branch and commits a fix, and the next round reviews the new state. The
gate stops approved as soon as a round has no blocking findings, and stops
blocked when the cap of MAX_REVIEW_ROUNDS reviews is reached, when the
reviewer produced no verdict, or when the implementer could not address the
findings. Nothing here pushes or notifies; the reporter does that.
"""

from pipeline.implementer import build_implementer
from pipeline.implementer import recursion_limit as implementer_recursion_limit
from pipeline.reviewer import build_reviewer, describe_findings
from pipeline.reviewer import recursion_limit as reviewer_recursion_limit


async def run_gate(repo_path, all_tools, implementer_tools, implementer_model, reviewer_model, task, branch,
                   implementer_settings, review_settings, log=print):
    """Run the review loop on a branch and return its outcome.

    Returns a dict with status ("approved" or "blocked"), the list of review
    results in order (rounds), the implementer states for the fix attempts
    (fixes), and the final review (review).
    """
    reviewer_graph = build_reviewer(repo_path, all_tools, reviewer_model, review_settings, log)
    implementer_graph = build_implementer(repo_path, all_tools, implementer_tools, implementer_model, implementer_settings, log)
    rounds = []
    fixes = []
    for round_number in range(1, review_settings["max_review_rounds"] + 1):
        log(f"[gate] review round {round_number} of {review_settings['max_review_rounds']} on {branch}")
        state = await reviewer_graph.ainvoke({"task": task, "branch": branch},
                                             {"recursion_limit": reviewer_recursion_limit(review_settings)})
        review = state["result"]
        rounds.append(review)
        if review["approved"]:
            return {"status": "approved", "rounds": rounds, "fixes": fixes, "review": review}
        # Without a verdict there are no findings to act on, so the implementer is not sent in circles.
        if review["no_verdict"] or round_number == review_settings["max_review_rounds"]:
            break
        # The implementer continues on the same branch with the blocking findings in its prompt.
        fix_task = {**task, "branch": branch, "review_findings": describe_findings(review["blocking_findings"])}
        log(f"[gate] sending {len(review['blocking_findings'])} blocking finding(s) to the implementer")
        fix_state = await implementer_graph.ainvoke({"task": fix_task},
                                                    {"recursion_limit": implementer_recursion_limit(implementer_settings)})
        fixes.append(fix_state)
        if fix_state["status"] != "passed":
            log(f"[gate] the implementer could not address the findings ({fix_state['status']})")
            break
    return {"status": "blocked", "rounds": rounds, "fixes": fixes, "review": rounds[-1]}
