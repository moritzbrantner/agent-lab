"""Bounded self-proposals and independent correctness/compute decisions."""

import copy
from collections import defaultdict
from dataclasses import asdict, replace

from agent_lab.experiments import compare, digest, validate_result
from agent_lab.runtime import AgentState, run
from agent_lab.trace import Trace


def materialize(baseline, proposal):
    if not isinstance(proposal, dict) or set(proposal) != {"changes", "reason"}:
        raise ValueError("Proposal requires changes and reason")
    reason, changes = proposal["reason"], proposal["changes"]
    if (
        not isinstance(reason, str)
        or not 1 <= len(reason) <= 1280
        or not isinstance(changes, dict)
        or not changes
    ):
        raise ValueError("Invalid bounded proposal")
    allowed = {"context_policy", "feedback", "max_retries", "max_output_tokens"}
    if changes.keys() - allowed:
        raise ValueError("Mutation cannot alter evaluation or tool authority")
    if "feedback" in changes and (
        not isinstance(changes["feedback"], str) or len(changes["feedback"]) > 1280
    ):
        raise ValueError("Instruction mutation too large")
    for name, bounds in {"max_retries": (0, 3), "max_output_tokens": (16, 512)}.items():
        if name in changes and (
            type(changes[name]) is not int
            or not bounds[0] <= changes[name] <= bounds[1]
        ):
            raise ValueError("Mutation exceeds budget bounds")
    return replace(baseline, **changes)


def development_view(results):
    groups = {}
    for result in results:
        validate_result(result)
        if result["protocol"]["split"] != "development":
            raise ValueError("Proposal cannot inspect held-out evidence")
        config = {k: v for k, v in result["configuration"].items() if k != "seed"}
        key = result["task"]["id"], digest(config)
        groups.setdefault(
            key,
            {
                "task": key[0],
                "configuration": config,
                "scores": [],
                "tokens": [],
                "run_ids": [],
            },
        )
        groups[key]["scores"].append(result["correctness"]["score"])
        groups[key]["tokens"].append(result["work"]["generated_tokens"])
        groups[key]["run_ids"].append(result["run_id"])
    return list(groups.values())


async def propose(adapter, baseline, development_results):
    from agent_lab.experiments import canonical

    view = development_view(development_results)
    prompt = (
        "Return a JSON reply with content and calls. Propose one bounded improvement. "
        "Call propose "
        "once with arguments {changes: object, reason: string}. Allowed changes: "
        "context_policy full/retained/relevance, feedback string (1280 chars), "
        "max_retries 0..3, max_output_tokens 16..512. Relevance selects only the "
        "requested fact for retrieval; retained compresses completed tool history. "
        "Use the observed development results, especially retrieval comparisons. "
        "You have no filesystem, fixture or evaluator authority.\n" + canonical(view)
    )
    proposals = []

    def capture(arguments):
        materialize(baseline, arguments)
        if proposals:
            raise ValueError("Only one mutation may be proposed")
        proposals.append(copy.deepcopy(arguments))
        return "proposal recorded; finish with calls: []"

    planner = replace(baseline, max_steps=2, max_tool_calls=1)
    trace = Trace(
        digest({"prompt": prompt, "config": asdict(planner)})[:24],
        "scaffolding-proposal",
        digest(asdict(planner)),
    )
    state = AgentState(prompt)
    error = None
    try:
        await run(
            adapter,
            {"propose": capture},
            planner,
            state,
            trace.emit,
        )
    except Exception as failure:
        error = type(failure).__name__
    return (
        proposals[0] if proposals else None,
        trace,
        {
            "state": state.status,
            "error": error,
            "input_tokens": state.input_tokens,
            "output_tokens": state.output_tokens,
            "model_calls": state.model_calls,
            "development_evidence_sha256": digest(view),
            "observed_final_content": state.output,
        },
    )


def _cohorts(results):
    groups = defaultdict(list)
    identities = set()
    for result in results:
        validate_result(result)
        key = result["protocol"]["split"], result["task"]["id"]
        identity = (*key, result["protocol"]["repeat"])
        if identity in identities:
            raise ValueError("Duplicate cohort sample")
        identities.add(identity)
        groups[key].append(result)
    return groups


def decide(baseline, candidate):
    before, after = _cohorts(baseline), _cohorts(candidate)
    if before.keys() != after.keys() or not {"development", "held-out"}.issubset(
        {k[0] for k in before}
    ):
        raise ValueError("Decision requires matching development and held-out cohorts")
    comparisons, regression, improvement, unknown = [], [], False, False
    for key, initial in before.items():
        changed = after[key]
        if len(initial) != len(changed):
            raise ValueError("Cohort sample counts differ")
        initial = sorted(initial, key=lambda r: r["protocol"]["repeat"])
        changed = sorted(changed, key=lambda r: r["protocol"]["repeat"])
        for a, b in zip(initial, changed, strict=True):
            comparisons.append(compare(a, b))
        a_scores, b_scores = (
            [r["correctness"]["score"] for r in initial],
            [r["correctness"]["score"] for r in changed],
        )
        if None in a_scores or None in b_scores:
            unknown = True
            continue
        if sum(b_scores) < sum(a_scores):
            regression.append(list(key))
        if key[0] == "development" and sum(b_scores) > sum(a_scores):
            improvement = True
    a_tokens = [r["work"]["generated_tokens"] for r in baseline]
    b_tokens = [r["work"]["generated_tokens"] for r in candidate]
    token_improvement = (
        None not in a_tokens and None not in b_tokens and sum(b_tokens) < sum(a_tokens)
    )
    status = (
        "reject"
        if regression or unknown
        else "accept"
        if improvement or token_improvement
        else "archive"
    )
    return {
        "status": status,
        "correctness_regressions": regression,
        "unknown_correctness": unknown,
        "development_improvement": improvement,
        "total_token_improvement": token_improvement,
        "comparisons": comparisons,
        "baseline_sha256": digest(baseline),
        "candidate_sha256": digest(candidate),
    }
