"""Incrementally retained task state and explicit model context projections."""

import copy

from agent_lab.experiments import canonical


def project_context(state, policy):
    if policy in ("full", "relevance"):
        return copy.deepcopy(state.messages)
    if policy != "retained":
        raise ValueError("Unknown context policy")
    completed = [
        [entry["name"], entry["arguments"], entry["result"]]
        for entry in state.working_memory
    ]
    return [
        copy.deepcopy(state.messages[0]),
        {
            "role": "user",
            "content": "Completed tools (name, args, result). No repeats.\n"
            + canonical({"completed": completed}),
        },
    ]


def select_input(task, policy):
    value = copy.deepcopy(task["input"])
    if policy == "relevance" and task["category"] == "retrieval":
        value["facts"] = {value["key"]: value["facts"][value["key"]]}
    return value
