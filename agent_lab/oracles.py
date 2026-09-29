"""Independent full recomputation oracle for incremental ordered state."""

import bisect
import copy

from agent_lab.experiments import canonical


class IncrementalIndex:
    def __init__(self):
        self.records = {}
        self.keys = []

    def apply(self, change):
        if change["op"] == "put":
            record = copy.deepcopy(change["record"])
            record_id = record["id"]
        elif change["op"] == "delete":
            record_id = change["id"]
        else:
            raise ValueError("Unknown index operation")
        old = self.records.pop(record_id, None)
        if old is not None:
            key = (old["rank"], record_id)
            self.keys.pop(bisect.bisect_left(self.keys, key))
        if change["op"] == "put":
            self.records[record_id] = record
            bisect.insort(self.keys, (record["rank"], record_id))
        return [copy.deepcopy(self.records[record_id]) for _, record_id in self.keys]


def reference_sequence(changes):
    records = {}
    outputs = []
    for change in changes:
        if change["op"] == "put":
            records[change["record"]["id"]] = copy.deepcopy(change["record"])
        elif change["op"] == "delete":
            records.pop(change["id"], None)
        else:
            raise ValueError("Unknown reference operation")
        outputs.append(
            sorted(
                copy.deepcopy(list(records.values())),
                key=lambda value: (value["rank"], value["id"]),
            )
        )
    return outputs


def incremental_sequence(changes):
    index = IncrementalIndex()
    return [index.apply(change) for change in changes]


def differential_report(changes):
    observed = incremental_sequence(changes)
    reference = reference_sequence(changes)
    return {
        "candidate": "incremental-index-v1",
        "reference": "full-sort-v1",
        "steps": len(changes),
        "observable_equal": canonical(observed) == canonical(reference),
    }
