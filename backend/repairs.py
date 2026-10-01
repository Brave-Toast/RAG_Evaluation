"""
repair methods - trys to fix a broken query/context before we sedn it to the ai
each repair funciton gets added to REPAIR_REGISTRY under a name, then apply_repair_methods
jsut looks up whichver ones were askd for and runs them one after antoher
"""

import re
from typing import List, Dict, Callable

#A dictionary that holds the label for repair and thjen would call the function to fix a query issue
REPAIR_REGISTRY: Dict[str, Callable] = {}


def register_repair(name):
    #reapir_registery is empty at the moment, for eahc key we need to asign a function to it so it can run
    #this decorator does that automaticaly for us instead of writing it manually evrytime
    def wrapper(fn):
        REPAIR_REGISTRY[name] = fn #save this function as the vlaue for this key
        return fn #hand the function back unchanged so it still works normaly elsewhere
    return wrapper


def apply_repair_methods(query, contexts, methods):
    #checking diffrent fixes ive selected to run on the query, one by one
    #for each one we look up the function using its key, and to actually run it we need the ()
    #if no function exists for that key we jsut say no such repair and move to next one
    #if it does exist we call it, then append what changed to the changes list
    #at the end we retrun the toal fixed query, fixed contexts, and all the appended changes
    changes = []
    for method in methods:
        fn = REPAIR_REGISTRY.get(method)
        if fn is None:
            changes.append({"method": method, "status": "skipped", "summary": f"no repair registerd for '{method}'"})
            continue
        query, contexts, record = fn(query, contexts)
        changes.append(record)
    return query, contexts, changes

#@register_repair("noop") calls register_repair fisrt with the name, then feeds repair_noop into wrapper to save it in the dict
@register_repair("noop")
def repair_noop(query, contexts):
    return query, contexts, {"method": "noop", "status": "applied", "summary": "no changes made (test repair)"}


# --- Conflicting-Context Repair Method (RS-44 / metadata injection ticket) ---
#
# Detects documents that look like they were fabricated by our own conflict
# or metadata-injection perturbations (see perturbations.py:
# build_metadata_boosted_candidate, and app.py: inject_conflicting_context)
# and filters them out before the query reaches generation.
#
# Two signals are checked, on purpose:
#   1. Title prefix "Synthetic" -- a strong, cheap signal, but it's really
#      just our own internal naming convention for test documents. Relying
#      on it alone would score a trivial 100% on our own synthetic test data
#      and tell us nothing useful about real-world detection.
#   2. Suspicious phrasing in the document TEXT itself (the actual injected
#      wording: "conflicting modern sources", "refute", "alternative facts",
#      the "[Topic tags: ...]" metadata tag prefix). This is the more
#      meaningful signal to evaluate, since it's the kind of heuristic that
#      could plausibly catch similar injected content even without the
#      "Synthetic" title marker attached.
#
# See test_repair_accuracy.py for a precision/recall evaluation of this
# detector against both injected and genuine (unperturbed) documents.

_INJECTION_TEXT_MARKERS = re.compile(
    r"conflicting modern sources|refute that|alternative facts|\[topic tags:",
    re.IGNORECASE,
)


def _looks_injected(doc: dict) -> bool:
    """Returns True if a document looks like a fabricated conflict/metadata
    injection, based on its title naming convention or suspicious text markers."""
    title = (doc.get("title") or "").strip()
    text = doc.get("text") or ""
    if title.lower().startswith("synthetic"):
        return True
    return bool(_INJECTION_TEXT_MARKERS.search(text))


@register_repair("contradiction_filter")
def repair_contradiction_filter(query, contexts):
    """
    Scans the supplied contexts for documents that look like fabricated
    conflicting-context or metadata-injection perturbations, and removes
    them from the list before it's passed on to generation.
    """
    kept = []
    removed_titles = []

    for doc in contexts:
        if _looks_injected(doc):
            removed_titles.append(doc.get("title", "untitled"))
        else:
            kept.append(doc)

    if removed_titles:
        summary = (
            f"Filtered {len(removed_titles)} suspected injected/contradicting "
            f"document(s): {', '.join(removed_titles)}."
        )
        status = "applied"
    else:
        summary = "No suspected injected or contradicting documents found; contexts unchanged."
        status = "skipped"

    record = {
        "method": "contradiction_filter",
        "status": status,
        "summary": summary,
        "details": {"removed_titles": removed_titles, "remaining_count": len(kept)},
    }
    return query, kept, record