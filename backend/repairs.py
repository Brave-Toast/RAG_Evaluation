"""
repair methods - trys to fix a broken query/context before we sedn it to the ai
each repair funciton gets added to REPAIR_REGISTRY under a name, then apply_repair_methods
jsut looks up whichver ones were askd for and runs them one after antoher
"""

import re
from typing import Dict, Callable

#A dictionary that holds the label for repair and thjen would call the function to fix a query issue
REPAIR_REGISTRY: Dict[str, Callable] = {}


def register_repair(name):
    #reapir_registery is empty at the moment, for eahc key we need to asign a function to it so it can run
    #this decorator does that automaticaly for us instead of writing it manually evrytime
    def wrapper(fn):
        REPAIR_REGISTRY[name] = fn #save this function as the vlaue for this key
        return fn #hand the function back unchanged so it still works normaly elsewhere
    return wrapper


def apply_repair_methods(query, contexts, methods, config=None):
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
        query, contexts, record = fn(query, contexts, config)
        changes.append(record)
    return query, contexts, changes

#@register_repair("noop") calls register_repair fisrt with the name, then feeds repair_noop into wrapper to save it in the dict
@register_repair("noop")
def repair_noop(query, contexts, config=None):
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
def repair_contradiction_filter(query, contexts, config=None):
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


@register_repair("fix_shuffle")
def repair_fix_shuffle(query, contexts, config=None):
    """Restore shuffled passages to descending stored retrieval-score order."""
    original_order = [doc.get("title", "untitled") for doc in contexts]
    # Missing scores cannot establish relevance, so they remain after scored
    # passages and retain their relative input order via Python's stable sort.
    ranked = sorted(
        enumerate(contexts),
        key=lambda item: (
            item[1].get("score") is not None,
            item[1].get("score") or 0.0,
        ),
        reverse=True,
    )
    repaired = [doc for _, doc in ranked]
    repaired_order = [doc.get("title", "untitled") for doc in repaired]
    changed = [index for index, _ in ranked] != list(range(len(contexts)))
    details = {
        "context_count": len(contexts),
        "original_order": original_order,
        "repaired_order": repaired_order,
        "relevance_scores": [
            {
                "original_index": original_index,
                "repaired_index": repaired_index,
                "title": doc.get("title", "untitled"),
                "score": doc.get("score"),
            }
            for repaired_index, (original_index, doc) in enumerate(ranked)
        ],
    }
    if len(contexts) < 2:
        summary = f"No reranking needed: received {len(contexts)} context(s)."
    elif changed:
        summary = f"Restored {len(contexts)} passages to descending retrieval-score order."
    else:
        summary = "Passages were already in descending retrieval-score order."
    return query, repaired, {
        "method": "fix_shuffle",
        "status": "applied" if changed else "skipped",
        "summary": summary,
        "details": details,
    }


def _find_full_text(doc, full_contexts):
    """Return one unambiguous same-title full text that extends the supplied prefix."""
    title = doc.get("title")
    truncated_text = doc.get("text") or ""
    matches = [
        candidate.get("text") or ""
        for candidate in full_contexts
        if candidate.get("title") == title
        and len(candidate.get("text") or "") > len(truncated_text)
        and (candidate.get("text") or "").startswith(truncated_text)
    ]
    return matches[0] if len(matches) == 1 else None


@register_repair("fix_truncate")
def repair_fix_truncate(query, contexts, config=None):
    """Detect truncated passage prefixes and restore exact post-retrieval full text."""
    full_contexts = (config or {}).get("full_contexts") or []
    repaired = []
    restored = []
    unresolved = []

    for index, doc in enumerate(contexts):
        full_text = _find_full_text(doc, full_contexts)
        if full_text is None:
            repaired.append(doc)
            if doc.get("text") and not re.search(r"[.!?][\"')\]]?\s*$", doc["text"]):
                unresolved.append({"index": index, "title": doc.get("title", "untitled")})
            continue
        restored_doc = dict(doc)
        restored_doc["text"] = full_text
        repaired.append(restored_doc)
        restored.append({
            "index": index,
            "title": doc.get("title", "untitled"),
            "truncated_characters": len(doc.get("text") or ""),
            "restored_characters": len(full_text),
        })

    changed = bool(restored)
    if changed:
        summary = f"Restored exact full text for {len(restored)} truncated passage(s)."
    elif unresolved:
        summary = "Detected possible cut-off passage(s), but no unambiguous full-text source matched; nothing was fabricated."
    else:
        summary = "No truncated passage with a matching full-text source was found."
    return query, repaired, {
        "method": "fix_truncate",
        "status": "applied" if changed else "skipped",
        "summary": summary,
        "details": {
            "context_count": len(contexts),
            "restored": restored,
            "unresolved_cutoffs": unresolved,
        },
    }
