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


# --- Conflicting-Context Repair Method (RS-57) ---
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
# A second filter, "contradiction_filter_text_only", uses signal 2 only, so
# test_repair_accuracy.py can show how much the title label contributes.
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


@register_repair("contradiction_filter_text_only")
def repair_contradiction_filter_text_only(query, contexts):
    """Same as contradiction_filter but ignores the title, so it checks text only."""
    kept, removed = [], []
    for doc in contexts:
        if _INJECTION_TEXT_MARKERS.search(doc.get("text") or ""):
            removed.append(doc.get("title", "untitled"))
        else:
            kept.append(doc)
    record = {
        "method": "contradiction_filter_text_only",
        "status": "applied" if removed else "skipped",
        "summary": f"Removed {len(removed)} document(s) using text markers only.",
        "details": {"removed_titles": removed, "remaining_count": len(kept)},
    }
    return query, kept, record


# --- Altered-Facts Repair Methods (RS-62) ---
#
# PASTE THIS BLOCK AT THE VERY END OF backend/repairs.py (it only needs `re`, which is
# already imported at the top of that file).
#
# The "alter_facts" perturbation (perturbations.py) changes numbers and dates inside the
# retrieved passages. These repairs cross-check numbers ACROSS the retrieved passages:
#
#   1. Every integer is given a "signature": the two words before it, the word after it,
#      and a size class (1-2 digits, 3-4 digits, longer). Two numbers in different passages
#      with the same signature are treated as the same fact.
#   2. If passages disagree on a fact and a clear majority (2+ passages) agrees on one value,
#      the out-voted figure is corrected to the majority value ("fix_alter_facts") or its
#      passage is dropped ("fix_alter_facts_drop").
#   3. If the disagreement is one against one, nobody can tell which side is right, so it is
#      only FLAGGED in the repair record and the passages are left unchanged.
#   4. Impossible dates (30 February, 31 April) and reversed year ranges (1850-1846) are
#      flagged ("fix_alter_facts") or the passage is dropped ("fix_alter_facts_drop").
#
# Passages whose titles are identical apart from digits (e.g. "2014 Bathurst 12 Hour" and
# "2015 Bathurst 12 Hour") are treated as siblings and never compared, because they
# legitimately differ.
#
# Only passages that REFER TO EACH OTHER are compared (one passage's text mentions the other's
# title). Without this, unrelated passages that merely share a sentence shape ("born April 1971"
# vs "born April 1975", "a 2008 film" vs "a 2012 film") were flagged as conflicts, and a chance
# majority among them could have "corrected" a genuine number.
#
# Limits (documented on purpose, see test_fix_alter_facts.py): in HotpotQA a fact usually
# appears in only ONE passage, so most altered numbers have nothing to cross-check against,
# and the same fact must be worded the same way in both passages to be matched.
# The repair never uses a passage's retrieval score, because alter_facts sets it to None.

_FACT_NUMBER = re.compile(r"(?<![\w.,])[0-9]{1,9}(?![\w.,])")
_FACT_WORD = re.compile(r"[A-Za-z]+")
_MONTH_DAYS = {
    "january": 31, "february": 29, "march": 31, "april": 30, "may": 31, "june": 30,
    "july": 31, "august": 31, "september": 30, "october": 31, "november": 30, "december": 31,
}
_MONTH_NAMES = "|".join(_MONTH_DAYS)
_DAY_MONTH = re.compile(r"\b([0-9]{1,2})\s+(" + _MONTH_NAMES + r")\b", re.IGNORECASE)
_MONTH_DAY = re.compile(r"\b(" + _MONTH_NAMES + r")\s+([0-9]{1,2})\b", re.IGNORECASE)
_YEAR_RANGE = re.compile(r"\b([0-9]{4})\s*[\u2013\u2014-]\s*([0-9]{4})\b")


def _title_stem(title):
    """Title with all digits removed; passages with the same stem are 'siblings'."""
    return re.sub(r"\s+", " ", re.sub(r"[0-9]+", " ", title or "")).strip().casefold()


def _base_title(title):
    """Title without a trailing qualifier such as '(2008 film)', lower-cased."""
    return re.sub(r"\s*\([^)]*\)", "", title or "").strip().casefold()


def _passages_linked(doc_a, doc_b):
    """True if either passage's text mentions the other passage's title."""
    for first, second in ((doc_a, doc_b), (doc_b, doc_a)):
        base = _base_title(second.get("title"))
        if len(base) >= 4 and base in (first.get("text") or "").casefold():
            return True
    return False


def _fact_signatures(text):
    """Every integer in `text` with a signature describing the words around it."""
    facts = []
    for match in _FACT_NUMBER.finditer(text):
        prefix = text[:match.start()]
        words = _FACT_WORD.findall(prefix[-60:])
        if len(prefix) > 60:
            words = words[1:]  # the first word of the window may be cut in half
        before = words[-2:]
        if len(before) < 2:
            continue
        after = _FACT_WORD.findall(text[match.end():match.end() + 20])[:1]
        value = match.group()
        size = "n" if len(value) <= 2 else ("y" if len(value) <= 4 else "L")
        after_word = after[0].lower() if after else ""
        signature = " ".join(w.lower() for w in before) + " | " + after_word + " | " + size
        facts.append({"value": value, "start": match.start(), "end": match.end(), "signature": signature})
    return facts


def _implausible_facts(text):
    """Impossible dates and reversed year ranges, which natural text almost never contains."""
    problems = []
    for match in _DAY_MONTH.finditer(text):
        day, month = int(match.group(1)), match.group(2).lower()
        if day < 1 or day > _MONTH_DAYS[month]:
            problems.append(match.group(0))
    for match in _MONTH_DAY.finditer(text):
        month, day = match.group(1).lower(), int(match.group(2))
        if day < 1 or day > _MONTH_DAYS[month]:
            problems.append(match.group(0))
    for match in _YEAR_RANGE.finditer(text):
        if int(match.group(1)) > int(match.group(2)):
            problems.append(match.group(0))
    return problems


def _fix_alter_facts(query, contexts, method_name, drop):
    docs = [dict(doc) for doc in contexts]
    families = [_title_stem(doc.get("title")) for doc in docs]

    # 1. Collect every number with its signature.
    entries = {}
    for i, doc in enumerate(docs):
        for fact in _fact_signatures(doc.get("text") or ""):
            entries.setdefault(fact["signature"], []).append((i, fact["value"], fact["start"], fact["end"]))

    # Which passages refer to each other (one mentions the other's title).
    links = {i: set() for i in range(len(docs))}
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if _passages_linked(docs[i], docs[j]):
                links[i].add(j)
                links[j].add(i)

    # 2. Cross-check: majority wins, one-against-one is only flagged.
    edits, outvoted, flagged = {}, set(), []
    for signature, items in entries.items():
        group = {item[0] for item in items}
        items = [item for item in items if links[item[0]] & group]  # keep only linked passages
        if not items:
            continue
        by_family = {}
        for i, value, start, end in items:
            by_family.setdefault(families[i], set()).add(value)
        if len(by_family) < 2 or any(len(values) > 1 for values in by_family.values()):
            continue  # nothing to cross-check, or sibling passages that legitimately differ
        votes = {}
        for family, values in by_family.items():
            votes.setdefault(next(iter(values)), set()).add(family)
        if len(votes) == 1:
            continue  # all passages agree
        ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
        top_value, top_families = ranked[0]
        if len(top_families) >= 2 and len(top_families) > len(ranked[1][1]):
            for i, value, start, end in items:
                if value != top_value:
                    outvoted.add(i)
                    edits.setdefault(i, []).append((start, end, value, top_value))
        else:
            values_by_title = {}
            for i, value, start, end in items:
                values_by_title.setdefault(value, []).append(docs[i].get("title", "untitled"))
            flagged.append({"signature": signature, "values": values_by_title})

    # 3. Impossible dates / reversed ranges inside a single passage.
    implausible = {}
    for i, doc in enumerate(docs):
        problems = _implausible_facts(doc.get("text") or "")
        if problems:
            implausible[i] = problems

    # 4. Apply: drop mode removes passages, correct mode rewrites the out-voted figure.
    kept, corrected, dropped_titles = [], [], []
    for i, doc in enumerate(docs):
        title = doc.get("title", "untitled")
        if drop and (i in outvoted or i in implausible):
            dropped_titles.append(title)
            continue
        if not drop and i in edits:
            text = doc.get("text") or ""
            for start, end, old, new in sorted(edits[i], key=lambda e: -e[0]):
                text = text[:start] + new + text[end:]
                corrected.append({"title": title, "original_value": old, "replacement": new})
            doc["text"] = text
        kept.append(doc)

    implausible_list = [{"title": docs[i].get("title", "untitled"), "problems": p} for i, p in implausible.items()]
    changed = bool(corrected or dropped_titles)
    if changed:
        parts = []
        if corrected:
            parts.append("corrected " + str(len(corrected)) + " figure(s) using agreement across passages")
        if dropped_titles:
            parts.append("dropped " + str(len(dropped_titles)) + " passage(s): " + ", ".join(dropped_titles))
        summary = "Altered-facts repair " + "; ".join(parts) + "."
    elif flagged or implausible_list:
        summary = ("No change. Flagged " + str(len(flagged)) + " unresolved figure conflict(s) and "
                   + str(len(implausible_list)) + " passage(s) with impossible dates.")
    else:
        summary = "No cross-checkable figures disagreed; contexts unchanged."

    record = {
        "method": method_name,
        "status": "applied" if changed else "skipped",
        "summary": summary,
        "details": {
            "corrected": corrected,
            "dropped_titles": dropped_titles,
            "flagged": flagged,
            "implausible": implausible_list,
            "remaining_count": len(kept),
        },
    }
    return query, kept, record


@register_repair("fix_alter_facts")
def repair_fix_alter_facts(query, contexts):
    """Cross-checks numbers across passages; corrects a figure the other passages contradict and flags the rest."""
    return _fix_alter_facts(query, contexts, "fix_alter_facts", drop=False)


@register_repair("fix_alter_facts_drop")
def repair_fix_alter_facts_drop(query, contexts):
    """Cross-checks numbers across passages; drops the passage whose figure is out-voted or has an impossible date."""
    return _fix_alter_facts(query, contexts, "fix_alter_facts_drop", drop=True)
