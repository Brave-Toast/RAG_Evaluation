"""Composable query/context perturbations. No retrieval, model calls, or input mutation."""

import copy
import hashlib
import random
import re
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


Method: TypeAlias = Literal[
    "spelling", "rephrase", "conflict", "remove_evidence",
    "irrelevant", "alter_facts", "shuffle", "truncate", "emoji", "negation_flip", "entity_substitution",
]
METHOD_ORDER = (
    "rephrase", "spelling", "emoji", "negation_flip", "entity_substitution", "conflict", "remove_evidence",
    "alter_facts", "truncate", "irrelevant", "shuffle",
)
PRE_RETRIEVAL = {"rephrase", "spelling", "emoji", "conflict", "negation_flip", "entity_substitution"}


class PerturbationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    remove_count: int = Field(default=1, ge=1, le=5)
    irrelevant_count: int = Field(default=1, ge=1, le=5)
    fact_count: int = Field(default=1, ge=1, le=20)
    retain_percent: int = Field(default=50, ge=1, le=100)
    emoji_style: Literal["insert", "substitute", "mixed"] = "mixed"
    emoji_count: int = Field(default=3, ge=1, le=20)
    emoji_position: Literal["prefix", "suffix", "between_words"] = "suffix"


def ordered_methods(methods):
    return [method for method in METHOD_ORDER if method in methods]


def method_rng(seed: int, method: str) -> random.Random:
    """Independent streams keep a method's choices stable across comparison modes."""
    digest = hashlib.sha256(f"{seed}:{method}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def change_record(method, stage, changed, summary, details=None):
    return {
        "method": method, "stage": stage,
        "status": "applied" if changed else "skipped",
        "summary": summary, "details": details or [],
    }


# Complete Unicode sequences are kept intact (a keycap digit is multiple code points).
KEYCAP_DIGITS = {digit: digit + "\ufe0f\u20e3" for digit in "0123456789"}
EMOJI_INSERTIONS = ("🤔", "📚", "🔎", "✨", "🧩", "📅")
EMOJI_LETTERS = {"a": "🅰️", "b": "🅱️", "i": "ℹ️", "m": "Ⓜ️", "o": "🅾️", "p": "🅿️"}
EMOJI_WORDS = {
    "before": "b4️⃣", "great": "gr8️⃣", "later": "l8️⃣r",
    "today": "2️⃣day", "tonight": "2️⃣night", "tomorrow": "2️⃣morrow",
    "someone": "some1️⃣", "anyone": "any1️⃣", "everyone": "every1️⃣",
    "first": "1️⃣st", "second": "2️⃣nd", "third": "3️⃣rd", "fourth": "4️⃣th",
    "for": "4️⃣", "to": "2️⃣", "too": "2️⃣",
    "be": "🐝", "bee": "🐝", "see": "👀", "eye": "👁️", "eyes": "👀",
    "love": "❤️", "heart": "❤️", "book": "📖", "books": "📚", "star": "⭐",
    "one": "1️⃣", "two": "2️⃣", "three": "3️⃣", "four": "4️⃣", "five": "5️⃣",
    "six": "6️⃣", "seven": "7️⃣", "eight": "8️⃣", "nine": "9️⃣", "zero": "0️⃣",
}
# Preserve number/date separators and leading zeros. Do not match part of an
# alphanumeric ID, existing keycap, or Unicode word.
EMOJI_TOKENS = re.compile(
    r"(?<![\w\ufe0f\u20e3])(?:[0-9]+(?:[./:\-][0-9]+)*|[A-Za-z]+(?:['’][A-Za-z]+)?)(?![\w\ufe0f\u20e3])"
)


def perturb_query_with_emojis(text, settings, seed):
    """Insert symbols or encode words/dates, with a seeded edit budget and audit trail."""
    rng = method_rng(seed, "emoji")
    query = text
    details = []
    candidates = []
    if text.strip() and settings.emoji_style != "insert":
        for match in EMOJI_TOKENS.finditer(text):
            token = match.group()
            if token[0].isdigit():
                kind = "number_or_date"
            elif token.lower() in EMOJI_WORDS:
                kind = "word_encoding"
            elif any(char.lower() in EMOJI_LETTERS for char in token):
                kind = "letter_encoding"
            else:
                continue
            candidates.append((match, kind))
        budget = settings.emoji_count if settings.emoji_style == "substitute" else max(1, settings.emoji_count // 2)
        selected = sorted(rng.sample(candidates, min(budget, len(candidates))), key=lambda item: item[0].start())
        for match, kind in selected:
            token = match.group()
            if kind == "number_or_date":
                replacement = "".join(KEYCAP_DIGITS.get(char, char) for char in token)
            elif kind == "word_encoding":
                replacement = EMOJI_WORDS[token.lower()]
            else:
                index = rng.choice([i for i, char in enumerate(token) if char.lower() in EMOJI_LETTERS])
                replacement = token[:index] + EMOJI_LETTERS[token[index].lower()] + token[index + 1:]
            details.append({"kind": kind, "offset": match.start(), "offset_basis": "input_query",
                            "original": token, "replacement": replacement})
        for edit in reversed(details):
            start = edit["offset"]
            query = query[:start] + edit["replacement"] + query[start + len(edit["original"]):]

    encoded_count = len(details)
    insert_count = settings.emoji_count - encoded_count if settings.emoji_style != "substitute" and text.strip() else 0
    additions = {}
    # Insert at token boundaries, never inside emoji sequences or words.
    boundaries = [match.end() for match in re.finditer(r"\S+(?=\s+\S)", query)]
    for _ in range(insert_count):
        symbol = rng.choice(EMOJI_INSERTIONS)
        position = settings.emoji_position
        if position == "prefix":
            offset, inserted = 0, symbol + " "
        elif position == "between_words" and boundaries:
            offset, inserted = rng.choice(boundaries), " " + symbol
        else:
            offset, inserted = len(query), " " + symbol
            position = "suffix"
        additions[offset] = additions.get(offset, "") + inserted
        details.append({"kind": "insertion", "offset": offset, "offset_basis": "after_encoding",
                        "original": "", "replacement": inserted, "emoji": symbol, "position": position})
    for offset in sorted(additions, reverse=True):
        query = query[:offset] + additions[offset] + query[offset:]

    summary = f"Encoded {encoded_count} words/numbers/dates and inserted {insert_count} emojis (budget {settings.emoji_count})."
    if not details:
        summary += " No eligible text was available; question unchanged."
    if encoded_count:
        summary += " Word encodings are intended readings; equivalent meaning is not guaranteed."
    if insert_count and settings.emoji_position == "between_words" and not boundaries:
        summary += " No between-word boundary; inserted at the end."
    record = change_record("emoji", "pre_retrieval", query != text, summary, details)
    record.update({"input_query": text, "output_query": query})
    return query, record


def distractor_pool(dataset, question_id, original_contexts, query=""):
    """Sample other questions' passages; exclude shared titles/text and duplicates."""
    titles = {d["title"].strip().casefold() for d in original_contexts}
    texts = {d["text"].strip().casefold() for d in original_contexts}
    pool = []
    for row in dataset:
        if (question_id is not None and str(row.get("id")) == str(question_id)) or row.get("question") == query:
            continue
        for doc in row.get("contexts", []):
            title, text = doc["title"].strip().casefold(), doc["text"].strip().casefold()
            if not text or title in titles or text in texts or "synthetic" in title:
                continue
            titles.add(title)
            texts.add(text)
            pool.append({"title": doc["title"], "text": doc["text"],
                         "score": None, "source_question_id": row.get("id")})
    return pool


def apply_context_methods(contexts, methods, settings, seed, supporting_titles=None, distractors=None):
    """Return final context and an audit trail, preserving the supplied retrieval output."""
    docs = copy.deepcopy(contexts)
    changes = []
    for method in ordered_methods(methods):
        if method in PRE_RETRIEVAL:
            continue
        rng = method_rng(seed, method)
        details = []

        if method == "remove_evidence":
            annotated = supporting_titles is not None
            candidates = [i for i, doc in enumerate(docs)
                          if not annotated or doc["title"] in supporting_titles]
            removed = candidates[:settings.remove_count]
            details = [{"index": i, "title": docs[i]["title"]} for i in removed]
            docs = [doc for i, doc in enumerate(docs) if i not in removed]
            basis = "annotated supporting passages" if annotated else "highest-ranked passages (support annotations unavailable)"
            summary = f"Removed {len(removed)} of {settings.remove_count} requested {basis}."
            if not removed:
                summary += " No matching passages were available."

        elif method == "alter_facts":
            # Bounded integer tokens include years/day numbers, leaving names and decimals intact.
            pattern = re.compile(r"(?<![\w.,])[0-9]{1,9}(?![\w.,])")
            matches = [(i, match) for i, doc in enumerate(docs) for match in pattern.finditer(doc["text"])]
            chosen = rng.sample(matches, min(settings.fact_count, len(matches)))
            for i, match in sorted(chosen, key=lambda item: (item[0], item[1].start()), reverse=True):
                original = match.group()
                replacement = str(int(original) + rng.choice([1, 2, 5, 10]))
                doc = docs[i]
                doc["text"] = doc["text"][:match.start()] + replacement + doc["text"][match.end():]
                doc["score"] = None  # Retrieval scores describe the original passage only.
                details.append({"index": i, "title": doc["title"], "offset": match.start(),
                                "original": original, "replacement": replacement})
            summary = f"Altered {len(details)} of {settings.fact_count} requested number/date tokens."
            if not details:
                summary += " No replaceable integer tokens were available."

        elif method == "truncate":
            for i, doc in enumerate(docs):
                text = doc["text"]
                limit = len(text) * settings.retain_percent // 100
                ends = [m.end() for m in re.finditer(r"\S+", text) if m.end() <= limit]
                retained = text[:ends[-1]] if ends else ""
                if settings.retain_percent == 100:
                    retained = text
                if retained != text:
                    details.append({"index": i, "title": doc["title"],
                                    "original_characters": len(text), "retained_characters": len(retained)})
                    doc["text"], doc["score"] = retained, None
            summary = f"Truncated {len(details)} passages to at most {settings.retain_percent}% of characters at word boundaries."
            if not details:
                summary += " No text changed."

        elif method == "irrelevant":
            titles = {d["title"].strip().casefold() for d in docs}
            texts = {d["text"].strip().casefold() for d in docs}
            available = []
            for doc in distractors or []:
                title, text = doc["title"].strip().casefold(), doc["text"].strip().casefold()
                if text and title not in titles and text not in texts:
                    available.append(doc)
                    titles.add(title)
                    texts.add(text)
            for doc in rng.sample(available, min(settings.irrelevant_count, len(available))):
                docs.append({"title": doc["title"], "text": doc["text"], "score": None})
                details.append({"title": doc["title"], "source_question_id": doc.get("source_question_id")})
            summary = f"Added {len(details)} of {settings.irrelevant_count} requested sampled distractors; irrelevance is not guaranteed."
            if not details:
                summary += " No distinct passages were available."

        elif method == "shuffle":
            order = list(range(len(docs)))
            rng.shuffle(order)
            # Ensure a visible change when distinct passages are available.
            if len(docs) > 1 and [docs[i] for i in order] == docs:
                order = order[1:] + order[:1]
            shuffled = [docs[i] for i in order]
            if shuffled != docs:
                details = [{"original_index": i, "new_index": j, "title": docs[i]["title"]}
                           for j, i in enumerate(order)]
                docs = shuffled
            summary = "Shuffled passage order." if details else "No change: fewer than two distinct passages available."

        changes.append(change_record(method, "post_retrieval", bool(details), summary, details))
    return docs, changes
