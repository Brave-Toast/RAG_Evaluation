
from backend.repairs import (
    repair_fix_alter_facts,
    repair_fix_alter_facts_drop,
)

query = "When was the Eiffel Tower completed?"

# Two independent sources agree; one has an altered year.



contexts = [
    {
        "title": "Eiffel Tower",
        "text": "The Eiffel Tower was completed in 1889 according to historical records. "
                "Paris Landmark and Tower History discuss its construction.",
        "score": None,
    },
    {
        "title": "Paris Landmark",
        "text": "The Eiffel Tower was completed in 1889 according to historical records. "
                "Eiffel Tower and Tower History describe the same landmark.",
        "score": None,
    },
    {
        "title": "Tower History",
        "text": "The Eiffel Tower was completed in 1999 according to historical records. "
                "Eiffel Tower and Paris Landmark describe its construction.",
        "score": None,
    },
]




for name, repair in [
    ("fix_alter_facts", repair_fix_alter_facts),
    ("fix_alter_facts_drop", repair_fix_alter_facts_drop),
]:
    _, repaired, record = repair(query, contexts)

    print(f"\n=== {name} ===")
    print("Status:", record["status"])
    print("Summary:", record["summary"])
    print("Details:", record["details"])
    print("Resulting passages:")
    for doc in repaired:
        print(f"  {doc['title']}: {doc['text']}")


from backend.repairs import (
    _fact_signatures,
    _passages_linked,
    _title_stem,
)

print("\n=== INTERNAL DIAGNOSTICS ===")

for i, doc in enumerate(contexts):
    print(f"\nDocument {i}: {doc['title']}")
    print("Family:", _title_stem(doc["title"]))
    print("Facts:", _fact_signatures(doc["text"]))

print("\nDocument links:")
for i in range(len(contexts)):
    for j in range(i + 1, len(contexts)):
        print(
            contexts[i]["title"],
            "<->",
            contexts[j]["title"],
            _passages_linked(contexts[i], contexts[j]),
        )
