"""
repair methods - trys to fix a broken query/context before we sedn it to the ai
each repair funciton gets added to REPAIR_REGISTRY under a name, then apply_repair_methods
jsut looks up whichver ones were askd for and runs them one after antoher
"""

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