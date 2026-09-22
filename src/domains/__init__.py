"""One package per domain. Each exposes `adapter()`, which the core finds by name.

A domain is everything the core cannot know: its sources, its contracts, how its raw
tables are cleaned and what context an item may see. See `mlops_core.adapter`.
"""
