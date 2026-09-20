"""ETL pipeline: external sources -> raw -> validated -> canonical clean tables.

Its product is the clean layer, which is model-agnostic: analytics, the agent's SQL
and any model consume it. This package never imports from `coffee_mlops.ml`.
"""
