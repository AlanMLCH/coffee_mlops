"""Model pipeline: clean tables -> features -> trained model -> predictions.

Features depend on the model spec (target, leakage, chosen columns), so they belong
here and not in the ETL. This package reads the clean layer from disk and never
imports from `mlops_core.data`.
"""
