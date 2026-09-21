"""Online serving: the champion model behind an HTTP API.

Reads the clean layer from disk and reuses the ml feature and registry code, so online
and batch predictions cannot drift apart. Never imports the ETL package.
"""
