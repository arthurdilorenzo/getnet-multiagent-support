"""Offline is the default even when the developer has a real local .env."""
import os

if os.environ.get("GETNET_LIVE_TESTS") != "1":
    os.environ["GEMINI_API_KEY"] = ""
