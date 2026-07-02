"""
config.py — Centralized configuration for the Vera bot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TEAM_NAME: str = os.getenv("TEAM_NAME", "Ansh Sharma")
CONTACT_EMAIL: str = os.getenv("CONTACT_EMAIL", "ansh.sharma@kalvium.community")
BOT_VERSION: str = "1.0.0"
SUBMITTED_AT: str = "2026-07-02T20:00:00Z"

if not GEMINI_API_KEY:
    import sys
    print("[FATAL] GEMINI_API_KEY is not set. Set it in your environment or .env file.")
    sys.exit(1)
