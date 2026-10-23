import os
import re
import sys
import json
import socket
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import TavilyClient

# 1. Automatically load worker/.env regardless of where the script is called from
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=ENV_PATH)

DATABASE_URL = os.getenv("DATABASE_URL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")

# 2. Centralized Nebius Token Factory Model Registry (Change once here to update everywhere)
NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
ULTRA_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"


def validate_env():
    """Ensures all required .env variables are loaded before running a pipeline step."""
    missing = [
        name for name, val in [
            ("DATABASE_URL", DATABASE_URL),
            ("TAVILY_API_KEY", TAVILY_API_KEY),
            ("NEBIUS_API_KEY", NEBIUS_API_KEY),
        ]
        if not val
    ]
    if missing:
        print(f"[ERROR] Missing environment variables in worker/.env: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def get_nebius_client() -> AsyncOpenAI:
    """Creates an AsyncOpenAI client bound to the current active event loop."""
    return AsyncOpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=NEBIUS_API_KEY,
        timeout=45.0,
        max_retries=3
    )


def get_tavily_client() -> TavilyClient:
    """Creates a Tavily search & extraction client."""
    return TavilyClient(api_key=TAVILY_API_KEY)


def normalize_and_verify_domain(raw_input: str) -> str:
    """
    Accepts any valid URL or domain format:
      - https://www.coca-cola.com/products?ref=123
      - http://coca-cola.com/
      - www.coca-cola.com
      - coca-cola.com
    Normalizes it to 'coca-cola.com' and checks DNS liveness.
    """
    cleaned = raw_input.strip().lower()
    if not cleaned:
        raise ValueError("URL or domain cannot be empty.")

    # Prepend scheme if missing so urlparse extracts netloc properly
    if "://" not in cleaned:
        parsed = urlparse(f"https://{cleaned}")
    else:
        parsed = urlparse(cleaned)

    host = parsed.netloc or parsed.path.split("/")[0]
    # Strip port numbers or @ credentials if present
    host = host.split("@")[-1].split(":")[0]

    if host.startswith("www."):
        host = host[4:]

    # Validate domain structure (allows hyphens like coca-cola.com and multi-part TLDs like co.uk)
    domain_regex = r"^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$"
    if not re.match(domain_regex, host):
        raise ValueError(f"Invalid URL or domain format: '{raw_input}'. Example: 'coca-cola.com' or 'https://coca-cola.com'.")

    # Verify the website exists on the live internet via DNS resolution
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        raise ValueError(f"Website '{host}' could not be reached or does not exist.")

    return host


def safe_parse_json(response) -> dict:
    """Extracts JSON safely from Nemotron responses without crashing on markdown blocks."""
    msg = response.choices[0].message
    raw_text = msg.content or getattr(msg, "reasoning_content", None)

    if not raw_text:
        return {}

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return {}
        return {}