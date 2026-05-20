"""Launch the FastAPI app + static React UI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import uvicorn  # noqa: E402

from rag_turkish_law.config import load_config  # noqa: E402


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    cfg = load_config()
    uvicorn.run(
        "rag_turkish_law.api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
