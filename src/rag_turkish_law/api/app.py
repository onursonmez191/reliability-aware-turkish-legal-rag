"""FastAPI app: JSON endpoints + static React prototype mount."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..config import get_repo_root, load_config
from .pipeline import run_pipeline
from .schemas import AskRequest, AskResponse

log = logging.getLogger(__name__)

app = FastAPI(
    title="Reliability-Aware Turkish Legal RAG",
    version="0.1.0",
    description=(
        "Educational RAG system. Outputs are informational and do not "
        "constitute professional legal advice."
    ),
)


def _setup_cors() -> None:
    cfg = load_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.api.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


_setup_cors()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return run_pipeline(req.question, req.mode, req.k)
    except FileNotFoundError as exc:
        log.exception("Index not found")
        raise HTTPException(
            status_code=503,
            detail=(
                "FAISS index or metadata file not found. Run "
                "`python scripts/build_index.py` first."
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface model/network errors to UI
        log.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def mount_static() -> None:
    cfg = load_config()
    static_dir = Path(cfg.api.static_dir)
    if not static_dir.is_absolute():
        static_dir = get_repo_root() / static_dir
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    else:
        log.warning("Static dir %s not found; UI will not be served.", static_dir)


mount_static()
