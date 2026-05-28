"""FastAPI app: JSON endpoints + static React prototype mount."""

from __future__ import annotations

import logging
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import get_repo_root, load_config
from .models import (
    ModelConfigError,
    OllamaRuntimeError,
    get_models_state,
    load_ollama_model,
    unload_ollama_model,
    validate_model_name,
)
from .pipeline import run_pipeline, run_pipeline_stream
from .schemas import AskRequest, AskResponse, ModelActionRequest, ModelActionResponse, ModelsResponse

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


@app.get("/api/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    return ModelsResponse(**get_models_state())


@app.post("/api/models/load", response_model=ModelActionResponse)
def load_model(req: ModelActionRequest) -> ModelActionResponse:
    try:
        state = load_ollama_model(req.model, req.keep_alive)
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelActionResponse(
        model=req.model,
        status="loaded",
        message=f"{req.model} loaded with keep_alive={state['keep_alive']}.",
        state=ModelsResponse(**state),
    )


@app.post("/api/models/unload", response_model=ModelActionResponse)
def unload_model(req: ModelActionRequest) -> ModelActionResponse:
    try:
        state = unload_ollama_model(req.model)
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OllamaRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ModelActionResponse(
        model=req.model,
        status="unloaded",
        message=f"{req.model} unloaded.",
        state=ModelsResponse(**state),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        validate_model_name(req.model)
        validate_model_name(req.verifier_model)
        return run_pipeline(req.question, req.mode, req.k, model=req.model, verifier_model=req.verifier_model)
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


def _sse(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    try:
        validate_model_name(req.model)
        validate_model_name(req.verifier_model)
    except ModelConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def events():
        try:
            for item in run_pipeline_stream(
                req.question,
                req.mode,
                req.k,
                model=req.model,
                verifier_model=req.verifier_model,
            ):
                yield _sse(item["event"], item["data"])
        except Exception as exc:  # noqa: BLE001 - stream errors must reach the UI as events
            log.exception("Streaming pipeline error")
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
