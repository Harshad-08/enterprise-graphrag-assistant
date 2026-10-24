"""Evaluation endpoints (synchronous: the synthetic dataset is small)."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import container_dependency
from app.evaluation.runner import run_evaluation
from app.models.schemas import EvaluationRequest
from app.services.container import Container

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.post("/run")
def run(request: EvaluationRequest, container: Container = Depends(container_dependency)) -> dict:
    try:
        return run_evaluation(
            container,
            configurations=request.configurations,
            include_ragas=request.include_ragas,
            limit=request.limit,
            categories=request.categories,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/latest")
def latest(container: Container = Depends(container_dependency)) -> dict:
    path = container.settings.eval_results_dir / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No evaluation has been run yet.")
    return json.loads(path.read_text())
