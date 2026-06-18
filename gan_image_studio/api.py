from __future__ import annotations

from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gan_image_studio.inference import (
    generate_images,
    interpolate_between_seeds,
    load_generator_from_checkpoint,
)
from gan_image_studio.plotting import save_image_grid


class GenerateRequest(BaseModel):
    checkpoint_path: str
    seed: int = 42
    count: int = Field(default=16, ge=1, le=64)
    output_path: str = "outputs/api-generation.png"


class InterpolationRequest(BaseModel):
    checkpoint_path: str
    start_seed: int
    end_seed: int
    steps: int = Field(default=8, ge=2, le=64)
    output_path: str = "outputs/api-interpolation.png"


def create_app() -> FastAPI:
    app = FastAPI(title="GAN Image Studio API")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, str]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = load_generator_from_checkpoint(Path(request.checkpoint_path), device)
            images = generate_images(
                generator,
                seed=request.seed,
                count=request.count,
                device=device,
            )
            output = save_image_grid(images, Path(request.output_path))
            return {"path": str(output)}
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/interpolate")
    def interpolate(request: InterpolationRequest) -> dict[str, str]:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = load_generator_from_checkpoint(Path(request.checkpoint_path), device)
            images = interpolate_between_seeds(
                generator,
                start_seed=request.start_seed,
                end_seed=request.end_seed,
                steps=request.steps,
                device=device,
            )
            output = save_image_grid(images, Path(request.output_path), nrow=request.steps)
            return {"path": str(output)}
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
