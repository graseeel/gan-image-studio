from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch

from gan_image_studio.checkpoints import load_checkpoint
from gan_image_studio.inference import (
    generate_images,
    interpolate_between_seeds,
    load_generator_from_checkpoint,
)
from gan_image_studio.plotting import save_image_grid
from gan_image_studio.supabase_client import SupabaseGateway, SupabaseSettings
from gan_image_studio.utils import ensure_directory


def _checkpoint_choices() -> list[str]:
    checkpoint_dir = Path(os.getenv("GAN_STUDIO_CHECKPOINT_DIR", "checkpoints"))
    return [str(path) for path in sorted(checkpoint_dir.glob("*.pt"))]


def _gateway() -> SupabaseGateway | None:
    url = os.getenv("SUPABASE_URL", "")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not anon_key:
        return None
    return SupabaseGateway(
        SupabaseSettings(
            url=url,
            anon_key=anon_key,
            service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY") or None,
        )
    )


def _generate(checkpoint_path: str, seed: int, count: int) -> str:
    if not checkpoint_path:
        raise gr.Error("Select a checkpoint first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator_from_checkpoint(Path(checkpoint_path), device)
    seed_value = int(seed)
    count_value = int(count)
    images = generate_images(generator, seed=seed_value, count=count_value, device=device)
    output_dir = ensure_directory(Path(os.getenv("GAN_STUDIO_OUTPUT_DIR", "outputs")))
    output_path = output_dir / f"generation-seed-{seed_value}.png"
    save_image_grid(images, output_path)
    return str(output_path)


def _interpolate(checkpoint_path: str, start_seed: int, end_seed: int, steps: int) -> str:
    if not checkpoint_path:
        raise gr.Error("Select a checkpoint first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator_from_checkpoint(Path(checkpoint_path), device)
    images = interpolate_between_seeds(
        generator,
        start_seed=int(start_seed),
        end_seed=int(end_seed),
        steps=int(steps),
        device=device,
    )
    output_dir = ensure_directory(Path(os.getenv("GAN_STUDIO_OUTPUT_DIR", "outputs")))
    output_path = output_dir / f"interpolation-{int(start_seed)}-{int(end_seed)}.png"
    save_image_grid(images, output_path, nrow=int(steps))
    return str(output_path)


def _sign_in(email: str, password: str) -> str:
    gateway = _gateway()
    if gateway is None:
        raise gr.Error("Supabase environment variables are not configured.")
    gateway.sign_in(email, password)
    return "signed in"


def _history() -> list[list[str]]:
    gateway = _gateway()
    if gateway is None:
        return []
    rows = gateway.list_generation_history()
    return [
        [
            str(row.get("id", "")),
            str(row.get("seed", "")),
            str(row.get("image_count", "")),
            str(row.get("storage_path", "")),
            str(row.get("created_at", "")),
        ]
        for row in rows
    ]


def _save_generation(grid_path: str, seed: int, count: int) -> str:
    if not grid_path:
        raise gr.Error("Generate a grid first.")
    gateway = _gateway()
    if gateway is None:
        raise gr.Error("Supabase environment variables are not configured.")
    user_id = gateway.current_user_id()
    remote_path = gateway.upload_generated_grid(Path(grid_path), user_id)
    row = gateway.save_generation_record(
        user_id=user_id,
        storage_path=remote_path,
        seed=int(seed),
        image_count=int(count),
        checkpoint_id=None,
        experiment_id=None,
        metadata={"source": "gradio"},
    )
    return str(row["id"])


def _favorite_generation(generation_id: str) -> str:
    if not generation_id:
        raise gr.Error("Provide a generation id.")
    gateway = _gateway()
    if gateway is None:
        raise gr.Error("Supabase environment variables are not configured.")
    gateway.favorite_generation(generation_id)
    return "favorited"


def _checkpoint_info(checkpoint_path: str) -> dict[str, Any]:
    if not checkpoint_path:
        raise gr.Error("Select a checkpoint first.")
    checkpoint = load_checkpoint(Path(checkpoint_path))
    return {
        "epoch": checkpoint["epoch"],
        "step": checkpoint["step"],
        "model_config": checkpoint["model_config"],
        "metrics": checkpoint["metrics"],
    }


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="GAN Image Studio") as demo:
        gr.Markdown("# GAN Image Studio")
        with gr.Row():
            email = gr.Textbox(label="Email")
            password = gr.Textbox(label="Password", type="password")
            auth_status = gr.Textbox(label="Auth", interactive=False)
        gr.Button("Sign in").click(_sign_in, inputs=[email, password], outputs=auth_status)

        with gr.Tab("Generate"):
            checkpoint = gr.Dropdown(label="Checkpoint", choices=_checkpoint_choices())
            seed = gr.Number(label="Seed", value=42, precision=0)
            count = gr.Slider(label="Quantity", minimum=1, maximum=64, step=1, value=16)
            output = gr.Image(label="Grid", type="filepath")
            download = gr.File(label="Download grid")
            saved_generation = gr.Textbox(label="Saved generation id", interactive=False)
            gr.Button("Generate").click(_generate, inputs=[checkpoint, seed, count], outputs=output)
            gr.Button("Save generation").click(
                _save_generation,
                inputs=[output, seed, count],
                outputs=saved_generation,
            )
            output.change(lambda path: path, inputs=output, outputs=download)

        with gr.Tab("Interpolate"):
            interp_checkpoint = gr.Dropdown(label="Checkpoint", choices=_checkpoint_choices())
            start_seed = gr.Number(label="Start seed", value=1, precision=0)
            end_seed = gr.Number(label="End seed", value=2, precision=0)
            steps = gr.Slider(label="Steps", minimum=2, maximum=32, step=1, value=8)
            interp_output = gr.Image(label="Grid", type="filepath")
            interp_download = gr.File(label="Download grid")
            gr.Button("Interpolate").click(
                _interpolate,
                inputs=[interp_checkpoint, start_seed, end_seed, steps],
                outputs=interp_output,
            )
            interp_output.change(lambda path: path, inputs=interp_output, outputs=interp_download)

        with gr.Tab("History"):
            history = gr.Dataframe(
                headers=["id", "seed", "count", "path", "created_at"],
                interactive=False,
            )
            favorite_id = gr.Textbox(label="Generation id")
            favorite_status = gr.Textbox(label="Favorite", interactive=False)
            gr.Button("Refresh").click(_history, outputs=history)
            gr.Button("Favorite").click(
                _favorite_generation,
                inputs=favorite_id,
                outputs=favorite_status,
            )

        with gr.Tab("Checkpoint"):
            info_checkpoint = gr.Dropdown(label="Checkpoint", choices=_checkpoint_choices())
            checkpoint_info = gr.JSON(label="Configuration and metrics")
            gr.Button("Inspect").click(
                _checkpoint_info,
                inputs=info_checkpoint,
                outputs=checkpoint_info,
            )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
