from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import uvicorn

from gan_image_studio.api import app
from gan_image_studio.config import ModelConfig, TrainingConfig
from gan_image_studio.data import inspect_image_folder
from gan_image_studio.inference import generate_images, load_generator_from_checkpoint
from gan_image_studio.plotting import save_image_grid
from gan_image_studio.training import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="gan-studio")
    subcommands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subcommands.add_parser("inspect-data")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.add_argument("--image-size", type=int, default=32)

    train_parser = subcommands.add_parser("train")
    train_parser.add_argument("--dataset", choices=["cifar10", "folder"], default="cifar10")
    train_parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    train_parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--latent-dim", type=int, default=100)
    train_parser.add_argument("--image-size", type=int, default=32)
    train_parser.add_argument("--generator-features", type=int, default=64)
    train_parser.add_argument("--discriminator-features", type=int, default=64)
    train_parser.add_argument("--save-every-steps", type=int, default=500)
    train_parser.add_argument("--sample-every-steps", type=int, default=250)
    train_parser.add_argument("--max-batches", type=int)
    train_parser.add_argument("--num-workers", type=int, default=2)
    train_parser.add_argument("--resume-checkpoint", type=Path)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--quick-cpu", action="store_true")

    generate_parser = subcommands.add_parser("generate")
    generate_parser.add_argument("checkpoint", type=Path)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--count", type=int, default=16)
    generate_parser.add_argument("--output", type=Path, default=Path("outputs/generated.png"))

    subcommands.add_parser("ui")
    api_parser = subcommands.add_parser("api")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "inspect-data":
        inspection = inspect_image_folder(args.path, args.image_size)
        print(
            json.dumps(
                {
                    "root": str(inspection.root),
                    "valid_count": inspection.valid_count,
                    "issues": [
                        {"path": str(issue.path), "reason": issue.reason}
                        for issue in inspection.issues
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "train":
        if args.quick_cpu:
            config = TrainingConfig.quick_cpu()
        else:
            config = TrainingConfig(
                dataset=args.dataset,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                save_every_steps=args.save_every_steps,
                sample_every_steps=args.sample_every_steps,
                max_batches=args.max_batches,
                num_workers=args.num_workers,
                resume_checkpoint=args.resume_checkpoint,
                device=args.device,
                model=ModelConfig(
                    latent_dim=args.latent_dim,
                    image_size=args.image_size,
                    generator_features=args.generator_features,
                    discriminator_features=args.discriminator_features,
                ),
            )
        train(config)
    elif args.command == "generate":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = load_generator_from_checkpoint(args.checkpoint, device)
        images = generate_images(generator, seed=args.seed, count=args.count, device=device)
        save_image_grid(images, args.output)
        print(args.output)
    elif args.command == "ui":
        from gan_image_studio.app import main as ui_main

        ui_main()
    elif args.command == "api":
        uvicorn.run(app, host=args.host, port=args.port)
