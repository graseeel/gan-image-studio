from pathlib import Path

MIGRATION = Path("supabase/migrations/20260618220129_init_gan_image_studio_schema.sql")
CONFIG = Path("supabase/config.toml")

TABLES = [
    "profiles",
    "experiments",
    "experiment_metrics",
    "model_checkpoints",
    "generations",
    "generation_favorites",
    "evaluation_reports",
]

BUCKETS = [
    "generated-images",
    "training-samples",
    "model-checkpoints",
    "evaluation-assets",
]


def test_migration_enables_rls_for_all_public_tables() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    for table in TABLES:
        assert f"create table public.{table}" in sql
        assert f"alter table public.{table} enable row level security" in sql


def test_migration_has_explicit_grants_for_current_supabase_defaults() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "grant usage on schema public to anon, authenticated, service_role" in sql
    for table in TABLES:
        assert f"public.{table}" in sql
        assert "grant " in sql


def test_storage_buckets_are_configured_and_seeded() -> None:
    config = CONFIG.read_text(encoding="utf-8")
    sql = MIGRATION.read_text(encoding="utf-8")

    for bucket in BUCKETS:
        assert f"[storage.buckets.{bucket}]" in config
        assert f"'{bucket}'" in sql


def test_checkpoint_upload_is_backend_only() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")

    assert "model-checkpoints" in sql
    assert "users upload their generated images" in sql
    assert "upload their model-checkpoints" not in sql
