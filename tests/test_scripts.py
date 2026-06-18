from pathlib import Path


def test_provision_script_refuses_existing_project_ref() -> None:
    source = Path("scripts/provision-supabase.ts").read_text(encoding="utf-8")

    assert "Refusing to provision with SUPABASE_PROJECT_REF already set" in source
    assert "randomBytes" in source
    assert "db_pass" in source
