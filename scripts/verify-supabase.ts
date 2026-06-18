import { readFileSync } from "node:fs";

const requiredTables = [
  "profiles",
  "experiments",
  "experiment_metrics",
  "model_checkpoints",
  "generations",
  "generation_favorites",
  "evaluation_reports",
];

const requiredBuckets = [
  "generated-images",
  "training-samples",
  "model-checkpoints",
  "evaluation-assets",
];

function assertIncludes(source: string, needle: string, label: string): void {
  if (!source.includes(needle)) {
    throw new Error(`Missing ${label}: ${needle}`);
  }
}

async function verifyRemoteRest(): Promise<void> {
  const url = process.env.SUPABASE_URL;
  const anonKey = process.env.SUPABASE_ANON_KEY;
  if (!url || !anonKey) {
    console.log("Remote REST check skipped: SUPABASE_URL or SUPABASE_ANON_KEY is missing.");
    return;
  }

  const response = await fetch(`${url}/rest/v1/experiments?select=id&limit=1`, {
    headers: {
      apikey: anonKey,
      Authorization: `Bearer ${anonKey}`,
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Remote REST verification failed: ${response.status} ${body}`);
  }
  console.log("Remote REST check passed for experiments.");
}

async function verifyManagementProject(): Promise<void> {
  const token = process.env.SUPABASE_ACCESS_TOKEN;
  const ref = process.env.SUPABASE_PROJECT_REF;
  if (!token || !ref) {
    console.log("Management API check skipped: token or project ref is missing.");
    return;
  }

  const response = await fetch(`https://api.supabase.com/v1/projects/${ref}`, {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Management API verification failed: ${response.status} ${body}`);
  }
  const project = (await response.json()) as Record<string, unknown>;
  console.log(`Management API check passed for project ref ${project.ref ?? ref}.`);
}

async function main(): Promise<void> {
  const migrationPath = process.env.SUPABASE_MIGRATION_PATH
    ?? "supabase/migrations/20260618220129_init_gan_image_studio_schema.sql";
  const migration = readFileSync(migrationPath, "utf8");
  const config = readFileSync("supabase/config.toml", "utf8");

  for (const table of requiredTables) {
    assertIncludes(migration, `create table public.${table}`, `table ${table}`);
    assertIncludes(migration, `alter table public.${table} enable row level security`, `RLS ${table}`);
  }

  for (const table of requiredTables) {
    assertIncludes(migration, `grant`, `explicit grants for ${table}`);
  }

  for (const bucket of requiredBuckets) {
    assertIncludes(config, `[storage.buckets.${bucket}]`, `bucket config ${bucket}`);
    assertIncludes(migration, `'${bucket}'`, `bucket seed ${bucket}`);
  }

  assertIncludes(migration, "schema app_private", "private schema for privileged functions");
  assertIncludes(migration, "model-checkpoints", "private checkpoint bucket");
  await verifyRemoteRest();
  await verifyManagementProject();
  console.log("Supabase verification checks passed.");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
