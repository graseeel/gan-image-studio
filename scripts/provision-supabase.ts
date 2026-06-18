import { execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";

type Region =
  | "us-west-1"
  | "us-east-1"
  | "us-east-2"
  | "ca-central-1"
  | "eu-west-1"
  | "eu-west-2"
  | "eu-west-3"
  | "eu-central-1"
  | "eu-central-2"
  | "eu-north-1"
  | "ap-south-1"
  | "ap-southeast-1"
  | "ap-northeast-1"
  | "ap-northeast-2"
  | "ap-southeast-2"
  | "sa-east-1";

const managementApi = "https://api.supabase.com/v1";
const regions: readonly Region[] = [
  "us-west-1",
  "us-east-1",
  "us-east-2",
  "ca-central-1",
  "eu-west-1",
  "eu-west-2",
  "eu-west-3",
  "eu-central-1",
  "eu-central-2",
  "eu-north-1",
  "ap-south-1",
  "ap-southeast-1",
  "ap-northeast-1",
  "ap-northeast-2",
  "ap-southeast-2",
  "sa-east-1",
];

function requiredEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`${name} is required`);
  }
  return value;
}

function regionFromEnv(value: string | undefined): Region {
  const region = value ?? "sa-east-1";
  if (!regions.includes(region as Region)) {
    throw new Error(`Unsupported Supabase region: ${region}`);
  }
  return region as Region;
}

async function request<T>(path: string, init: RequestInit, token: string): Promise<T> {
  const response = await fetch(`${managementApi}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Supabase API ${response.status} ${response.statusText}: ${body}`);
  }
  return (await response.json()) as T;
}

async function waitForProject(ref: string, token: string): Promise<void> {
  for (let attempt = 0; attempt < 60; attempt += 1) {
    const project = await request<Record<string, unknown>>(
      `/projects/${ref}`,
      { method: "GET" },
      token,
    );
    const database = project.database as Record<string, unknown> | undefined;
    const status = String(project.status ?? database?.status ?? "");
    if (["ACTIVE_HEALTHY", "ACTIVE", "healthy", "RUNNING"].includes(status)) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }
  throw new Error(`Supabase project ${ref} did not become ready in time`);
}

function runSupabaseCli(args: string[]): void {
  execFileSync("supabase", args, {
    stdio: "inherit",
    env: process.env,
  });
}

async function main(): Promise<void> {
  if (process.env.SUPABASE_PROJECT_REF) {
    throw new Error(
      "Refusing to provision with SUPABASE_PROJECT_REF already set. This script creates a new project only.",
    );
  }

  const token = requiredEnv("SUPABASE_ACCESS_TOKEN");
  const organizationId = requiredEnv("SUPABASE_ORGANIZATION_ID");
  const dbPassword = requiredEnv("SUPABASE_DB_PASSWORD");
  const region = regionFromEnv(process.env.SUPABASE_REGION);
  const prefix = process.env.SUPABASE_PROJECT_NAME_PREFIX ?? "gan-image-studio";
  const name = `${prefix}-${randomBytes(4).toString("hex")}`;

  const existingProjects = await request<Array<{ name?: string }>>(
    "/projects",
    { method: "GET" },
    token,
  );
  if (existingProjects.some((project) => project.name === name)) {
    throw new Error(`Generated project name already exists: ${name}`);
  }

  const created = await request<Record<string, unknown>>(
    "/projects",
    {
      method: "POST",
      body: JSON.stringify({
        organization_id: organizationId,
        name,
        region,
        db_pass: dbPassword,
      }),
    },
    token,
  );

  const ref = String(created.ref ?? created.id ?? created.project_ref ?? "");
  if (!ref) {
    throw new Error("Supabase API did not return a project ref");
  }

  console.log(`Created Supabase project ${name}`);
  console.log(`Project ref: ${ref}`);
  await waitForProject(ref, token);

  process.env.SUPABASE_PROJECT_REF = ref;
  runSupabaseCli(["link", "--project-ref", ref, "--password", dbPassword]);
  runSupabaseCli(["db", "push", "--yes"]);
  runSupabaseCli(["functions", "deploy", "checkpoint-upload", "--project-ref", ref]);

  console.log("Provisioning complete. Add these non-secret values to your local .env:");
  console.log(`SUPABASE_PROJECT_REF=${ref}`);
  console.log(`SUPABASE_URL=https://${ref}.supabase.co`);
  console.log("Fetch publishable/secret API keys from the project's API Keys settings.");
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
