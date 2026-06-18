# Supabase Setup

This repository must use a new Supabase project dedicated to `gan-image-studio`.
Do not reuse an existing project, bucket, schema, URL, service role key, or Edge
Function.

## Local Development

1. Start the local stack:

   ```bash
   supabase start
   ```

2. Apply migrations and seed data:

   ```bash
   supabase db reset
   ```

3. Verify the local assets:

   ```bash
   npm run verify:supabase
   ```

The local config defines these private buckets:

- `generated-images`
- `training-samples`
- `model-checkpoints`
- `evaluation-assets`

## Remote Provisioning

The provisioning script refuses to run if `SUPABASE_PROJECT_REF` is already set.
That guard prevents accidental reuse of an existing project.

Required environment variables:

```bash
export SUPABASE_ACCESS_TOKEN="..."
export SUPABASE_ORGANIZATION_ID="..."
export SUPABASE_DB_PASSWORD="..."
export SUPABASE_REGION="sa-east-1"
```

Run:

```bash
npm run provision:supabase
```

The script creates a unique name such as `gan-image-studio-a1b2c3d4`, waits for
the project to become ready, links the CLI, pushes migrations, and deploys the
`checkpoint-upload` Edge Function. It prints only non-secret values.

## Security Model

- All public tables have RLS enabled.
- Public tables include explicit grants for `anon`, `authenticated`, and
  `service_role` where needed because new Supabase projects no longer expose
  public tables to the Data API automatically.
- Users can read and write only their own private generations.
- Public experiment visibility is controlled by `experiments.is_public`.
- Checkpoint storage has no authenticated upload policy. Checkpoint upload is a
  backend-only path through the service role or the `checkpoint-upload` Edge
  Function.
- Privileged trigger functions live in the private `app_private` schema instead
  of an exposed schema.

## Required App Values

After provisioning, configure:

```bash
SUPABASE_URL=https://<new-project-ref>.supabase.co
SUPABASE_ANON_KEY=<publishable-or-legacy-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<backend-only-secret-or-legacy-service-role-key>
```

Do not commit `.env`, database passwords, access tokens, service role keys, or
secret API keys.
