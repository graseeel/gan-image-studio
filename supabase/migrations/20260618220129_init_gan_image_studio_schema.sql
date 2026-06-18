create extension if not exists pgcrypto;

create schema if not exists app_private;
revoke all on schema app_private from public;

create or replace function app_private.set_updated_at()
returns trigger
language plpgsql
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create or replace function app_private.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'display_name', split_part(new.email, '@', 1)))
  on conflict (id) do nothing;
  return new;
end;
$$;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.experiments (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users(id) on delete cascade,
  name text not null,
  description text,
  dataset_kind text not null check (dataset_kind in ('cifar10', 'folder')),
  dataset_uri text,
  latent_dim integer not null check (latent_dim > 0),
  image_size integer not null check (image_size >= 32),
  image_channels integer not null check (image_channels in (1, 3)),
  batch_size integer not null check (batch_size > 0),
  config jsonb not null default '{}'::jsonb,
  status text not null default 'draft' check (status in ('draft', 'running', 'completed', 'failed')),
  is_public boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.experiment_metrics (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references public.experiments(id) on delete cascade,
  epoch integer not null check (epoch >= 0),
  step integer not null check (step >= 0),
  generator_loss double precision,
  discriminator_loss double precision,
  fid double precision,
  metrics jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  unique (experiment_id, step)
);

create table public.model_checkpoints (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references public.experiments(id) on delete cascade,
  owner_id uuid not null references auth.users(id) on delete cascade,
  storage_bucket text not null default 'model-checkpoints',
  storage_path text not null,
  epoch integer not null check (epoch >= 0),
  step integer not null check (step >= 0),
  metrics jsonb not null default '{}'::jsonb,
  sha256 text not null check (length(sha256) = 64),
  size_bytes bigint not null check (size_bytes > 0),
  is_validated boolean not null default false,
  created_at timestamptz not null default now(),
  unique (storage_bucket, storage_path)
);

create table public.generations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  experiment_id uuid references public.experiments(id) on delete set null,
  checkpoint_id uuid references public.model_checkpoints(id) on delete set null,
  storage_bucket text not null default 'generated-images',
  storage_path text not null,
  seed integer not null,
  image_count integer not null check (image_count > 0),
  metadata jsonb not null default '{}'::jsonb,
  is_private boolean not null default true,
  created_at timestamptz not null default now(),
  unique (storage_bucket, storage_path)
);

create table public.generation_favorites (
  user_id uuid not null references auth.users(id) on delete cascade,
  generation_id uuid not null references public.generations(id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (user_id, generation_id)
);

create table public.evaluation_reports (
  id uuid primary key default gen_random_uuid(),
  experiment_id uuid not null references public.experiments(id) on delete cascade,
  checkpoint_id uuid references public.model_checkpoints(id) on delete set null,
  owner_id uuid not null references auth.users(id) on delete cascade,
  fid double precision,
  sample_count integer not null check (sample_count >= 0),
  representative boolean not null default false,
  report_path text,
  metrics jsonb not null default '{}'::jsonb,
  notes text,
  created_at timestamptz not null default now()
);

create trigger set_profiles_updated_at
before update on public.profiles
for each row execute function app_private.set_updated_at();

create trigger set_experiments_updated_at
before update on public.experiments
for each row execute function app_private.set_updated_at();

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function app_private.handle_new_user();

alter table public.profiles enable row level security;
alter table public.experiments enable row level security;
alter table public.experiment_metrics enable row level security;
alter table public.model_checkpoints enable row level security;
alter table public.generations enable row level security;
alter table public.generation_favorites enable row level security;
alter table public.evaluation_reports enable row level security;

grant usage on schema public to anon, authenticated, service_role;

grant select on public.experiments to anon;
grant select on public.experiment_metrics to anon;
grant select on public.model_checkpoints to anon;
grant select on public.generations to anon;
grant select on public.evaluation_reports to anon;

grant select, insert, update on public.profiles to authenticated;
grant select, insert, update, delete on public.experiments to authenticated;
grant select, insert, update, delete on public.experiment_metrics to authenticated;
grant select on public.model_checkpoints to authenticated;
grant select, insert, update, delete on public.generations to authenticated;
grant select, insert, delete on public.generation_favorites to authenticated;
grant select, insert, update, delete on public.evaluation_reports to authenticated;

grant all on public.profiles to service_role;
grant all on public.experiments to service_role;
grant all on public.experiment_metrics to service_role;
grant all on public.model_checkpoints to service_role;
grant all on public.generations to service_role;
grant all on public.generation_favorites to service_role;
grant all on public.evaluation_reports to service_role;

create policy "profiles are visible to their owner"
on public.profiles for select
to authenticated
using ((select auth.uid()) = id);

create policy "profiles are inserted by their owner"
on public.profiles for insert
to authenticated
with check ((select auth.uid()) = id);

create policy "profiles are updated by their owner"
on public.profiles for update
to authenticated
using ((select auth.uid()) = id)
with check ((select auth.uid()) = id);

create policy "public experiments are readable"
on public.experiments for select
to anon, authenticated
using (is_public or (select auth.uid()) = owner_id);

create policy "experiment owners can insert"
on public.experiments for insert
to authenticated
with check ((select auth.uid()) = owner_id);

create policy "experiment owners can update"
on public.experiments for update
to authenticated
using ((select auth.uid()) = owner_id)
with check ((select auth.uid()) = owner_id);

create policy "experiment owners can delete"
on public.experiments for delete
to authenticated
using ((select auth.uid()) = owner_id);

create policy "metrics follow experiment visibility"
on public.experiment_metrics for select
to anon, authenticated
using (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id
      and (e.is_public or e.owner_id = (select auth.uid()))
  )
);

create policy "experiment owners can insert metrics"
on public.experiment_metrics for insert
to authenticated
with check (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id and e.owner_id = (select auth.uid())
  )
);

create policy "experiment owners can update metrics"
on public.experiment_metrics for update
to authenticated
using (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id and e.owner_id = (select auth.uid())
  )
)
with check (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id and e.owner_id = (select auth.uid())
  )
);

create policy "experiment owners can delete metrics"
on public.experiment_metrics for delete
to authenticated
using (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id and e.owner_id = (select auth.uid())
  )
);

create policy "checkpoints follow experiment visibility"
on public.model_checkpoints for select
to anon, authenticated
using (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id
      and (e.is_public or e.owner_id = (select auth.uid()))
  )
);

create policy "generations are readable by owner or when public"
on public.generations for select
to anon, authenticated
using (not is_private or (select auth.uid()) = user_id);

create policy "users insert their own generations"
on public.generations for insert
to authenticated
with check ((select auth.uid()) = user_id);

create policy "users update their own generations"
on public.generations for update
to authenticated
using ((select auth.uid()) = user_id)
with check ((select auth.uid()) = user_id);

create policy "users delete their own generations"
on public.generations for delete
to authenticated
using ((select auth.uid()) = user_id);

create policy "users read their own favorites"
on public.generation_favorites for select
to authenticated
using ((select auth.uid()) = user_id);

create policy "users insert their own favorites"
on public.generation_favorites for insert
to authenticated
with check ((select auth.uid()) = user_id);

create policy "users delete their own favorites"
on public.generation_favorites for delete
to authenticated
using ((select auth.uid()) = user_id);

create policy "evaluation reports follow experiment visibility"
on public.evaluation_reports for select
to anon, authenticated
using (
  exists (
    select 1 from public.experiments e
    where e.id = experiment_id
      and (e.is_public or e.owner_id = (select auth.uid()))
  )
);

create policy "owners insert evaluation reports"
on public.evaluation_reports for insert
to authenticated
with check ((select auth.uid()) = owner_id);

create policy "owners update evaluation reports"
on public.evaluation_reports for update
to authenticated
using ((select auth.uid()) = owner_id)
with check ((select auth.uid()) = owner_id);

create policy "owners delete evaluation reports"
on public.evaluation_reports for delete
to authenticated
using ((select auth.uid()) = owner_id);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('generated-images', 'generated-images', false, 20971520, array['image/png', 'image/jpeg', 'image/webp']),
  ('training-samples', 'training-samples', false, 52428800, array['image/png', 'image/jpeg', 'image/webp']),
  ('model-checkpoints', 'model-checkpoints', false, 524288000, null),
  ('evaluation-assets', 'evaluation-assets', false, 52428800, array['image/png', 'image/jpeg', 'image/webp', 'application/json'])
on conflict (id) do update set
  public = excluded.public,
  file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

alter table storage.objects enable row level security;

create policy "users read their generated images"
on storage.objects for select
to authenticated
using (
  bucket_id = 'generated-images'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);

create policy "users upload their generated images"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'generated-images'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);

create policy "users manage their training samples"
on storage.objects for all
to authenticated
using (
  bucket_id = 'training-samples'
  and (storage.foldername(name))[1] = (select auth.uid())::text
)
with check (
  bucket_id = 'training-samples'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);

create policy "users manage their evaluation assets"
on storage.objects for all
to authenticated
using (
  bucket_id = 'evaluation-assets'
  and (storage.foldername(name))[1] = (select auth.uid())::text
)
with check (
  bucket_id = 'evaluation-assets'
  and (storage.foldername(name))[1] = (select auth.uid())::text
);
