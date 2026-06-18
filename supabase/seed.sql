insert into public.experiments (
  id,
  owner_id,
  name,
  description,
  dataset_kind,
  latent_dim,
  image_size,
  image_channels,
  batch_size,
  config,
  status,
  is_public
)
select
  '00000000-0000-0000-0000-000000000001',
  id,
  'Local DCGAN smoke experiment',
  'Seeded only when a local auth user already exists.',
  'cifar10',
  100,
  32,
  3,
  8,
  '{"quick_cpu": true}'::jsonb,
  'draft',
  false
from auth.users
limit 1
on conflict (id) do nothing;
