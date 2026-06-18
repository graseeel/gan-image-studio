BEGIN;
SELECT plan(19);

SELECT has_table('public', 'profiles', 'profiles table exists');
SELECT has_table('public', 'experiments', 'experiments table exists');
SELECT has_table('public', 'experiment_metrics', 'experiment_metrics table exists');
SELECT has_table('public', 'model_checkpoints', 'model_checkpoints table exists');
SELECT has_table('public', 'training_sample_grids', 'training_sample_grids table exists');
SELECT has_table('public', 'generations', 'generations table exists');
SELECT has_table('public', 'generation_favorites', 'generation_favorites table exists');
SELECT has_table('public', 'evaluation_reports', 'evaluation_reports table exists');

SELECT ok(
  exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'generations'
      and policyname = 'generations are readable by owner or when public'
  ),
  'generations has owner/private visibility policy'
);

SELECT ok(
  exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'experiments'
      and policyname = 'public experiments are readable'
  ),
  'experiments has explicit public publish policy'
);

SELECT ok(
  exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'training_sample_grids'
      and policyname = 'training grids follow experiment visibility'
  ),
  'training grids follow experiment visibility'
);

SELECT ok((select relrowsecurity from pg_class where oid = 'public.profiles'::regclass), 'profiles RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.experiments'::regclass), 'experiments RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.experiment_metrics'::regclass), 'experiment_metrics RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.model_checkpoints'::regclass), 'model_checkpoints RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.training_sample_grids'::regclass), 'training_sample_grids RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.generations'::regclass), 'generations RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.generation_favorites'::regclass), 'generation_favorites RLS enabled');
SELECT ok((select relrowsecurity from pg_class where oid = 'public.evaluation_reports'::regclass), 'evaluation_reports RLS enabled');

SELECT * FROM finish();
ROLLBACK;
