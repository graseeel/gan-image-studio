import { createClient } from "jsr:@supabase/supabase-js@2";

const supabaseUrl = Deno.env.get("SUPABASE_URL");
const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");

Deno.serve(async (request) => {
  if (request.method !== "POST") {
    return new Response("method not allowed", { status: 405 });
  }
  if (!supabaseUrl || !serviceRoleKey) {
    return new Response("backend storage credentials are not configured", { status: 500 });
  }

  const authHeader = request.headers.get("Authorization");
  if (!authHeader) {
    return new Response("missing bearer token", { status: 401 });
  }

  const payload = await request.json();
  const { path, contentType } = payload;
  if (typeof path !== "string" || !path.endsWith(".pt")) {
    return new Response("checkpoint path must end with .pt", { status: 400 });
  }
  if (contentType && contentType !== "application/octet-stream") {
    return new Response("unsupported checkpoint content type", { status: 400 });
  }

  const token = authHeader.replace(/^Bearer\s+/i, "");
  const supabase = createClient(supabaseUrl, serviceRoleKey);
  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data.user) {
    return new Response("invalid user token", { status: 401 });
  }

  if (!path.startsWith(`${data.user.id}/checkpoints/`)) {
    return new Response("checkpoint path must be scoped to the signed-in user", { status: 403 });
  }

  return Response.json({
    bucket: "model-checkpoints",
    path,
    allowed: true,
  });
});
