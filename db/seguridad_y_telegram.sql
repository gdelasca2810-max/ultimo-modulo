-- Ejecutar en Supabase > SQL Editor (una vez).

-- 1) RLS: cada usuario solo ve/crea/borra sus alertas
alter table alertas_usuario enable row level security;
create policy "alertas_select_propias" on alertas_usuario for select using (auth.uid() = user_id);
create policy "alertas_insert_propias" on alertas_usuario for insert with check (auth.uid() = user_id);
create policy "alertas_delete_propias" on alertas_usuario for delete using (auth.uid() = user_id);

-- 2) Zonas: lectura publica, sin escritura desde la web (la API usa service key)
alter table zonas_sismicas enable row level security;
create policy "zonas_lectura_publica" on zonas_sismicas for select using (true);

-- 3) Vinculo Telegram por usuario
create table if not exists vinculos_telegram (
  user_id uuid primary key references auth.users(id) on delete cascade,
  codigo  text unique not null,
  chat_id text
);
alter table vinculos_telegram enable row level security;
create policy "vinculo_select_propio" on vinculos_telegram for select using (auth.uid() = user_id);
create policy "vinculo_insert_propio" on vinculos_telegram for insert with check (auth.uid() = user_id);
create policy "vinculo_update_propio" on vinculos_telegram for update using (auth.uid() = user_id) with check (auth.uid() = user_id and chat_id is not distinct from (select chat_id from vinculos_telegram v where v.user_id = auth.uid()));
