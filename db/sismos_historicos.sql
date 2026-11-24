-- Ejecutar en Supabase > SQL Editor (una vez). Historial de sismos M4.5+ (USGS).
create table if not exists sismos_historicos (
  id text primary key,            -- id del evento USGS
  fecha timestamptz not null,
  mag double precision not null,
  prof double precision,
  lat double precision,
  lon double precision,
  lugar text
);
create index if not exists idx_sismos_hist_fecha on sismos_historicos (fecha);
create index if not exists idx_sismos_hist_mag on sismos_historicos (mag desc);
alter table sismos_historicos enable row level security;  -- sin politicas: solo la API (service key)
