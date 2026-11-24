-- Ejecutar en Supabase > SQL Editor (una vez). Guarda el riesgo de hoy para mostrarlo en la web.
alter table zonas_sismicas add column if not exists multiplicador_riesgo double precision;
alter table zonas_sismicas add column if not exists probabilidad_m65_7d double precision;
alter table zonas_sismicas add column if not exists actualizado_en timestamptz;
