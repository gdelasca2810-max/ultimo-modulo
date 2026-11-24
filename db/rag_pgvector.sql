-- Ejecutar en Supabase > SQL Editor (una vez). RAG con pgvector.
create extension if not exists vector;

create table if not exists conocimiento (
  id bigint generated always as identity primary key,
  fuente text not null,
  titulo text not null,
  contenido text not null,
  embedding vector(768) not null,
  unique (fuente, titulo)
);
alter table conocimiento enable row level security;  -- sin politicas: solo la API (service key)

create or replace function buscar_conocimiento(query_embedding vector(768), match_count int default 3)
returns table (titulo text, contenido text, similitud float)
language sql stable as $$
  select titulo, contenido, 1 - (embedding <=> query_embedding) as similitud
  from conocimiento
  order by embedding <=> query_embedding
  limit match_count;
$$;
