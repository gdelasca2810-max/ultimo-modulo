# AGENTS.md - Monitor de Riesgo Sísmico (Módulo 10 Inesdi)

## Idea y Alcance
Sistema de alerta temprana de enjambres sísmicos y cálculo de riesgo relativo de eventos M6.5+ (7d y 48h) en 20 zonas tectónicas, con dashboard privado (Supabase Auth + RLS), API ML (FastAPI en Render), servidor MCP y alertas automáticas por Telegram.

## Datos y Modelo
- **Fuente:** USGS ComCat (`zona_dia_11a.csv`, 78.360 filas, 2016–2026). Uso académico.
- **Features permitidas:** `loc200_n96_con`, `loc200_n96_sin45`, `loc500_n96_con`, `lat_n96_prof70_150` (prohibido usar variables lunares o de identidad pura como `p90_*` o `dest_lat_c`).
- **Target:** `y_M65_7d` (y `y_M65_48h`), evaluado con partición temporal estricta (`train <= 2023`, hueco 7d, `test >= 2024`) comparando AUC y Average Precision frente a la tasa histórica de zona.

## Estructura de Carpetas
- `public/`: Frontend y app con login (Vercel, configurado en `vercel.json`).
- `api/`: Entrenamiento scikit-learn y API REST con FastAPI (Render, `render.yaml`).
- `mcp/`: Servidor MCP con herramientas `buscar_sismos_zona`, `predecir_riesgo_sismico` y `comparar_baseline_zona`.
- `agente/`: Worker autónomo de vigilancia USGS y alertas Telegram.

## Reglas de Trabajo
1. Responde siempre en español, directo y conciso.
2. Todas las métricas y cifras deben calcularse ejecutando código Python real, nunca estimadas.
3. Cero claves, tokens o URLs privadas en el código; usar siempre variables de entorno (`.env`).
4. Avanza en pasos pequeños: explica el plan y los archivos a tocar antes de modificar nada.