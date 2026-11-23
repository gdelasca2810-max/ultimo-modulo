"""Tokens MEDIDOS de la sesión de Claude Code, por sub-tarea y por modelo.

Claude Code guarda cada llamada al modelo en ~/.claude/projects/<proyecto>/<sesión>.jsonl
(y las de los agentes en <sesión>/subagents/*.jsonl) con su `usage` real. Este script suma
ese uso; no estima nada.

Uso:
  python scripts/telemetry/uso_tokens.py --mark "Sub-tarea 1: Diagnóstico"   # al empezar cada sub-tarea
  python scripts/telemetry/uso_tokens.py --report                            # al terminar (tabla Markdown)
  python scripts/telemetry/uso_tokens.py --json                              # lo mismo, estructurado (panel /tokens)
  python scripts/telemetry/uso_tokens.py --reset                             # borra las marcas de la sesión

Límite: la respuesta que se está escribiendo en ese momento aún no está en el registro, así
que el reporte cubre todo hasta la última llamada terminada.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import sys
from pathlib import Path

# Precios USD por 1M tokens: (entrada, caché 5 min, caché 1 h, lectura de caché, salida).
# Fuente: platform.claude.com/docs/en/about-claude/pricing (verificado 2026-10-14).
# Con suscripción no se cobra por token: es el costo EQUIVALENTE en la API.
PRICES = {
    'opus': (4.0, 5.0, 8.0, 0.20, 20.0),     # Opus 5.5
    'sonnet': (2.0, 2.5, 4.0, 0.10, 10.0),   # Sonnet 5.5
    'haiku': (1.0, 1.25, 2.0, 0.10, 5.0),    # Haiku 4.5
    # Fable 5.2 (salió 2026-11-09): 8 / 40 y lectura de caché 0.20, según
    # platform.claude.com/docs/en/models/fable-5-2/migration-guide (verificado 2026-11-12).
    # Escritura de caché = entrada × 1.25 / × 2, la misma proporción que el resto (no leída del doc).
    'fable': (8.0, 10.0, 16.0, 0.20, 40.0),
}

# Versiones con precio distinto al de su familia (se busca el texto dentro del id del modelo).
PRICES_BY_VERSION = {
    'fable-5-1': (10.0, 12.5, 20.0, 0.25, 50.0),  # Fable 5.1
}


def project_dir(cwd: Path) -> Path:
    slug = ''.join(c if c.isalnum() else '-' for c in str(cwd.resolve()))
    return Path.home() / '.claude' / 'projects' / slug


def current_session(pdir: Path) -> Path:
    files = sorted(pdir.glob('*.jsonl'), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f'No hay sesiones en {pdir}')
    return files[-1]


def family(model: str) -> str:
    m = (model or '').lower()
    for k in ('opus', 'sonnet', 'haiku', 'fable'):
        if k in m:
            return k
    return m or '?'


def parse_ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))


def requests(path: Path, agent: str | None, covered: set | None = None):
    """Una fila por llamada al modelo. El registro repite la llamada por cada bloque de la
    respuesta: se usa el ÚLTIMO registro de cada una (el más completo).

    Las consultas al advisor SÍ traen su uso real: `usage.iterations[]` incluye un elemento
    `type == 'advisor_message'` con `model` y los tokens de Opus (comprobado 2026-11-05 en
    Claude Code 2.1.316; la hipótesis anterior «nunca aparece» era falsa). El `usage` de nivel
    superior suma solo las iteraciones `message`, así que no hay doble conteo. Cada una sale
    como fila aparte (agent «advisor…»); sus requestId se anotan en [covered] para que
    `advisor_calls` (respaldo opaco) no las cuente otra vez."""
    last = {}
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') == 'assistant':
                msg = d.get('message') or {}
                last[d.get('requestId') or msg.get('id')] = line
    seen = set()
    for line in last.values():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get('type') != 'assistant':
            continue
        msg = d.get('message') or {}
        rid = d.get('requestId') or msg.get('id')
        usage = msg.get('usage')
        model = msg.get('model') or ''
        if not usage or rid in seen or model.startswith('<'):
            continue
        seen.add(rid)
        cc = usage.get('cache_creation') or {}
        for it in usage.get('iterations') or []:
            if it.get('type') != 'advisor_message':
                continue
            icc = it.get('cache_creation') or {}
            if covered is not None:
                covered.add(rid)
            yield {
                'ts': parse_ts(d['timestamp']),
                'model': it.get('model') or d.get('advisorModel') or 'opus',
                'agent': f"advisor{' de ' + agent if agent else ''}",
                'input': it.get('input_tokens', 0),
                'cache_write_5m': icc.get('ephemeral_5m_input_tokens', 0),
                'cache_write_1h': icc.get('ephemeral_1h_input_tokens', 0),
                'cache_read': it.get('cache_read_input_tokens', 0),
                'output': it.get('output_tokens', 0),
                'opaque': False,
            }
        yield {
            'ts': parse_ts(d['timestamp']),
            'model': model,
            'agent': agent,
            'input': usage.get('input_tokens', 0),
            'cache_write_5m': cc.get('ephemeral_5m_input_tokens', 0)
            if cc else usage.get('cache_creation_input_tokens', 0),
            'cache_write_1h': cc.get('ephemeral_1h_input_tokens', 0),
            'cache_read': usage.get('cache_read_input_tokens', 0),
            'output': usage.get('output_tokens', 0),
            'opaque': False,
        }


def advisor_calls(path: Path, agent: str | None, covered: set | None = None):
    """Una fila por consulta al advisor (herramienta de servidor).

    RESPALDO: solo cuenta las consultas cuyo `usage` NO trae la iteración `advisor_message`
    (versiones de Claude Code que no la exponen). El contenido vuelve cifrado
    (`advisor_tool_result`), así que sin esa iteración solo se sabe CUÁNTAS consultas hubo,
    no sus tokens (fila «no expuesto»)."""
    seen = set()
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant':
                continue
            msg = d.get('message') or {}
            content = msg.get('content') or []
            if not any(c.get('type') == 'server_tool_use' and c.get('name') == 'advisor' for c in content):
                continue
            key = d.get('requestId') or msg.get('id')
            if key in seen or (covered is not None and key in covered):
                continue
            seen.add(key)
            yield {
                'ts': parse_ts(d['timestamp']),
                'model': d.get('advisorModel') or 'opus',
                'agent': f"advisor{' de ' + agent if agent else ''}",
                'input': 0, 'cache_write_5m': 0, 'cache_write_1h': 0, 'cache_read': 0, 'output': 0,
                'opaque': True,
            }


def agent_name(path: Path) -> str:
    meta = path.with_suffix('.meta.json')
    if meta.exists():
        try:
            m = json.loads(meta.read_text(encoding='utf-8'))
            return m.get('agentType') or m.get('subagent_type') or path.stem
        except (OSError, json.JSONDecodeError):
            pass
    return path.stem


def all_requests(session: Path):
    covered: set = set()
    rows = list(requests(session, None, covered))  # consume `requests` ANTES de advisor_calls
    rows += list(advisor_calls(session, None, covered))
    sub = session.with_suffix('') / 'subagents'
    if sub.is_dir():
        for f in sorted(sub.glob('*.jsonl')):
            agent = agent_name(f)
            cov: set = set()
            rows.extend(requests(f, agent, cov))
            rows.extend(advisor_calls(f, agent, cov))
    return sorted(rows, key=lambda r: r['ts'])


def prices_for(model: str):
    m = (model or '').lower()
    for version, prices in PRICES_BY_VERSION.items():
        if version in m:
            return prices
    return PRICES.get(family(m), (0.0,) * 5)


def cost(r) -> float:
    p_in, p_w5, p_w1h, p_read, p_out = prices_for(r['model'])
    return (r['input'] * p_in + r['cache_write_5m'] * p_w5 + r['cache_write_1h'] * p_w1h
            + r['cache_read'] * p_read + r['output'] * p_out) / 1e6


def marks_file(session: Path) -> Path:
    d = Path.home() / '.claude' / 'telemetry'
    d.mkdir(parents=True, exist_ok=True)
    return d / f'{session.stem}.marks.json'


def load_marks(session: Path):
    f = marks_file(session)
    return json.loads(f.read_text(encoding='utf-8')) if f.exists() else []


def fmt(n: int) -> str:
    return f'{n:,}'.replace(',', '.')


def aggregate(session: Path):
    """(agg, rows): agg[(sub-tarea, 'sesión (sonnet)')] = Counter de llamadas/tokens/USD."""
    marks = [(parse_ts(m['ts']), m['name']) for m in load_marks(session)]
    rows = all_requests(session)
    if marks:
        rows = [r for r in rows if r['ts'] >= marks[0][0]]

    def bucket(ts):
        name = 'Antes de las marcas'
        for t, n in marks:
            if ts >= t:
                name = n
        return name

    agg = collections.OrderedDict()
    for r in rows:
        who = (r['agent'] if r['agent'].startswith('advisor') else f"agente {r['agent']}") if r['agent'] else 'sesión'
        key = (bucket(r['ts']), f"{who} ({family(r['model'])})")
        a = agg.setdefault(key, collections.Counter())
        a['llamadas'] += 1
        a['opacas'] += 1 if r['opaque'] else 0
        a['entrada'] += r['input'] + r['cache_write_5m'] + r['cache_write_1h']
        a['cache_read'] += r['cache_read']
        a['salida'] += r['output']
        a['usd_micro'] += round(cost(r) * 1e6)
    return agg, rows


def report_json(session: Path) -> dict:
    """Mismos números que `report`, estructurados para que un panel los alinee y resuma."""
    agg, rows = aggregate(session)
    tasks, models = [], collections.OrderedDict()
    for (task, who), a in agg.items():
        fam = who.split('(')[-1].rstrip(')')
        opaque = a['opacas'] == a['llamadas']
        tasks.append({'tarea': task, 'quien': who, 'modelo': fam, 'llamadas': a['llamadas'],
                      'entrada': None if opaque else a['entrada'],
                      'cache': None if opaque else a['cache_read'],
                      'salida': None if opaque else a['salida'],
                      'tokens': None if opaque else a['entrada'] + a['salida'],
                      'usd': None if opaque else a['usd_micro'] / 1e6})
        m = models.setdefault(fam, collections.Counter())
        m.update(a)
    model_rows = [{'modelo': f, 'llamadas': m['llamadas'], 'entrada': m['entrada'], 'cache': m['cache_read'],
                   'salida': m['salida'], 'tokens': m['entrada'] + m['salida'],
                   'usd': m['usd_micro'] / 1e6} for f, m in models.items()]
    total_usd = sum(m['usd'] for m in model_rows)
    ent = sum(m['entrada'] for m in model_rows)
    cache = sum(m['cache'] for m in model_rows)
    priced = [t for t in tasks if t['usd'] is not None]
    # Costo por sub-tarea (suma de quienes la ejecutaron)
    per_task = collections.OrderedDict()
    for t in priced:
        per_task[t['tarea']] = per_task.get(t['tarea'], 0.0) + t['usd']
    costliest = max(per_task.items(), key=lambda kv: kv[1]) if per_task else None
    adv = sum(t['usd'] for t in priced if t['quien'].startswith('advisor'))
    adv_rows = [t for t in tasks if t['quien'].startswith('advisor')]
    adv_calls = sum(t['llamadas'] for t in adv_rows)
    adv_tokens = sum(t['tokens'] or 0 for t in adv_rows)
    first, last = (rows[0]['ts'], rows[-1]['ts']) if rows else (None, None)
    return {
        'sesion': session.stem,
        'tareas': tasks, 'modelos': model_rows,
        'total_usd': total_usd, 'llamadas': sum(m['llamadas'] for m in model_rows),
        'cache_hit_pct': (100.0 * cache / (cache + ent)) if (cache + ent) else 0.0,
        'advisor_usd': adv,
        'advisor_llamadas': adv_calls,
        'advisor_tokens': adv_tokens,
        # Entrada + salida (sin lecturas de caché): lo que más pesa contra el límite del plan.
        'tokens_totales': sum(m['tokens'] for m in model_rows),
        'tarea_mas_cara': {'tarea': costliest[0], 'usd': costliest[1]} if costliest else None,
        'minutos': ((last - first).total_seconds() / 60.0) if first and last else 0.0,
        'hay_opacas': any(t['usd'] is None for t in tasks),
    }


def report(session: Path) -> str:
    agg, rows = aggregate(session)

    def cells(a: collections.Counter) -> tuple[str, str, str, str]:
        if a['opacas'] == a['llamadas']:
            return 'no expuesto', 'no expuesto', 'no expuesto', 'no expuesto'
        return fmt(a['entrada']), fmt(a['cache_read']), fmt(a['salida']), f"{a['usd_micro'] / 1e6:.3f}"

    out = ['| Sub-tarea | Modelo | Llamadas | Entrada nueva | Entrada en caché | Salida | USD (equiv. API) |',
           '|---|---|---|---|---|---|---|']
    by_model = collections.defaultdict(collections.Counter)
    for (task, model), a in agg.items():
        entrada, cache_read, salida, usd = cells(a)
        out.append(f"| {task} | {model} | {a['llamadas']} | {entrada} | {cache_read} | {salida} | {usd} |")
        by_model[model.split('(')[-1].rstrip(')')].update(a)
    for fam, a in by_model.items():
        entrada, cache_read, salida, usd = cells(a)
        out.append(f"| **Total {fam}** | | {a['llamadas']} | {entrada} | {cache_read} | {salida} | {usd} |")
    out.append('')
    out.append(f'Sesión `{session.stem}`. Medido del registro de Claude Code (no incluye la respuesta en curso).')
    if any(a['opacas'] for a in agg.values()):
        out.append('Las filas «advisor» cuentan consultas, no tokens: el contenido vuelve cifrado '
                    '(`advisor_tool_result`) y Claude Code no expone su costo real en el registro local.')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mark', help='marca el inicio de una sub-tarea')
    ap.add_argument('--at', help='hora ISO de la marca (por defecto, ahora)')
    ap.add_argument('--report', action='store_true', help='tabla de tokens medidos por sub-tarea y modelo')
    ap.add_argument('--json', action='store_true', help='lo mismo que --report, estructurado (lo usa el panel /tokens)')
    ap.add_argument('--reset', action='store_true', help='borra las marcas de esta sesión')
    ap.add_argument('--session', help='archivo .jsonl de la sesión (por defecto, la más reciente)')
    args = ap.parse_args()

    session = Path(args.session) if args.session else current_session(project_dir(Path.cwd()))
    if args.reset:
        marks_file(session).unlink(missing_ok=True)
        print('Marcas borradas.')
    if args.mark:
        marks = load_marks(session)
        ts = args.at or dt.datetime.now(dt.timezone.utc).isoformat()
        marks.append({'ts': ts, 'name': args.mark})
        marks.sort(key=lambda m: parse_ts(m['ts']))
        marks_file(session).write_text(json.dumps(marks, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'Marca: {args.mark}')
    if args.report:
        sys.stdout.reconfigure(encoding='utf-8')
        print(report(session))
    if args.json:
        sys.stdout.reconfigure(encoding='utf-8')
        print(json.dumps(report_json(session), ensure_ascii=False))
    if not (args.mark or args.report or args.reset or args.json):
        ap.print_help()


if __name__ == '__main__':
    main()
