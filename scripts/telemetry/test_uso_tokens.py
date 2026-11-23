"""Tests de uso_tokens.py — corre con: python -m unittest scripts/telemetry/test_uso_tokens.py

Bug real (2026-11-05): las consultas al advisor (Opus) salían como «no expuesto» porque el
script asumía que el registro de Claude Code no traía su uso. Sí lo trae: en
`message.usage.iterations[]` hay un elemento `type == "advisor_message"` con `model` y los
tokens reales. Los registros de abajo tienen la forma del real (verificado en la sesión
0f4a2368, Claude Code 2.1.316).
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import uso_tokens as u  # noqa: E402


def _line(rid, usage, content, ts='2026-11-05T12:00:00.000Z'):
    return json.dumps({
        'type': 'assistant', 'requestId': rid, 'timestamp': ts,
        'message': {'id': 'msg_' + rid, 'model': 'claude-sonnet-5-5', 'usage': usage, 'content': content},
    })


ADVISOR_USAGE = {
    'input_tokens': 4, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 359920,
    'output_tokens': 506,
    'iterations': [
        {'type': 'message', 'input_tokens': 2, 'output_tokens': 70, 'cache_read_input_tokens': 177906},
        {'type': 'advisor_message', 'model': 'claude-opus-5-5', 'input_tokens': 175200,
         'output_tokens': 13809, 'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0},
        {'type': 'message', 'input_tokens': 2, 'output_tokens': 436, 'cache_read_input_tokens': 182014},
    ],
}
ADVISOR_BLOCK = [{'type': 'server_tool_use', 'name': 'advisor', 'input': {}}]


class AdvisorUsage(unittest.TestCase):
    def _session(self, lines):
        d = Path(tempfile.mkdtemp())
        f = d / 'sesion-de-prueba-uso-tokens.jsonl'
        f.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return f

    def test_advisor_muestra_los_tokens_reales_de_opus(self):
        rows = u.all_requests(self._session([_line('r1', ADVISOR_USAGE, ADVISOR_BLOCK)]))
        opus = [r for r in rows if u.family(r['model']) == 'opus']
        self.assertEqual(len(opus), 1)
        self.assertFalse(opus[0]['opaque'], 'no debe salir «no expuesto» si el registro trae el uso')
        self.assertEqual((opus[0]['input'], opus[0]['output']), (175200, 13809))
        self.assertAlmostEqual(u.cost(opus[0]), (175200 * 4.0 + 13809 * 20.0) / 1e6, places=6)

    def test_no_cuenta_dos_veces_la_consulta(self):
        rows = u.all_requests(self._session([_line('r1', ADVISOR_USAGE, ADVISOR_BLOCK)]))
        self.assertEqual(len([r for r in rows if r['agent'] and r['agent'].startswith('advisor')]), 1)
        sonnet = [r for r in rows if u.family(r['model']) == 'sonnet']
        self.assertEqual(sonnet[0]['input'], 4, 'la fila de la sesión no incluye los tokens del advisor')

    def test_respaldo_opaco_si_el_registro_no_trae_la_iteracion(self):
        usage = {'input_tokens': 4, 'cache_read_input_tokens': 10, 'output_tokens': 5,
                 'iterations': [{'type': 'message', 'input_tokens': 4, 'output_tokens': 5}]}
        rows = u.all_requests(self._session([_line('r2', usage, ADVISOR_BLOCK)]))
        adv = [r for r in rows if r['agent'] and r['agent'].startswith('advisor')]
        self.assertEqual(len(adv), 1)
        self.assertTrue(adv[0]['opaque'])

    def test_report_json_resume_el_advisor_para_el_panel(self):
        d = u.report_json(self._session([_line('r1', ADVISOR_USAGE, ADVISOR_BLOCK)]))
        self.assertEqual(d['advisor_llamadas'], 1)
        self.assertEqual(d['advisor_tokens'], 175200 + 13809)
        self.assertGreater(d['tokens_totales'], d['advisor_tokens'])
        self.assertFalse(d['hay_opacas'])


class PreciosFable(unittest.TestCase):
    """Fable 5.2 (2026-11-09) cuesta 8/40 por 1M; Fable 5.1 sigue en 10/50."""

    def _row(self, model):
        return {'model': model, 'input': 1_000_000, 'output': 1_000_000, 'cache_read': 1_000_000,
                'cache_write_5m': 0, 'cache_write_1h': 0}

    def test_fable_5_2_usa_su_precio(self):
        self.assertAlmostEqual(u.cost(self._row('claude-fable-5-2')), 8.0 + 40.0 + 0.20, places=6)

    def test_fable_5_1_conserva_el_precio_anterior(self):
        self.assertAlmostEqual(u.cost(self._row('claude-fable-5-1')), 10.0 + 50.0 + 0.25, places=6)

    def test_ambas_se_agrupan_como_fable(self):
        self.assertEqual(u.family('claude-fable-5-2'), 'fable')
        self.assertEqual(u.family('claude-fable-5-1'), 'fable')


if __name__ == '__main__':
    unittest.main()
