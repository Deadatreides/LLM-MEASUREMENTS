"""
lab/debug_step.py — один шаг Mycelium на одной задаче, чтобы вытрясти
рантайм-ошибки до настоящего прогона.

Пока не докачался Qwen3-0.6B, роль S временно подменяется на qwen3-1.7b:
это ТОЛЬКО отладка плумбинга, никакие числа отсюда в отчёт не идут.
    python lab/debug_step.py --remap-s qwen3-1.7b
"""
import argparse
import asyncio
import json
import logging
import os
import sys

os.environ.setdefault('MYCELIUM_ALLOW_WINDOWS_CODE_EXEC', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import run_experiment as lab            # noqa: E402
import orchestrator as orch_mod         # noqa: E402


async def main(args):
    corpus = json.load(open(os.path.join(HERE, 'data', 'corpus30.json'),
                            encoding='utf-8'))['tasks']
    task = corpus[args.task_index]

    recorder = lab.CallRecorder()
    orch_mod.call_model = recorder.wrap(orch_mod.call_model)
    logbuf = lab.StepLogBuffer()
    logging.getLogger().addHandler(logbuf)

    orch = lab.make_orchestrator(args.condition, f'data/lab_debug_{args.condition}')
    if args.remap_s:
        for mid, mc in orch.models_by_id.items():
            if mid.startswith('qwen06'):
                mc['model_name'] = args.remap_s
        print(f"!! DEBUG: роль S подменена на {args.remap_s}")

    rec = await lab.run_one(orch, orch.scorer, args.condition, task,
                            logbuf, recorder)
    slim = {k: v for k, v in rec.items() if k not in ('answer', 'calls')}
    print("\n===== RECORD =====")
    print(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
    print("\n===== CALLS =====")
    for c in rec.get('calls', []):
        print(" ", c)
    print("\n===== ANSWER (300) =====")
    print(rec.get('answer', '')[:300])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='M')
    ap.add_argument('--task-index', type=int, default=0)
    ap.add_argument('--remap-s', default=None)
    asyncio.run(main(ap.parse_args()))
