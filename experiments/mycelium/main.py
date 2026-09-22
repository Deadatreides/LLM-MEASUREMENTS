"""
main.py — Точка входа Σ_v8.5 «Мицелий»

Три режима:
  python main.py                  — интерактивный REPL
  python main.py --task "..."     — одиночный запрос
  python main.py --nightly        — ночной DPO цикл
  python main.py --opencode       — режим для OpenCode (stdin/stdout)
"""

import asyncio
import argparse
import json
import os
import sys
import time
import threading
import logging

from orchestrator import MyceliumOrchestrator
from agents.trainer import NightlyTrainer

try:
    import schedule
except ImportError:
    schedule = None

from core.logging_setup import setup_logging
setup_logging()
log = logging.getLogger('main')



def run_nightly(orch: MyceliumOrchestrator):
    trainer = NightlyTrainer(orch.cfg, orch.memory.store)
    beta_kl = orch.chaos.current_beta_kl
    trainer.run(beta_kl=beta_kl)


def schedule_nightly(orch: MyceliumOrchestrator):
    """Запускает планировщик в отдельном потоке"""
    if schedule is None:
        log.warning("Package 'schedule' is not installed; nightly scheduler disabled.")
        return
    nightly_hour = orch.cfg['training']['nightly_hour']
    schedule.every().day.at(f"{nightly_hour:02d}:00").do(run_nightly, orch=orch)
    log.info(f"Nightly DPO scheduled at {nightly_hour:02d}:00")

    def _loop():
        while True:
            schedule.run_pending()
            time.sleep(60)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


async def interactive(orch: MyceliumOrchestrator):
    """REPL для интерактивного использования"""
    print("\n" + "═"*60)
    print("  Σ_v8.5 «Мицелий» — Emergent Swarm")
    print("  Введи задачу. Команды: /quit /status /save")
    print("═"*60 + "\n")

    while True:
        try:
            task = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not task:
            continue

        if task == '/quit':
            break
        elif task == '/status':
            print(json.dumps({
                'step': orch.step,
                'I': round(orch.memory.I, 4),
                'lambda_L': round(orch.chaos.current_lambda, 4),
                'k_act': orch.chaos.current_k_act,
                'regime': orch.chaos._regime(orch.chaos.current_lambda),
                'alive_agents': sum(1 for a in orch.bandit.agents.values() if a.alive),
                'tokens_used': orch.tokens_used,
            }, indent=2))
            continue
        elif task == '/save':
            orch.bandit.save('data/bandit_state.json')
            orch.memory.faiss.save()
            orch.memory.graph._save()
            print("State saved.")
            continue

        # Определяем тип задачи
        task_type = 'general'
        if any(kw in task.lower() for kw in ['python', 'def ', 'function', 'code', 'class ', 'import ']):
            task_type = 'code'
        elif any(kw in task.lower() for kw in ['equation', 'solve', 'calculate', 'math', '∫', 'Σ']):
            task_type = 'math'

        print(f"\n[{task_type}] Processing...\n")
        t0 = time.monotonic()

        result = await orch.run_step(task, task_type)

        elapsed = time.monotonic() - t0
        print(f"\n{'─'*60}")
        print(result['answer'])
        print(f"\n{'─'*60}")
        print(f"  E={result['e_total']:.3f} | λ={result['lambda_L']:.3f} | "
              f"K={result['k_act']} | {elapsed:.1f}s | "
              f"regime={result['regime']} | I={result['I']:.4f}")
        print()


async def _handle_opencode_request(orch: MyceliumOrchestrator, req: dict):
    try:
        task = req.get('task', '')
        task_type = req.get('type', 'general')
        result = await orch.run_step(task, task_type)
        out = {
            'answer': result['answer'],
            'e_total': round(result['e_total'], 4),
            'delta_I': round(result.get('delta_I', 0.0), 6),
            'step_time': round(result['step_time'], 2),
            'models': result['models'],
        }
        print(json.dumps(out, ensure_ascii=False), flush=True)
    except Exception as e:
        print(json.dumps({'error': str(e)}, ensure_ascii=False), flush=True)


async def opencode_mode(orch: MyceliumOrchestrator):
    """
    Режим для OpenCode: читает задачи из stdin (JSON-lines), выдаёт в stdout.
    OpenCode вызывает: python main.py --opencode
    Протокол: {"task": "...", "type": "code"} → {"answer": "...", "e_total": 0.3, ...}
    
    Модифицировано: чтение в фоновом потоке, чтобы не блокировать event loop.
    """
    log.info("OpenCode mode: reading from stdin (non-blocking thread queue)")
    
    # Force UTF-8 environment
    from core.logging_setup import force_utf8_env
    force_utf8_env()

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _read_stdin():
        try:
            for line in sys.stdin:
                loop.call_soon_threadsafe(queue.put_nowait, line)
        except Exception as stdin_err:
            log.error(f"Stdin reader thread exception: {stdin_err}")

    # Run stdin reader in a daemon thread
    reader_thread = threading.Thread(target=_read_stdin, daemon=True)
    reader_thread.start()

    decoder = json.JSONDecoder()
    buffer = ""

    while True:
        try:
            line = await queue.get()
        except (asyncio.CancelledError, EOFError):
            break

        if not line.strip() and not buffer:
            continue
        buffer += line
        while buffer.strip():
            stripped = buffer.lstrip()
            try:
                req, idx = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                break
            if not isinstance(req, dict):
                print(json.dumps({'error': 'request must be object'}, ensure_ascii=False), flush=True)
            else:
                await _handle_opencode_request(orch, req)
            buffer = stripped[idx:]



def main():
    from core.logging_setup import force_utf8_env
    force_utf8_env()

    parser = argparse.ArgumentParser(description='Σ_v8.5 Мицелий')
    parser.add_argument('--task', type=str, help='Одиночный запрос')
    parser.add_argument('--type', type=str, default='general',
                        choices=['general', 'code', 'math'])
    parser.add_argument('--nightly', action='store_true', help='Запустить DPO сейчас')
    parser.add_argument('--opencode', action='store_true', help='Режим OpenCode (stdin)')
    args = parser.parse_args()

    orch = MyceliumOrchestrator()


    if args.nightly:
        run_nightly(orch)
        return

    # Запускаем планировщик в фоне для всех режимов кроме --nightly
    schedule_nightly(orch)

    # FIX 6: ежечасная очистка swarm/context/ (не больше 100 файлов)
    def _cleanup_context():
        import glob, os as _os
        files = sorted(glob.glob('swarm/context/*.json'), key=_os.path.getmtime)
        for f in files[:-100]:   # оставить последние 100
            try: _os.remove(f)
            except Exception: pass

    # [Ш5] Было вне охраны: при отсутствующем пакете schedule (а он не в
    # requirements как обязательный) здесь падало AttributeError на None —
    # то есть main.py не запускался вообще ни в одном режиме, кроме --nightly.
    if schedule is not None:
        schedule.every(1).hours.do(_cleanup_context)
    else:
        _cleanup_context()      # разовая очистка вместо периодической

    if args.task:
        result = asyncio.run(orch.run_step(args.task, args.type))
        print(result['answer'])
        print(f"\nE={result['e_total']:.3f} | {result['step_time']:.1f}s")
    elif args.opencode:
        asyncio.run(opencode_mode(orch))
    else:
        asyncio.run(interactive(orch))


if __name__ == '__main__':
    main()
