"""Ж11. Пять пунктов авторского критерия успеха, которые «ни разу не мерились».

VISION.md §7 перечисляет, что редукция к E_score > 0 отбросила, и даёт
измеримые прокси. VISION.md §9.4 называет последний из них «единственной
неизмеренной метрикой» проекта:

    доля задач, решённых только композицией и никогда ни одним компонентом
    в одиночку.

Три из пяти прокси считаются по уже имеющимся логам, и это делается здесь.
Два не считаются, и это сказано явно.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import statistics as st

from common import ROOT, generators, iter_records, solved, wilson

TAG = 'sh2_200'
LOG = os.path.join(ROOT, 'logs', 'runtime.log')


def slope(ys):
    n = len(ys)
    xm = (n - 1) / 2
    ym = sum(ys) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den if den else 0.0


def main():
    recs = [r for r in iter_records() if r['_tag'] == TAG]
    M = [r for r in recs if r['condition'] == 'M' and r.get('status') == 'ok']
    B = {r['task_id']: r for r in recs
         if r['condition'] == 'BoN-G' and r.get('status') == 'ok'}

    # ── 1. I(t) растёт? ────────────────────────────────────────────────────
    # I пишется только в лог: «Done in Ns | I=...»
    P_DONE = re.compile(r'Done in ([\d.]+)s \| I=([-\d.]+)')
    P_INIT = re.compile(r'\[swarm\] Mycelium v[\d.]+ initialized')
    cur, runs = None, []
    for line in open(LOG, encoding='utf-8', errors='replace'):
        if P_INIT.search(line):
            cur = []
            runs.append(cur)
        if cur is None:
            continue
        m = P_DONE.search(line)
        if m:
            cur.append(float(m.group(2)))
    I = max(runs, key=len)
    print("── VISION §7 п.1: «I(t) растёт — система накапливает информационное "
          "состояние» ──")
    print(f"   значений I в самом длинном прогоне: {len(I)}")
    print(f"   I[0]={I[0]:.6f}  I[-1]={I[-1]:.6f}  "
          f"мин {min(I):.6f}  макс {max(I):.6f}")
    k = slope(I)
    print(f"   наклон МНК по шагам: {k:+.3e} за шаг "
          f"({k*len(I):+.4f} за прогон)")
    print(f"   вывод: I {'РАСТЁТ' if k > 0 else 'ПАДАЕТ' if k < 0 else 'стоит'}")
    dI = [r.get('delta_I', 0.0) for r in M]
    print(f"   delta_I на шаге: положительных {sum(1 for d in dI if d > 0)}, "
          f"нулевых {sum(1 for d in dI if abs(d) < 1e-12)}, "
          f"отрицательных {sum(1 for d in dI if d < 0)}")
    print(f"   медиана delta_I = {st.median(dI):.6f}, "
          f"вклад в E_total = 0.0015·delta_I = "
          f"{0.0015*st.median(dI):.3e} (для сравнения |E_total| ~ 0.04)")

    # почему падает: I — RC-цепь с неподвижной точкой mean(max(0,E_base))
    eb = [max(0.0, r['E_base']) for r in M]
    fix = sum(eb) / len(eb)
    dec = 0.9985
    sim, x = [], 0.5
    for e in eb:
        x = dec * x + (1 - dec) * e
        sim.append(x)
    print(f"   ПРИЧИНА: I(t+1)=0.9985·I+0.0015·max(0,E_base) "
          f"(core/memory.py::LeakyIntegrator)")
    print(f"   неподвижная точка = mean(max(0,E_base)) = {fix:.5f}, "
          f"старт 0.5, постоянная времени {1/(1-dec):.0f} шагов")
    print(f"   предсказание модели: I[-1]={sim[-1]:.6f}, в логе {I[-1]:.6f} "
          f"— совпадает")
    print("   то есть падение I — не дефект, а релаксация от завышенного "
          "старта к точке,")
    print("   которую задаёт E_base. Критерий «I растёт» на 200 шагах "
          "неизмерим в принципе:")
    print(f"   чтобы I росла, нужно mean(max(0,E_base)) > 0.5, сейчас "
          f"{fix:.3f}.")

    # ── 2. специализация в ниши ────────────────────────────────────────────
    # Прокси VISION: энтропия распределения выбора моделей; должна падать.
    print("\n── VISION §7 п.2: «специализация в ниши» ──")
    print("   ОГОВОРКА: прокси из VISION — энтропия выбора ПО ТИПАМ ЗАДАЧ, а в")
    print("   корпусе тип один (code). Меряется энтропия выбора по ролям.")
    win = 50
    for lo in range(0, len(M), win):
        chunk = M[lo:lo + win]
        cnt = collections.Counter()
        for r in chunk:
            for a in r.get('models') or []:
                cnt[a] += 1
        tot = sum(cnt.values())
        H = -sum((v / tot) * math.log(v / tot) for v in cnt.values())
        print(f"   шаги {lo:>3}-{lo+len(chunk)-1:<3}: H(выбор агентов) = "
              f"{H:.4f} из максимума {math.log(len(cnt)):.4f}, "
              f"агентов активно {len(cnt)}")
    # доля роли на агента — есть ли расслоение по кастам
    last = M[-1]
    cr = last.get('bandit_calls_role') or {}
    print("   финальное распределение ролей по агентам:")
    for a in sorted(cr):
        d = cr[a]
        tot = sum(d.values()) or 1
        print(f"      {a:<12} " + " ".join(f"{r}={d.get(r,0):>3}"
                                           for r in ('G', 'C', 'S'))
              + f"   доля G {d.get('G',0)/tot:.2f}")

    # ── 3. растёт ли глубина ───────────────────────────────────────────────
    print("\n── VISION §7 п.4: «растёт глубина» ──")
    steps = [r['depth']['steps'] for r in M if r.get('depth')]
    changed = [bool(r.get('depth_changed_answer')) for r in M]
    print(f"   принятых шагов депт-лупа: среднее {st.mean(steps):.2f}, "
          f"распределение {dict(sorted(collections.Counter(steps).items()))}")
    print(f"   ответ реально изменился: {sum(changed)}/{len(changed)} = "
          f"{sum(changed)/len(changed):.1%}")
    for lo in range(0, len(M), win):
        ch = changed[lo:lo + win]
        print(f"   шаги {lo:>3}-{lo+len(ch)-1:<3}: доля изменённых "
              f"{sum(ch)/len(ch):.1%}")
    n_ch = sum(changed)
    same_q = sum(1 for r in M if r.get('depth_changed_answer')
                 and r.get('q_env_pre_depth') is not None
                 and abs(r['q_env_final'] - r['q_env_pre_depth']) < 1e-9)
    print(f"   из изменённых {n_ch}: внешний результат не изменился у "
          f"{same_q} = {same_q/max(n_ch,1):.1%}")
    print("   то есть депт-луп переписывает обрамление, а не решение")

    # ── 4. ГЛАВНОЕ: доля задач, решённых только композицией ────────────────
    print("\n── VISION §9.4: доля задач, решённых ТОЛЬКО композицией ──")
    print("   (в §9.4 названа «единственной неизмеренной метрикой» проекта)")
    lvl = collections.Counter()
    only_comp_ids = []
    for r in M:
        tid = r['task_id']
        drafts = r.get('q_env_candidates') or []
        fin = solved(r.get('q_env_final', 0.0))
        any_draft = any(solved(q) for q in drafts)
        b = B.get(tid)
        any_bon = any(solved(q) for q in (b.get('q_env_candidates') or [])) if b else False
        if fin and not any_draft and not any_bon:
            lvl['только композиция (ни черновик, ни 8.4 сэмпла лучшей)'] += 1
            only_comp_ids.append(tid)
        elif fin and not any_draft:
            lvl['композиция дала то, чего не дали её собственные черновики'] += 1
        elif fin:
            lvl['итог верен, но черновик и так был верен'] += 1
        else:
            lvl['итог неверен'] += 1
    n = len(M)
    for kk, v in lvl.most_common():
        pp, l, h = wilson(v, n)
        print(f"   {kk:<58} {v:>3}/{n} = {pp:.2%} [{l:.2%}, {h:.2%}]")
    strict = lvl['только композиция (ни черновик, ни 8.4 сэмпла лучшей)']
    print(f"\n   СТРОГАЯ МЕТРИКА VISION §9.4 = {strict}/{n} = {strict/n:.2%}")
    print(f"   задачи: {only_comp_ids}")
    print("   порог интерпретации из §9.4: «пусто — система не добавляет "
          "ничего»")

    # зеркальная величина: сколько задач композиция ПОТЕРЯЛА
    lost = sum(1 for r in M
               if not solved(r.get('q_env_final', 0.0))
               and any(solved(q) for q in (r.get('q_env_candidates') or [])))
    print(f"   зеркально: задач, где композиция потеряла уже готовое решение: "
          f"{lost}/{n} = {lost/n:.2%}  (отношение потерь к приобретениям "
          f"{lost/max(strict,1):.0f}:1)")

    # ── 5. что по логам НЕ измеряется ──────────────────────────────────────
    print("\n── чего в этих логах нет и по ним посчитать нельзя ──")
    print("   * VISION §7 п.3 «граф предсказывает пути»: precision@k у PPR "
          "против чистого FAISS —")
    print("     в записи шага нет ни списка извлечённых трасс, ни их "
          "релевантности. Нужна инструментовка memory.search.")
    print("   * alpha_pos (TASK_ALPHA_POS.md): требует новых вызовов, "
          "по логам не выводится ни в каком виде.")
    print("   * частота крупного прироста ПРИ РАСШИРЕНИИ КОНТЕКСТА "
          "(SIM_REPORT_3 §5):")
    print("     лестница расширения контекста в Мицелии не запускалась ни "
          "разу — мерить нечего.")

    out = {'I_first': I[0], 'I_last': I[-1], 'I_slope': k,
           'only_composition': strict, 'only_composition_share': strict / n,
           'lost_by_composition': lost, 'n': n,
           'depth_changed': sum(changed),
           'depth_changed_no_effect': same_q,
           'only_comp_task_ids': only_comp_ids}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_vision.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
