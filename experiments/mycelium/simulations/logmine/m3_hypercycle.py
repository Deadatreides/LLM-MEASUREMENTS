"""Ж3+Ж4. Гиперцикл по логам: фактическая ставка мутации, каст-контраст, шум k,
контрфактический квантователь с остатками, раунды до различимости.

Задание:
  DIAGNOSE_HYPERCYCLE.md §4 — «1. Проверить фактическую ставку мутации в
  прогоне, давшем 0.0174 — по логам или по конфигу прогона, не по
  settings.yaml. 2. Пересчитать каст-контраст по логам
  (recompute_hypercycle.py). 3. Поставить квантователь с накоплением остатков.»

Здесь ставка мутации не угадывается по равновесию, а ВОССТАНАВЛИВАЕТСЯ ТОЧНО:
в записи каждого шага есть x_before, x_after, k и lambda_L, а формула обновления
детерминирована (core/hypercycle.py::update). Один неизвестный параметр —
достаточно решить уравнение.
"""
from __future__ import annotations

import json
import os

import numpy as np

from common import ROOT, iter_records

CASTES = ['G', 'C', 'S']
ALPHA = 0.0207
PHI_MIN = 0.05
X_MIN, X_MAX = 0.04, 0.87
K_ACT = 6
TAG = 'sh2_200'


def step_forward(x, k, lam, mut):
    """Точная копия Hypercycle.update шагов 2-4 (k уже обновлён)."""
    a = float(np.clip(ALPHA * (1.0 + 0.5 * abs(lam)), 0.001, 0.5))
    phi = sum(a * k[CASTES[j]] * x[CASTES[j]] * x[CASTES[j - 1]]
              for j in range(3))
    phi = max(phi, PHI_MIN)
    xn = {}
    for j, c in enumerate(CASTES):
        xn[c] = x[c] * (1 + a * k[c] * x[CASTES[j - 1]]) / (1 + a * phi)
    tot = sum(xn.values())
    if tot > 0:
        for c in CASTES:
            xn[c] = float(np.clip(xn[c] / tot, X_MIN, X_MAX))
    else:
        xn = {c: 1 / 3 for c in CASTES}
    for c in CASTES:
        xn[c] = (1.0 - mut) * xn[c] + mut / 3
    tot = sum(xn.values())
    return {c: v / tot for c, v in xn.items()}


def main():
    recs = [r for r in iter_records()
            if r['_tag'] == TAG and r['condition'] == 'M'
            and r.get('status') == 'ok' and r.get('hypercycle_x_before')]
    print(f"шагов с траекторией гиперцикла: {len(recs)}\n")

    # ── 1. фактическая ставка мутации ──────────────────────────────────────
    grid = np.concatenate([np.linspace(0.0, 0.06, 601), [0.005, 0.03]])
    errs = []
    for mut in grid:
        e = 0.0
        for r in recs:
            pred = step_forward(r['hypercycle_x_before'], r['hypercycle_k'],
                                r.get('lambda_L', 0.0), float(mut))
            for c in CASTES:
                e += (pred[c] - r['hypercycle_x_after'][c]) ** 2
        errs.append(e)
    errs = np.array(errs)
    best = float(grid[int(errs.argmin())])
    rms = float(np.sqrt(errs.min() / (3 * len(recs))))
    print("── Ж3. фактическая ставка мутации (обратная задача по x_before→x_after) ──")
    print(f"   найденная ставка = {best:.5f}   RMS невязки {rms:.3e}")
    for probe in (0.005, 0.03):
        i = int(np.argmin(np.abs(grid - probe)))
        print(f"   невязка при mut={probe:<6} : "
              f"{np.sqrt(errs[i]/(3*len(recs))):.3e}")
    print(f"   в settings.yaml: hypercycle_mutation=true, rate=0.005")
    print(f"   вывод: ставка в прогоне {'СОВПАДАЕТ' if abs(best-0.005)<5e-4 else 'НЕ совпадает'}"
          f" с конфигом")

    # ── наблюдавшееся отклонение x от 1/3 ──────────────────────────────────
    devs = [max(abs(r['hypercycle_x_after'][c] - 1 / 3) for c in CASTES)
            for r in recs]
    print(f"\n   max|x-1/3| : финал {devs[-1]:.4f}, максимум по прогону "
          f"{max(devs):.4f}, медиана {np.median(devs):.4f}")
    print(f"   порог жёсткого квантователя round(x*K), K={K_ACT}: "
          f"{1/(2*K_ACT):.4f}")

    # ── 2. каст-контраст по k, который РЕАЛЬНО видит гиперцикл ─────────────
    K = np.array([[r['hypercycle_k'][c] for c in CASTES] for r in recs])
    print("\n── Ж4.1. k, который реально видит гиперцикл (по кастам) ──")
    for i, c in enumerate(CASTES):
        print(f"   {c}: среднее {K[:, i].mean():.4f}  "
              f"ст.откл по шагам {K[:, i].std(ddof=1):.4f}  "
              f"финал {K[-1, i]:.4f}")
    mean = K.mean(axis=0)
    ratio = mean.max() / max(mean.min(), 1e-9)
    ratio_fin = K[-1].max() / max(K[-1].min(), 1e-9)
    print(f"   КАСТ-КОНТРАСТ по среднему = {ratio:.3f}   по финалу = "
          f"{ratio_fin:.3f}   (порог при K_ACT=6 ≈ 1.62)")
    print(f"   для сравнения: спред baseline моделей даёт 1.42 — "
          f"это ДРУГАЯ величина")

    # ── 2b. закрывает ли измеренный контраст наблюдавшиеся 0.0174 ──────────
    # DIAGNOSE_HYPERCYCLE §1 объяснял 0.0174 ставкой мутации 0.03 при
    # контрасте 1.42. Ставка оказалась 0.005 (см. выше), значит объяснение
    # должно давать сам контраст. Прогоняем настоящую формулу до равновесия.
    print("\n── Ж4.1b. равновесие при ИЗМЕРЕННОМ k и ставке 0.005 ──")
    k_fix = {c: float(mean[i]) for i, c in enumerate(CASTES)}
    for mut in (0.005, 0.03):
        x = {c: 1 / 3 for c in CASTES}
        for _ in range(20000):
            x = step_forward(x, k_fix, 0.0, mut)
        d = max(abs(x[c] - 1 / 3) for c in CASTES)
        print(f"   mut={mut:<6} равновесие max|x-1/3| = {d:.4f}")
    # и при контрасте, который предполагал отчёт
    scaled = dict(k_fix)
    lo0, hi0 = mean.min(), mean.max()
    for target in (1.42, 1.62):
        # растянуть k вокруг среднего до нужного контраста
        mid = (lo0 + hi0) / 2
        s = (target - 1) / max(hi0 / lo0 - 1, 1e-9)
        kk = {c: mid + (k_fix[c] - mid) * s for c in CASTES}
        x = {c: 1 / 3 for c in CASTES}
        for _ in range(20000):
            x = step_forward(x, kk, 0.0, 0.005)
        d = max(abs(x[c] - 1 / 3) for c in CASTES)
        print(f"   контраст {target}: равновесие {d:.4f} "
              f"{'(порог 0.0833 взят)' if d >= 1/(2*K_ACT) else ''}")

    # ── 2c. какой контраст нужен ПРИ НАСТОЯЩЕЙ alpha ───────────────────────
    # SIM_REPORT_2 §4 даёт «минимальный контраст 1.59-1.62», но его ОДУ
    # использовала шаг a=0.05, а в коде alpha=0.0207 (config/settings.yaml).
    # Равновесие определяется отношением скорости отбора (∝ alpha) к ставке
    # мутации, поэтому число из отчёта к этому коду не относится.
    global ALPHA
    print("\n── Ж4.1c. требуемый контраст при разных alpha (mut=0.005) ──")
    saved_alpha = ALPHA
    mid = float(mean.mean())
    for a_val in (0.05, ALPHA):
        ALPHA = a_val
        need = None
        row = []
        for target in (1.27, 1.42, 1.62, 1.8, 2.0, 2.1, 2.2, 3.0):
            s = (target - 1) / max(mean.max() / mean.min() - 1, 1e-9)
            kk = {c: mid + (float(mean[i]) - mid) * s
                  for i, c in enumerate(CASTES)}
            x = {c: 1 / 3 for c in CASTES}
            for _ in range(40000):
                x = step_forward(x, kk, 0.0, 0.005)
            d = max(abs(x[c] - 1 / 3) for c in CASTES)
            row.append(f"{target}:{d:.4f}")
            if need is None and d >= 1 / (2 * K_ACT):
                need = target
        print(f"   alpha={a_val:<7} " + "  ".join(row))
        print(f"      порог 0.0833 берётся начиная с контраста ~{need}")
    ALPHA = saved_alpha

    # ── 3. шум k → допустимое заострение beta ──────────────────────────────
    sig = float(K.std(axis=0, ddof=1).mean())
    gap = float(mean.max() - mean.min())
    print("\n── Ж4.2. шум измерения k (нужен для выбора beta) ──")
    print(f"   ст.откл по шагам {sig:.4f}, шагов {len(K)}, "
          f"SE среднего {sig/np.sqrt(len(K)):.4f}")
    print(f"   разрыв между кастами {gap:.4f} → сигнал/шум {gap/max(sig,1e-9):.2f}")
    for beta in (1.0, 1.5, 2.0, 3.0):
        cap = gap / (2 * np.sqrt(2)) / beta
        print(f"   beta={beta}: допустимый шум <= {cap:.4f} "
              f"{'— проходит' if sig <= cap else '— НЕ проходит'}")

    # ── 4. контрфактический квантователь ───────────────────────────────────
    xs = [np.array([r['hypercycle_x_after'][c] for c in CASTES]) for r in recs]
    hard = np.sum([np.round(x * K_ACT) for x in xs], axis=0)
    err = np.zeros(3)
    soft = np.zeros(3)
    for x in xs:
        acc = x * K_ACT + err
        take = np.round(acc)
        err = acc - take
        soft += take
    print("\n── Ж4.3. контрфактическая раскладка слотов за 200 раундов ──")
    print(f"   {'каста':>6} {'жёстко':>9} {'с остатками':>13} {'разница':>9}")
    for i, c in enumerate(CASTES):
        print(f"   {c:>6} {hard[i]:>9.0f} {soft[i]:>13.0f} "
              f"{soft[i]-hard[i]:>+9.0f}")
    if np.allclose(hard, hard[0]):
        print("   жёсткий квантователь раскладку НЕ сдвинул — порог не взят")
    if not np.allclose(soft, soft[0]):
        print("   с накоплением остатков раскладка сдвигается")

    # фактическая раскладка из логов — сверка
    alloc = [r.get('alloc_log') for r in recs if r.get('alloc_log')]
    import collections
    cnt = collections.Counter(a.split('alloc:')[-1].strip() for a in alloc)
    print(f"   фактические раскладки в логе: {dict(cnt)}")

    # ── 5. раунды до различимости ──────────────────────────────────────────
    dev = float(np.abs(np.mean(xs, axis=0) - 1 / 3).max())
    p = 1 / 3 + dev
    print("\n── Ж4.4. раундов до отличимости от мультиномиального шума ──")
    print(f"   среднее отклонение |x-1/3| = {dev:.4f}")
    for z in (2, 3):
        n = 1
        while n < 500000:
            mu_ = n * K_ACT * dev
            sd = np.sqrt(n * K_ACT * p * (1 - p))
            if mu_ >= z * sd:
                break
            n += 1
        print(f"   до {z} сигм: {n} раундов")

    # ── 6. пол: вычитание общей подложки ───────────────────────────────────
    print("\n── Ж4.5. вычитание пола: во что превращается контраст ──")
    for floor in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        lo, hi = mean.min() - floor, mean.max() - floor
        if lo <= 0.01:
            print(f"   пол {floor:.2f} → каста обнуляется, не годится")
            continue
        print(f"   пол {floor:.2f} → контраст {hi/lo:.2f} "
              f"{'(проходит порог 1.62)' if hi/lo >= 1.62 else ''}")

    out = {'mut_fitted': best, 'mut_rms': rms,
           'k_mean': {c: float(mean[i]) for i, c in enumerate(CASTES)},
           'contrast_mean': float(ratio), 'contrast_final': float(ratio_fin),
           'k_noise': sig, 'k_gap': gap,
           'dev_final': devs[-1], 'dev_max': float(max(devs)),
           'alloc_counts': dict(cnt)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'out_hypercycle.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(f"\nсохранено: {path}")


if __name__ == '__main__':
    main()
