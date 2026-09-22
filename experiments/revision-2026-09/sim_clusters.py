# -*- coding: utf-8 -*-
"""Кластеры весов + кривая Гильберта + отбор спинового стекла: офлайн-проверка на трассе.

Прежние замеры этой линии ошибались в постановке, и здесь каждая ошибка исправлена явно:

  единица      срез (слой, эксперт) целиком — 12.6 МиБ у gpt-oss в MXFP4. Не каналы (§4.52,
               §4.64) и не перекрывающиеся псевдоэксперты (§4.87): кластеры — РАЗБИЕНИЕ,
               отрезки порядка раскладки фиксированной длины S.
  утечка       граф со-обращений, вложение, порядок и связи J учатся на первой половине трассы;
               меряется только вторая. В hilbert_nd.py (§4.72) вложение считалось по той же
               трассе, на которой мерили.
  мера         промахи ТЁПЛОГО кэша заданной ёмкости и цена их обслуживания, а не отрезки
               объединения эпохи при пустом кэше (§4.59, §4.72).
  цена         промах по требованию обслуживается мелкими порциями — в деле 128 МБ/с (§4.105);
               явное чтение — крупным блоком 367 МБ/с (§4.103) плюс REQ_MS на запрос. Прежняя
               модель считала позиционирование и паспортную последовательную скорость.
  оракул       настоящий Белади для кэша по требованию, а не объединение эпохи, усечённое по
               частоте за всю трассу (tiersim2.py, §4.58).
  контроли     случайный порядок той же длины отрезков; одномерный Фидлер; исходный порядок
               (слой, эксперт); у отбора стекла — тот же отбор без связей (lam = 0).

Политики:
  lru          кэш по требованию, как у ОС; промахи по 128 МБ/с
  belady       потолок кэша по требованию при той же цене промаха
  exact        промах выбранного эксперта читается явно одним запросом на отрезок раскладки
               (подкачка только нерезидентного, §4.105) — без предсказания
  around       промах тянет весь отрезок раскладки длины S, в котором лежит (кластер)
  ising        раз в EPOCH токенов резидентный набор R = RES_FRAC * ёмкость выбирается по
               теплоте h (EMA спроса) и связям J (положительная PMI со-обращений в окне),
               недостающее из R читается явно, R защищён от вытеснения до следующей границы;
               остальное — как exact

Запуск: python sim_clusters.py <трасса MYCSIG0> [top_k=4]
Переменные: WIN (окно со-обращений, 4), EPOCH (16), RES_FRAC (0.7), CAPS ("0.28,0.45,0.6"),
WARM (200 токенов прогрева перед счётом), UNIT_MIB (12.6).
"""
import os
import struct
import sys
import time

import numpy as np

FAULT_MBS = 128.0
READ_MBS = 367.0
REQ_MS = 0.3
UNIT_MB = float(os.environ.get("UNIT_MIB", "12.6")) * 1.048576
WIN = int(os.environ.get("WIN", "4"))
EPOCH = int(os.environ.get("EPOCH", "16"))
RES_FRAC = float(os.environ.get("RES_FRAC", "0.7"))
WARM = int(os.environ.get("WARM", "200"))
CAPS = [float(x) for x in os.environ.get("CAPS", "0.28,0.45,0.6").split(",")]
BIG = 1 << 40


# --- трасса --------------------------------------------------------------------
def load(path, top_k):
    per = {}
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic[:7] == b"MYCSIG0", magic
        width, _ = struct.unpack("<ii", f.read(8))
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            il, nt = struct.unpack("<ii", h)
            raw = f.read(nt * width * 4)
            if len(raw) < nt * width * 4:
                break
            P = np.frombuffer(raw, dtype=np.float32).reshape(nt, width)
            per.setdefault(il, []).append(np.argpartition(-P, top_k - 1, axis=1)[:, :top_k])
    per = {il: np.concatenate(v, 0) for il, v in per.items()}
    full = max(v.shape[0] for v in per.values())
    layers = [il for il in sorted(per) if per[il].shape[0] >= 0.9 * full]
    dropped = sorted(set(per) - set(layers))
    T = min(per[il].shape[0] for il in layers)
    sel = np.stack([per[il][:T] for il in layers], axis=1).astype(np.int64)     # [T, L, k]
    return sel, width, layers, dropped


def incidence(sel, n_exp):
    T, L, k = sel.shape
    X = np.zeros((T, L * n_exp), np.float32)
    cols = (np.arange(L)[None, :, None] * n_exp + sel).reshape(T, -1)
    X[np.repeat(np.arange(T), cols.shape[1]), cols.ravel()] = 1.0
    return X


# --- структура, выученная на первой половине ----------------------------------------
def couplings(X, win):
    cs = np.vstack([np.zeros((1, X.shape[1]), np.float32), np.cumsum(X, 0)])
    Xw = ((cs[win:] - cs[:-win]) > 0).astype(np.float32)
    n = Xw.shape[0]
    C = Xw.T @ Xw
    c = np.diag(C).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.log((C / n) / np.maximum(np.outer(c, c) / (n * n), 1e-12))
    J[~np.isfinite(J)] = 0.0
    J[C < 3] = 0.0                      # меньше трёх совместных окон — связи нет
    np.fill_diagonal(J, 0.0)
    return np.maximum(J, 0.0), c


def spectral(W, k):
    d = W.sum(1)
    d[d <= 0] = 1.0
    dm = 1.0 / np.sqrt(d)
    Lap = np.eye(len(W)) - dm[:, None] * W * dm[None, :]
    w, v = np.linalg.eigh(Lap)
    return w, v[:, 1:k + 1]


def hilbert_nd(coords, order, ndim):
    """Индекс Гильберта (Skilling) — та же реализация, что прошла самопроверку в hilbert_nd.py."""
    X = [int(c) for c in coords]
    M = 1 << (order - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(ndim):
            if X[i] & Q:
                X[0] ^= P
            else:
                t = (X[0] ^ X[i]) & P
                X[0] ^= t
                X[i] ^= t
        Q >>= 1
    for i in range(1, ndim):
        X[i] ^= X[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if X[ndim - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(ndim):
        X[i] ^= t
    idx = 0
    for b in range(order - 1, -1, -1):
        for i in range(ndim):
            idx = (idx << 1) | ((X[i] >> b) & 1)
    return idx


def rank_grid(V, order):
    side = (1 << order) - 1
    out = np.empty(V.shape, np.int64)
    for j in range(V.shape[1]):
        out[:, j] = np.round(np.argsort(np.argsort(V[:, j], kind="stable")) / max(len(V) - 1, 1) * side)
    return out


def layouts(J, L, n_exp, seed=0):
    U = L * n_exp
    out = {"исходный (слой,эксп)": np.arange(U),
           "случайный": np.random.default_rng(seed).permutation(U)}
    w, V = spectral(J, 3)
    out["Фидлер 1D"] = np.argsort(V[:, 0], kind="stable")
    q = rank_grid(V[:, :3], 8)
    out["Гильберт 3D"] = np.argsort(np.array([hilbert_nd(q[i], 8, 3) for i in range(U)]), kind="stable")
    inside = []
    for l in range(L):
        idx = np.arange(l * n_exp, (l + 1) * n_exp)
        _, v = spectral(J[np.ix_(idx, idx)], 1)
        inside.append(idx[np.argsort(v[:, 0], kind="stable")])
    out["внутри слоя по Фидлеру"] = np.concatenate(inside)
    return out, w


# --- симуляция кэша -------------------------------------------------------------
def runs_of(positions):
    if len(positions) == 0:
        return 0
    p = np.sort(positions)
    return 1 + int(np.count_nonzero(np.diff(p) > 1))


def simulate(sel, t0, n_exp, cap, pos, S, policy, J=None, lam=0.0):
    T, L, k = sel.shape
    U = L * n_exp
    X = incidence(sel, n_exp).astype(bool)
    need_all = (np.arange(L)[None, :, None] * n_exp + sel).reshape(T, -1)

    nxt_after = None
    if policy == "belady":
        nxt_after = np.empty((T, U), np.int64)
        nxt = np.full(U, BIG, np.int64)
        for t in range(T - 1, -1, -1):
            nxt_after[t] = nxt
            nxt[X[t]] = t

    in_cache = np.zeros(U, bool)
    last = np.full(U, -BIG, np.int64)
    fresh = np.zeros(U, bool)           # подтянуто заранее и ещё не использовано
    pinned = np.zeros(U, bool)
    heat = np.zeros(U, np.float64)
    unit_at = np.argsort(pos)
    st = dict(refs=0, hits=0, fault=0, read=0, runs=0, wasted=0, pre=0, pre_used=0)
    start = max(t0 - WARM, 0)

    def admit(units, t, counted, protect):
        units = units[~in_cache[units]]
        if len(units) == 0:
            return
        free = cap - int(in_cache.sum())
        over = len(units) - free
        if over > 0:
            cand = np.flatnonzero(in_cache & ~protect & ~pinned)
            if len(cand) < over:
                cand = np.flatnonzero(in_cache & ~protect)
            if policy == "belady":
                key = -nxt_after[t][cand]
            else:
                key = last[cand]
            victims = cand[np.argpartition(key, over - 1)[:over]] if over < len(cand) else cand
            if counted:
                st["wasted"] += int(np.count_nonzero(fresh[victims]))
            in_cache[victims] = False
            fresh[victims] = False
        in_cache[units] = True

    for t in range(start, T):
        counted = t >= t0
        need = need_all[t]
        protect = np.zeros(U, bool)
        protect[need] = True

        if policy == "ising" and (t - start) % EPOCH == 0 and t > start:
            n_res = int(RES_FRAC * cap)
            s = np.zeros(U, bool)
            s[np.argpartition(-heat, n_res - 1)[:n_res]] = True
            for _ in range(5):
                coup = J @ (s * heat) / max(n_res, 1)
                score = heat + lam * coup
                s[:] = False
                s[np.argpartition(-score, n_res - 1)[:n_res]] = True
            pinned[:] = s
            want = np.flatnonzero(s & ~in_cache)
            if len(want):
                if counted:
                    st["read"] += len(want)
                    st["runs"] += runs_of(pos[want])
                    st["pre"] += len(want)
                fresh[want] = True
                admit(want, t, counted, protect | s)
                last[want] = t - 1

        hit = in_cache[need]
        miss = need[~hit]
        if counted:
            st["refs"] += len(need)
            st["hits"] += int(hit.sum())
            st["pre_used"] += int(np.count_nonzero(fresh[need] & hit))
        fresh[need] = False

        if len(miss):
            if policy in ("lru", "belady"):
                if counted:
                    st["fault"] += len(miss)
                admit(miss, t, counted, protect)
            elif policy in ("exact", "ising"):
                if counted:
                    st["read"] += len(miss)
                    st["runs"] += runs_of(pos[miss])
                admit(miss, t, counted, protect)
            elif policy == "around":
                chunks = np.unique(pos[miss] // S)
                region = unit_at[np.clip((chunks[:, None] * S + np.arange(S)[None, :]).ravel(), 0, U - 1)]
                region = np.unique(region)
                todo = region[~in_cache[region]]
                if counted:
                    st["read"] += len(todo)
                    st["runs"] += runs_of(pos[todo])
                    st["pre"] += len(todo) - len(miss)
                extra = np.setdiff1d(todo, miss, assume_unique=True)
                fresh[extra] = True
                admit(todo, t, counted, protect)
                last[extra] = t - 1

        last[need] = t
        heat *= 0.9
        heat[need] += 0.1

    n_tok = T - t0
    fault_ms = st["fault"] * UNIT_MB / FAULT_MBS * 1000 / n_tok
    read_ms = st["read"] * UNIT_MB / READ_MBS * 1000 / n_tok + st["runs"] * REQ_MS / n_tok
    return dict(hit=st["hits"] / st["refs"], fault_mb=st["fault"] * UNIT_MB / n_tok,
                read_mb=st["read"] * UNIT_MB / n_tok, runs=st["runs"] / n_tok,
                wasted_mb=st["wasted"] * UNIT_MB / n_tok,
                pre_acc=(st["pre_used"] / st["pre"]) if st["pre"] else float("nan"),
                io_ms=fault_ms + read_ms)


def main():
    path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t_start = time.time()
    sel, n_exp, layers, dropped = load(path, top_k)
    T, L, _ = sel.shape
    U = L * n_exp
    t0 = T // 2
    print("трасса %s: слоёв %d (отброшены неполные %s), экспертов %d, top-%d, токенов %d"
          % (os.path.basename(path), L, dropped, n_exp, top_k, T))
    print("учёба на токенах 0..%d, замер на %d..%d, прогрев %d; срез %.1f МБ, окно со-обращений %d"
          % (t0 - 1, t0, T - 1, WARM, UNIT_MB, WIN))
    print("рабочий набор токена %d срезов = %.0f МБ\n" % (L * top_k, L * top_k * UNIT_MB))

    Xtr = incidence(sel[:t0], n_exp)
    J, cnt = couplings(Xtr, WIN)
    lays, w = layouts(J, L, n_exp)
    print("спектр нормированного лапласиана: " + "  ".join("%.4f" % x for x in w[:5]))
    print("срезов без единого обращения на учебной половине: %d из %d\n" % (int((cnt == 0).sum()), U))

    for frac in CAPS:
        cap = int(frac * U)
        print("=== ёмкость %.0f %% (%d срезов, %.1f ГБ) ===" % (100 * frac, cap, cap * UNIT_MB / 1000))
        print("%-40s %7s %9s %9s %8s %9s %8s %9s %7s"
              % ("политика / раскладка", "попад.", "промах МБ", "чтение МБ", "запросов",
                 "впустую МБ", "точн.", "мс/токен", "к LRU"))
        base = simulate(sel, t0, n_exp, cap, lays["исходный (слой,эксп)"], 1, "lru")
        rows = [("lru (как у ОС)", base),
                ("belady (потолок кэша)", simulate(sel, t0, n_exp, cap, lays["исходный (слой,эксп)"], 1, "belady"))]
        for name in ("исходный (слой,эксп)", "внутри слоя по Фидлеру", "Гильберт 3D"):
            rows.append(("exact / " + name, simulate(sel, t0, n_exp, cap, lays[name], 1, "exact")))
        for S in (4, 8):
            for name in ("случайный", "исходный (слой,эксп)", "Фидлер 1D", "Гильберт 3D"):
                rows.append(("around S=%d / %s" % (S, name),
                             simulate(sel, t0, n_exp, cap, lays[name], S, "around")))
        for lam in (0.0, 1.0, 4.0):
            rows.append(("ising lam=%.0f / Гильберт 3D" % lam,
                         simulate(sel, t0, n_exp, cap, lays["Гильберт 3D"], 1, "ising", J, lam)))
        for name, r in rows:
            print("%-40s %6.1f%% %9.1f %9.1f %8.1f %9.1f %8.2f %9.1f %6.2fx"
                  % (name, 100 * r["hit"], r["fault_mb"], r["read_mb"], r["runs"], r["wasted_mb"],
                     r["pre_acc"], r["io_ms"], base["io_ms"] / max(r["io_ms"], 1e-9)))
        print()
    print("готово за %.0f с" % (time.time() - t_start))
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
