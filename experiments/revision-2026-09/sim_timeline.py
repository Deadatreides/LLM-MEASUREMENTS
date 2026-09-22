# -*- coding: utf-8 -*-
"""Вертикальные кластеры как упредитель внутри токена: событийная симуляция по слоям.

sim_clusters.py обращался с токеном как с одним моментом: все его срезы нужны сразу. Поэтому
межслойное упреждение — самый сильный из измеренных сигналов (§4.81: выбор слоя L по слоям
0..L-1 того же токена угадывается с точностью 0.699, с заделом в 4 слоя — 0.658) — в той
симуляции проявиться не мог по построению. А именно там живут вертикальные кластеры исходного
замысла: псевдоэксперт как срез модели сквозь все слои.

Здесь токен идёт по слоям во времени:
  * слой j начинается, роутер отдаёт выбор; нужные срезы обязаны быть в памяти, иначе слой ждёт;
  * один накопитель, одна очередь: запрос по требованию встаёт в голову очереди, упреждающие —
    в хвост; незавершённые упреждающие запросы токена снимаются, когда токен кончился;
  * кэш ёмкостью CAP срезов, вытеснение по давности; срезы, которых ждёт текущий слой, не
    вытесняются; подтянутое заранее занимает место наравне с остальным;
  * счёт слоя — LAYER_MS, чтение — по полосе накопителя.

Цена чтения: явный запрос крупным блоком READ_MBS (367 МБ/с, §4.103) + REQ_MS; промах по
требованию страничным механизмом ОС — FAULT_MBS (128 МБ/с в деле, §4.105).

Упредители (у всех, кроме none-fault, запрос по требованию явный):
  none-fault   нет упреждения, промахи по страницам — как сейчас у ОС
  none-exact   нет упреждения, промах выбранного читается явным запросом
  oracle       настоящий выбор слоёв j+1..j+LOOK — потолок упреждения с этим заделом
  prev-token   выбор тех же будущих слоёв на прошлом токене — чистая история
  table        таблица совместного выбора внутри токена: для будущего слоя b берутся PER_LAYER
               срезов с наибольшей суммой P(v|u) по срезам u последних CTX_LAYERS слоёв
  cluster S    вертикальные кластеры: разбиение на отрезки длины S порядка Гильберта над
               спектральным вложением графа совместного выбора внутри токена; выбранный срез
               тянет членов своего кластера из слоёв j+1..j+LOOK
  cluster-rnd  то же при случайном порядке (контроль: есть ли в кластерах информация)

Структура учится на первой половине трассы, меряется вторая. Запуск:
    python sim_timeline.py <трасса MYCSIG0> [top_k=4]
Переменные: LAYER_MS (10), LOOK (4), PER_LAYER (= top_k), CTX_LAYERS (4), CAPS, WARM (200),
UNIT_MIB (12.6), S_LIST ("16,32").
"""
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_clusters as sc                                    # noqa: E402

READ_MBS = float(os.environ.get("READ_MBS", "367"))       # 11.09: замерено 366–380 МиБ/с
FAULT_MBS = float(os.environ.get("FAULT_MBS", "128"))     # 11.09: замерено 335 МиБ/с, а не 128
REQ_MS = 0.3
UNIT_MB = float(os.environ.get("UNIT_MIB", "12.6")) * 1.048576
LAYER_MS = float(os.environ.get("LAYER_MS", "10"))
LOOK = int(os.environ.get("LOOK", "4"))
CTX_LAYERS = int(os.environ.get("CTX_LAYERS", "4"))
WARM = int(os.environ.get("WARM", "200"))
CAPS = [float(x) for x in os.environ.get("CAPS", "0.28,0.45,0.6").split(",")]
S_LIST = [int(x) for x in os.environ.get("S_LIST", "16,32").split(",")]

ABSENT, QUEUED, READING, PRESENT = 0, 1, 2, 3


class Store:
    """Кэш срезов + один накопитель с очередью. Время — в миллисекундах."""

    def __init__(self, U, cap, demand_mbs):
        self.cap = cap
        self.demand_mbs = demand_mbs
        self.state = np.zeros(U, np.int8)
        self.last = np.full(U, -1e18)
        self.fresh = np.zeros(U, bool)
        self.protect = np.zeros(U, bool)
        self.n_present = 0
        self.queue = deque()             # [unit, kind, t_enq, token]
        self.inflight = None             # (unit, t_end, kind)
        self.free_at = 0.0
        self.counting = False
        self.st = dict(dem=0, pre=0, pre_done=0, pre_used=0, wasted=0, cancelled=0)

    def _make_present(self, u, t, kind):
        if self.n_present >= self.cap:
            cand = np.flatnonzero((self.state == PRESENT) & ~self.protect)
            if len(cand) == 0:
                raise RuntimeError("ёмкость меньше набора одного слоя")
            v = cand[np.argmin(self.last[cand])]
            if self.fresh[v] and self.counting:
                self.st["wasted"] += 1
            self.state[v] = ABSENT
            self.fresh[v] = False
            self.n_present -= 1
        self.state[u] = PRESENT
        self.n_present += 1
        self.last[u] = t
        if kind == "p":
            self.fresh[u] = True
            if self.counting:
                self.st["pre_done"] += 1

    def advance(self, T):
        """Довести накопитель до момента T: завершить и начать всё, что успевает."""
        while True:
            if self.inflight is not None:
                u, t_end, kind = self.inflight
                if t_end > T:
                    return
                self.inflight = None
                self.free_at = t_end
                self._make_present(u, t_end, kind)
            if not self.queue:
                return
            u, kind, t_enq, _tok = self.queue[0]
            start = max(self.free_at, t_enq)
            if start > T:
                return
            self.queue.popleft()
            if self.state[u] != QUEUED:
                continue
            mbs = READ_MBS if kind == "p" else self.demand_mbs
            self.state[u] = READING
            self.inflight = (u, start + UNIT_MB / mbs * 1000.0 + REQ_MS, kind)
            if self.counting:
                self.st["pre" if kind == "p" else "dem"] += 1

    def prefetch(self, units, t, token):
        for u in units:
            if self.state[u] == ABSENT:
                self.state[u] = QUEUED
                self.queue.append([int(u), "p", t, token])

    def cancel_token(self, token):
        keep = deque()
        for item in self.queue:
            if item[1] == "p" and item[3] == token and self.state[item[0]] == QUEUED:
                self.state[item[0]] = ABSENT
                if self.counting:
                    self.st["cancelled"] += 1
            else:
                keep.append(item)
        self.queue = keep

    def demand(self, units, t, token):
        """Слой требует units в момент t. Возвращает момент, когда все они в памяти."""
        self.protect[units] = True
        self.advance(t)
        front = []
        for u in units:
            s = self.state[u]
            if s == QUEUED:
                for i, item in enumerate(self.queue):          # повышение до требования
                    if item[0] == u:
                        del self.queue[i]
                        break
                front.append(u)
            elif s == ABSENT:
                self.state[u] = QUEUED
                front.append(u)
        for u in reversed(front):
            self.queue.appendleft([int(u), "d", t, token])
        now = t
        while not all(self.state[u] == PRESENT for u in units):
            if self.inflight is None:
                self.advance(now)
                if self.inflight is None:
                    raise RuntimeError("накопитель простаивает, а срезы не пришли")
            now = max(now, self.inflight[1])
            self.advance(now)
        for u in units:
            if self.fresh[u] and self.counting:
                self.st["pre_used"] += 1
            self.fresh[u] = False
            self.last[u] = now
        self.protect[units] = False
        return now


def vertical_clusters(Xtr, L, n_exp, S, seed=None):
    """Разбиение на отрезки длины S порядка Гильберта над вложением графа совместного выбора."""
    U = L * n_exp
    if seed is not None:
        order = np.random.default_rng(seed).permutation(U)
    else:
        C = Xtr.T @ Xtr
        c = np.diag(C).copy()
        W = C / np.sqrt(np.maximum(np.outer(c, c), 1.0))
        np.fill_diagonal(W, 0.0)
        _, V = sc.spectral(W, 3)
        q = sc.rank_grid(V[:, :3], 8)
        order = np.argsort(np.array([sc.hilbert_nd(q[i], 8, 3) for i in range(U)]), kind="stable")
    pos = np.empty(U, np.int64)
    pos[order] = np.arange(U)
    return pos // S


def run(policy, sel, t0, L, n_exp, k, cap, P=None, chunk=None):
    T = sel.shape[0]
    U = L * n_exp
    demand_mbs = FAULT_MBS if policy == "none-fault" else READ_MBS
    store = Store(U, cap, demand_mbs)
    units = sel + (np.arange(L)[None, :, None] * n_exp)           # [T, L, k]
    members = None
    if chunk is not None:
        members = {}
        for u in range(U):
            members.setdefault(int(chunk[u]), []).append(u)
        members = {c: np.array(v) for c, v in members.items()}
    layer_of = np.repeat(np.arange(L), n_exp)
    per_layer = int(os.environ.get("PER_LAYER", str(k)))

    clock = 0.0
    wait_ms, tok_ms = 0.0, 0.0
    n_tok = 0
    for t in range(max(t0 - WARM, 1), T):
        store.counting = t >= t0
        t_start = clock
        for j in range(L):
            lo, hi = j + 1, min(j + LOOK, L - 1)
            if hi >= lo and policy not in ("none-fault", "none-exact"):
                if policy == "oracle":
                    pred = units[t, lo:hi + 1].ravel()
                elif policy == "prev-token":
                    pred = units[t - 1, lo:hi + 1].ravel()
                elif policy == "table":
                    ctx = units[t, max(0, j - CTX_LAYERS + 1):j + 1].ravel()
                    score = P[ctx].sum(0)
                    pred = []
                    for b in range(lo, hi + 1):
                        seg = score[b * n_exp:(b + 1) * n_exp]
                        pred.extend(b * n_exp + np.argpartition(-seg, per_layer - 1)[:per_layer])
                    pred = np.array(pred)
                else:                                              # cluster, cluster-rnd
                    pred = []
                    for u in units[t, j]:
                        m = members[int(chunk[u])]
                        m = m[(layer_of[m] >= lo) & (layer_of[m] <= hi)]
                        pred.extend(m)
                    pred = np.unique(np.array(pred, dtype=np.int64))
                store.prefetch(pred, clock, t)
            ready = store.demand(units[t, j], clock, t)
            if store.counting:
                wait_ms += ready - clock
            clock = ready + LAYER_MS
            store.advance(clock)
        store.cancel_token(t)
        if store.counting:
            tok_ms += clock - t_start
            n_tok += 1
    s = store.st
    return dict(tok_ms=tok_ms / n_tok, wait_ms=wait_ms / n_tok,
                dem_mb=s["dem"] * UNIT_MB / n_tok, pre_mb=s["pre"] * UNIT_MB / n_tok,
                prec=s["pre_used"] / s["pre_done"] if s["pre_done"] else float("nan"),
                wasted_mb=s["wasted"] * UNIT_MB / n_tok, cancelled=s["cancelled"] / n_tok)


def main():
    path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t_begin = time.time()
    sel, n_exp, layers, dropped = sc.load(path, k)
    T, L, _ = sel.shape
    U = L * n_exp
    t0 = T // 2
    print("трасса %s: слоёв %d, экспертов %d, top-%d, токенов %d; учёба 0..%d, замер %d..%d"
          % (os.path.basename(path), L, n_exp, k, T, t0 - 1, t0, T - 1))
    print("счёт слоя %.0f мс (токен без ожиданий %.0f мс), задел %d слоёв, контекст таблицы %d слоёв, "
          "срез %.1f МБ\n" % (LAYER_MS, L * LAYER_MS, LOOK, CTX_LAYERS, UNIT_MB))

    Xtr = sc.incidence(sel[:t0], n_exp)
    C = Xtr.T @ Xtr
    cnt = np.diag(C).copy()
    P = C / np.maximum(cnt[:, None], 1.0)                       # P(v | u) внутри токена
    np.fill_diagonal(P, 0.0)

    chunks = {("cluster S=%d" % S): vertical_clusters(Xtr, L, n_exp, S) for S in S_LIST}
    chunks.update({("cluster-rnd S=%d" % S): vertical_clusters(Xtr, L, n_exp, S, seed=7) for S in S_LIST})

    for frac in CAPS:
        cap = int(frac * U)
        print("=== ёмкость %.0f %% (%d срезов) ===" % (100 * frac, cap))
        print("%-22s %9s %9s %10s %10s %7s %10s %9s %7s"
              % ("упредитель", "мс/токен", "ожидание", "треб. МБ", "упрежд. МБ", "точн.",
                 "впустую МБ", "снято/ток", "к ОС"))
        base = run("none-fault", sel, t0, L, n_exp, k, cap)
        rows = [("none-fault (как ОС)", base),
                ("none-exact", run("none-exact", sel, t0, L, n_exp, k, cap)),
                ("oracle", run("oracle", sel, t0, L, n_exp, k, cap)),
                ("prev-token", run("prev-token", sel, t0, L, n_exp, k, cap)),
                ("table", run("table", sel, t0, L, n_exp, k, cap, P=P))]
        for name, ch in chunks.items():
            rows.append((name, run("cluster", sel, t0, L, n_exp, k, cap, chunk=ch)))
        for name, r in rows:
            print("%-22s %9.1f %9.1f %10.1f %10.1f %7.2f %10.1f %9.1f %6.2fx"
                  % (name, r["tok_ms"], r["wait_ms"], r["dem_mb"], r["pre_mb"], r["prec"],
                     r["wasted_mb"], r["cancelled"], base["tok_ms"] / r["tok_ms"]))
        print()
    print("готово за %.0f с" % (time.time() - t_begin))
    print("RUN-COMPLETE")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
