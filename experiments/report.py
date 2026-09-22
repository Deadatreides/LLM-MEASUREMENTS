"""Plots and tables for the report (§9).

Reads out/*.analysis.json produced by sim.py and writes out/figs/*.png plus
out/tables.md.  The prose of the report is written by hand around these.
"""

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:                 # tables and verdict still work without it
    HAVE_MPL = False

ROOT = Path(__file__).parent
OUT = ROOT / "out"
FIGS = OUT / "figs"
POLICIES = ("STATIC", "LRU", "LFU", "EPOCH", "EPOCH_BAL")
EPOCH_LEN = 15                      # must match sim.EPOCH_LEN
COL = {"STATIC": "#888888", "LRU": "#1f77b4", "LFU": "#2ca02c",
       "EPOCH": "#d62728", "EPOCH_BAL": "#9467bd"}

# §0 verdict table
VERDICT = [
    (0.93, 1.01, "60+ tok/s — схема работает как задумано"),
    (0.83, 0.93, "30–60 tok/s — цель достижима"),
    (0.72, 0.83, "20–30 tok/s — приемлемо, нужен кэш крупнее"),
    (0.60, 0.72, "12–20 tok/s — пограничный"),
    (-1.0, 0.60, "<12 tok/s — ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА"),
]


def verdict(q):
    for lo, hi, txt in VERDICT:
        if lo <= q < hi:
            return txt
    return "?"


def tok_s(q, n_act=144, t_res=6.3, t_miss=1.10):
    """§0 cost model for the GPT-OSS-120B target configuration."""
    ms = t_res + n_act * (1 - q) * t_miss
    return 1000.0 / ms


def load(tag):
    p = OUT / f"{tag}.analysis.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_q_vs_m(res, tag):
    t = res["q_table"]
    Ms = sorted(int(m) for m in t)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for pol in POLICIES:
        y = [t[str(m)][pol]["q_steady"] for m in Ms]
        ax.plot(Ms, y, "o-", color=COL[pol], label=pol, lw=1.8, ms=4)
    # a cache below one token's working set cannot reach q = 1 at all
    ceil = [min(1.0, t[str(m)]["EPOCH"]["q_ceiling"]) for m in Ms]
    ax.plot(Ms, ceil, "k--", lw=1.2, label="ceiling (cache/working set)")
    ax.fill_between(Ms, ceil, 1.0, color="k", alpha=0.06, lw=0)
    iso = res.get("iso_ratio_point")
    if iso:
        ax.axvline(iso["M_percent"], color="k", ls=":", lw=1)
        ax.text(iso["M_percent"], 0.02, f"  iso-ratio\n  M={iso['M_percent']}%",
                fontsize=7, va="bottom")
    for lo, _, _ in VERDICT[:-1]:
        ax.axhline(lo, color="#cccccc", lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel("cache budget M, % of FFN/MoE mass")
    ax.set_ylabel("q  (steady state, last 50 % of trace)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{tag}: {res['trace']['repo']} ({res['trace']['quant']})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}_q_vs_M.png", dpi=150)
    plt.close(fig)


def fig_layers(res, tag):
    H = res["H_layer"]
    qbl = res.get("q_by_layer_M5_EPOCH")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax[0].plot(range(1, len(H) + 1), H, "o-", color="#d62728", ms=4)
    ax[0].set_xlabel("layer boundary l -> l+1")
    ax[0].set_ylabel("H(sig_{l+1} | sig_l), bits")
    ax[0].set_title(f"conditional entropy per layer  (sum = {sum(H):.2f} bits)")
    ax[0].grid(alpha=0.25)
    if qbl:
        ax[1].plot(range(len(qbl)), qbl, "o-", color="#1f77b4", ms=4)
        ax[1].set_xlabel("layer")
        ax[1].set_ylabel("q at M = 5 %, EPOCH")
        ax[1].set_ylim(0, 1)
        ax[1].set_title("hit rate by layer")
        ax[1].grid(alpha=0.25)
    fig.suptitle(tag, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}_layers.png", dpi=150)
    plt.close(fig)


def fig_persist(res, tag):
    ok = res["persistence"]["o_k"]
    ks = sorted(int(k) for k in ok)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(ks, [ok[str(k)] for k in ks], "-", color="#2ca02c", lw=2)
    ax.axhline(0.5, color="k", ls="--", lw=1)
    lp = res["persistence"]["L_persist"]
    if lp:
        ax.axvline(lp, color="k", ls=":", lw=1)
        ax.text(lp, 0.92, f" L_persist = {lp}", fontsize=8)
    ax.set_xlabel("token distance k")
    ax.set_ylabel("median overlap o_k")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{tag}: route persistence")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}_persistence.png", dpi=150)
    plt.close(fig)


def fig_saturation(res, tag):
    c = np.array(res["saturation"], dtype=float)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, (name, col) in enumerate([("unique units", "#1f77b4"),
                                     ("unique transitions", "#ff7f0e"),
                                     ("unique full paths", "#d62728")]):
        ax[i].plot(c[:, 0], c[:, i + 1], "-", color=col, lw=2)
        ax[i].plot([c[0, 0], c[-1, 0]], [c[0, i + 1],
                   c[0, i + 1] * c[-1, 0] / max(c[0, 0], 1)], "k--", lw=0.8,
                   label="linear growth")
        ax[i].set_xlabel("requests processed")
        ax[i].set_ylabel(name)
        ax[i].set_ylim(0, c[-1, i + 1] * 1.15)
        ax[i].legend(fontsize=7)
        ax[i].grid(alpha=0.25)
    fig.suptitle(f"{tag}: saturation (§6.3)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}_saturation.png", dpi=150)
    plt.close(fig)


def fig_gap(res, tag):
    g = res.get("gap")
    if not g:
        return
    ls = sorted(int(k) for k in g)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for q, col in [("q05", "#d62728"), ("q25", "#ff7f0e"), ("q50", "#1f77b4")]:
        ax.plot(ls, [g[str(l)][q] for l in ls], "o-", color=col, label=q, ms=4)
    ax.set_xlabel("layer")
    ax.set_ylabel("logit[rank k] - logit[rank k+1]")
    ax.set_title(f"{tag}: router cut gap (§7)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / f"{tag}_gap.png", dpi=150)
    plt.close(fig)


def md_q_table(res, tag):
    t = res["q_table"]
    Ms = sorted(int(m) for m in t)
    tr = res["trace"]
    lines = [f"#### {tag} — `{tr['repo']}` ({tr['quant']}), "
             f"q в установившемся режиме (последние 50 % трассы)", "",
             f"Всего единиц {tr['units_total']}, активно на токен "
             f"{tr['mean_active_per_token']}, токенов в трассе {tr['tokens']}.", "",
             "| M, % | кэш, единиц | потолок | STATIC | LRU | LFU-decay | "
             "EPOCH | EPOCH-BAL | churn EPOCH |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in Ms:
        row = t[str(m)]
        ch = row["EPOCH"]["churn"]
        chs = "-" if ch is None else f"{ch:.3f}"
        cells = " | ".join(f"{row[p]['q_steady']:.3f}" for p in POLICIES)
        lines.append(f"| {m} | {row['EPOCH']['capacity_units']} | "
                     f"{row['EPOCH']['q_ceiling']:.3f} | {cells} | {chs} |")
    iso = res.get("iso_ratio_point")
    if iso and iso.get("clamped"):
        lines += ["", f"Архитектурно сопоставимый бюджет (кэш/активации = 1.60, "
                      f"как у целевой GPT-OSS-120B) **недостижим**: понадобилось "
                      f"бы M = {iso['wanted_M_percent']} %, больше, чем есть в "
                      f"модели. Единица учёта слишком крупная, чтобы быть "
                      f"разрежённой."]
    elif iso:
        ch = iso.get("churn")
        tail = f", churn = {ch:.3f}" if ch is not None else ""
        lines += ["", f"Архитектурно сопоставимый бюджет "
                      f"(кэш/активации = 1.60, как у целевой GPT-OSS-120B): "
                      f"**M = {iso['M_percent']} %**, q = "
                      f"**{iso['q_steady']:.3f}**{tail}."]
    lines += ["", "Столбец «потолок» — `min(1, кэш / активаций на токен)`: доля "
                  "попаданий не может его превысить ни при какой политике."]
    return "\n".join(lines) + "\n"


def sec_persistence(res, tag):
    ks = [1, 2, 4, 8, 16, 32, 64]
    head = " | ".join(f"k={k}" for k in ks)
    out = [f"**{tag}** — медианное перекрытие `o_k` активных множеств\n",
           f"| множество | L_persist | {head} |",
           f"|---|---|{'---|' * len(ks)}"]

    def row(label, blk):
        ok, lp = blk["o_k"], blk["L_persist"]
        cells = " | ".join(f"{ok[str(k)]:.3f}" for k in ks)
        return f"| {label} | {lp} | {cells} |"

    out.append(row("все единицы", res["persistence"]))
    if "persistence_variable" in res:
        out.append(row("без постоянного ядра", res["persistence_variable"]))
    c = res.get("core")
    if c:
        out.append("")
        out.append(f"Постоянное ядро (единицы, активные не менее чем в "
                   f"{int(c['threshold']*100)} % токенов): **{c['n_core']} единиц**, "
                   f"{100*c['core_of_units']:.1f} % модели и "
                   f"{100*c['core_of_working_set']:.1f} % того, что нужно токену. "
                   f"Оставшиеся **{c['variable_per_token']} единиц на токен** — "
                   f"единственное, на чём политика вытеснения может выиграть "
                   f"или проиграть.")
        out.append("")
        out.append("Разделение ядра и переменной части — сверх задания. Без него "
                   "`o_k` меряет размер постоянно активного множества, а не "
                   "устойчивость режима, ради которой §6.1 и вводится.")
    out.append("")
    out.append("`L_persist` — наименьшее k, при котором медианное перекрытие падает "
               "ниже 0.5; `None` означает, что до k = 64 этого не происходит.")
    return "\n".join(out) + "\n"


def sec_entropy(res, tag):
    cv = res["H_layer_cv"]
    t1 = res["top1_chain"]
    n = len(cv["H_cond"])
    return (
        f"**{tag}**\n\n"
        f"| оценка | сумма по слоям | на слой | примечание |\n|---|---|---|---|\n"
        f"| подстановочная по сигнатурам множеств (как в §6.2) | "
        f"{sum(res['H_layer_plugin']):.2f} бит | "
        f"{sum(res['H_layer_plugin'])/n:.3f} | схлопывается к 0, когда сигнатуры "
        f"уникальны — см. раздел об ограничениях |\n"
        f"| держательная по сигнатурам множеств | {sum(cv['H_cond']):.2f} бит | "
        f"{sum(cv['H_cond'])/n:.3f} | доля невиданных символов "
        f"{np.mean(cv['novel_rate']):.3f} |\n"
        f"| держательная по доминирующей единице (алфавит {t1['alphabet']}) | "
        f"{t1['sum_H_cond']:.2f} бит | {t1['sum_H_cond']/n:.3f} | "
        f"устойчива; отчётная |\n\n"
        f"Сколько информации предыдущий слой несёт о следующем: "
        f"**{t1['sum_info_gain']:.2f} бит** суммарно по {n} границам "
        f"(0 означало бы, что слои маршрутизируются независимо).\n\n"
        f"N_eff = 2^{t1['sum_H_cond']:.2f} по цепочке доминирующих единиц.\n\n"
        f"Ориентир §6.2 — 0.2–0.5 бит/слой для компактного пространства, "
        f"около 1.0 бит/слой для непригодного.\n")


def sec_saturation(res, tag):
    c = np.array(res["saturation"], dtype=float)
    q = [0, len(c) // 4, len(c) // 2, len(c) - 1]
    lines = [f"**{tag}**\n", "| запросов | уникальных единиц | уникальных "
             "переходов | уникальных полных путей |", "|---|---|---|---|"]
    for i in q:
        lines.append(f"| {int(c[i,0])} | {int(c[i,1])} | {int(c[i,2])} | "
                     f"{int(c[i,3])} |")
    first, last = c[0], c[-1]
    growth = [(last[j] / max(first[j], 1)) / (last[0] / max(first[0], 1))
              for j in (1, 2, 3)]
    lines.append("")
    lines.append(f"Рост относительно линейного (1.00 — продолжает расти с каждым "
                 f"запросом, 0 — вышло на плато): единицы {growth[0]:.3f}, "
                 f"переходы {growth[1]:.3f}, полные пути {growth[2]:.3f}.")
    return "\n".join(lines) + "\n"


def sec_continuation(res, tag):
    a, b = res["top1_continuation"], res["continuation"]
    n_sig = max(res["signature_counts"])
    out = [f"**{tag}** — медианная доля массы, которую держат старшие ветви узла\n",
           "| определение узла | p1 | p1+p2 | p1+p2+p3 | узлов |",
           "|---|---|---|---|---|",
           f"| доминирующая единица | {a['p1']:.3f} | {a['p12']:.3f} | "
           f"{a['p123']:.3f} | {a['nodes']} |",
           f"| полное активное множество | {b['p1']:.3f} | {b['p12']:.3f} | "
           f"{b['p123']:.3f} | {b['nodes']} |", ""]
    if b["nodes"] < 0.05 * n_sig:
        out.append(f"**Вторую строку читать нельзя.** Узел учитывается только "
                   f"при 5 и более посещениях, а различных сигнатур на слое до "
                   f"{n_sig}; условию удовлетворяют лишь {b['nodes']} узлов, то "
                   f"есть вырожденное подмножество повторяющихся состояний. "
                   f"Отчётная — первая строка.")
        out.append("")
    out.append("Прямо отвечает, сколько ветвей держать в теневом буфере (§6.4).")
    return "\n".join(out) + "\n"


def sec_gap(res, tag):
    g = res.get("gap")
    if not g:
        return f"**{tag}** — плотная модель, роутера нет, неприменимо.\n"
    ls = sorted(int(k) for k in g)
    lines = [f"**{tag}** — `logit[ранг k] − logit[ранг k+1]`\n",
             "| слой | 5 % | 25 % | 50 % | среднее |", "|---|---|---|---|---|"]
    for l in ls:
        d = g[str(l)]
        lines.append(f"| {l} | {d['q05']:.4f} | {d['q25']:.4f} | "
                     f"{d['q50']:.4f} | {d['mean']:.4f} |")
    lo = min(g[str(l)]["q05"] for l in ls)
    lines.append("")
    lines.append(f"Наименьший 5 %-квантиль по слоям = **{lo:.4f}**. Это предельное "
                 f"смещение роутера κ, при котором выбор остаётся неизменным для "
                 f"95 % токенов на худшем слое. Если разрывы велики, смещение к "
                 f"резидентным экспертам бесполезно и опираться надо на догрузку.")
    p = OUT / f"{tag}_introspect.json"
    if p.exists():
        f = json.loads(p.read_text(encoding="utf-8"))
        floor = f.get("router_noise_floor")
        if floor:
            lines.append("")
            lines.append(
                f"**Шумовой пол роутера — {floor:.3e}** (расхождение fp32-пересчёта "
                f"с матмулем модели в `{f.get('dtype')}`, размах логитов "
                f"{f.get('logit_range', float('nan')):.3f}). Зазоры ниже этой "
                f"величины в разрядности модели неразрешимы: там выбор эксперта "
                f"определяется ошибкой округления, а не роутером. "
                + (f"Наименьший 5 %-квантиль ({lo:.4f}) **ниже пола** — на этих "
                   f"слоях выбор эксперта у 5 % токенов определяется округлением, "
                   f"и смещать роутер бессмысленно."
                   if lo < floor else
                   f"Наименьший 5 %-квантиль ({lo:.4f}) — это всего "
                   f"{lo/floor:.1f}× от пола, то есть **величина того же "
                   f"порядка**. Запаса под смещение роутера нет: κ пришлось бы "
                   f"выбирать из диапазона, сравнимого с ошибкой округления."
                   if lo < 3 * floor else
                   f"Наименьший 5 %-квантиль ({lo:.4f}) выше пола в "
                   f"{lo/floor:.1f} раза, то есть κ ограничено структурой, а не "
                   f"разрядностью."))
    return "\n".join(lines) + "\n"


def sec_concentration(tag):
    """Table from dense_probe.py: how much of a layer carries how much mass."""
    p = OUT / f"{tag}_concentration.npz"
    if not p.exists():
        return ""
    z = np.load(p)
    fr, ch, bd = z["fracs"], z["chan"], z["band"]
    head = " | ".join(f"{int(f*100)} %" for f in fr)
    lines = [f"**{tag}** — какая доля слоя нужна, чтобы покрыть заданную долю "
             f"массы активаций (среднее по слоям)\n",
             f"| гранулярность | {head} |", f"|---|{'---|' * len(fr)}",
             "| каналы (s = 1) | " + " | ".join(f"{v:.3f}" for v in ch.mean(0)) + " |",
             "| полосы (s = 48) | " + " | ".join(f"{v:.3f}" for v in bd.mean(0)) + " |",
             "",
             f"По слоям при пороге 90 %, который использует §3.2: каналы от "
             f"{ch[:,2].min():.3f} (слой {int(ch[:,2].argmin())}) до "
             f"{ch[:,2].max():.3f} (слой {int(ch[:,2].argmax())}); полосы от "
             f"{bd[:,2].min():.3f} до {bd[:,2].max():.3f}.",
             "",
             "Агрегация каналов в полосы делает активное множество *менее* "
             "концентрированным, а не более: полоса суммирует 48 каналов, "
             "разбросанных по слою произвольно, поэтому суммы выравниваются. "
             "Это арифметика, а не свойство конкретной модели, и это цена, "
             "которую §3.1 платит за устойчивость к квантованию."]
    return "\n".join(lines) + "\n"


def sec_dense_extra(res, tag):
    cv = res.get("channel_vs_band")
    if not cv:
        return ""
    return (f"**{tag}** — сколько слоя реально отбирает критерий 90 % массы\n\n"
            f"| гранулярность | активно на токен | всего | доля |\n"
            f"|---|---|---|---|\n"
            f"| полосы (s = {res['trace']['band_s']}) | "
            f"{cv['mean_active_bands']:.1f} | {res['trace']['n_bands']} | "
            f"{cv['band_frac']:.3f} |\n"
            f"| каналы (s = 1) | {cv['mean_active_channels']:.1f} | "
            f"{res['trace']['d_ffn']} | {cv['channel_frac']:.3f} |\n")


def write_report(tags):
    """Assemble REPORT.md with every §9 section filled from the analyses."""
    loaded = [(t, load(t)) for t in tags]
    loaded = [(t, r) for t, r in loaded if r]
    if not loaded:
        return
    moe = [(t, r) for t, r in loaded if r["trace"]["kind"] == "moe"]
    head_tag, head_res = (moe or loaded)[0]
    q5 = head_res["q_table"]["5"]["EPOCH"]["q_steady"]
    ceil5 = head_res["q_table"]["5"]["EPOCH"]["q_ceiling"]
    iso = head_res.get("iso_ratio_point", {})

    L = []
    L.append("# Зонд компактности пространства вычислительных траекторий\n")
    L.append("```")
    L.append(f"q при M = 5 %, политика EPOCH, модель {head_tag} "
             f"({head_res['trace']['repo']}), установившийся режим = {q5:.4f}")
    L.append("```\n")
    if ceil5 < 0.95:
        L.append(f"> Читать вместе с потолком: при M = 5 % кэш этой модели держит "
                 f"{head_res['q_table']['5']['EPOCH']['capacity_units']} единиц, "
                 f"а одному токену нужно "
                 f"{head_res['trace']['mean_active_per_token']}, поэтому q "
                 f"**не может превысить {ceil5:.3f}** независимо от того, "
                 f"насколько структурен роутинг. См. раздел 3.\n")
    if iso and iso.get("clamped"):
        L.append(f"Архитектурно сопоставимая точка (то же отношение кэш/активации, "
                 f"что у целевой GPT-OSS-120B) для этой модели **недостижима**: "
                 f"понадобился бы бюджет M = {iso['wanted_M_percent']} %, то есть "
                 f"больше, чем весит вся модель. Единица учёта слишком крупная, "
                 f"чтобы быть разрежённой.\n")
    elif iso:
        L.append(f"При архитектурно сопоставимом бюджете (то же отношение "
                 f"кэш/активации, что у целевой GPT-OSS-120B, "
                 f"M = {iso.get('M_percent')} %): "
                 f"q = **{iso.get('q_steady', float('nan')):.4f}**.\n")

    L.append("\n## 1. q(M, политика)\n")
    for t, r in loaded:
        L.append(md_q_table(r, t))
    L.append("\n## 2. Графики\n")
    for t, _ in loaded:
        L.append(f"- `out/figs/{t}_q_vs_M.png` — q от M, логарифмическая ось, "
                 f"все политики")
        L.append(f"- `out/figs/{t}_layers.png` — условная энтропия и доля "
                 f"попаданий по слоям")
        L.append(f"- `out/figs/{t}_persistence.png` — перекрытие o_k")
        L.append(f"- `out/figs/{t}_saturation.png` — три кривые насыщения")
    L.append("")

    L.append("\n## 3. Вердикт по таблице §0\n")
    L.append("| модель | q при M=5 %, EPOCH | потолок при M=5 % | tok/s по модели "
             "§0 | вердикт |\n|---|---|---|---|---|")
    for t, r in loaded:
        row = r["q_table"]["5"]["EPOCH"]
        L.append(f"| {t} | {row['q_steady']:.3f} | {row['q_ceiling']:.3f} | "
                 f"{tok_s(row['q_steady']):.1f} | {verdict(row['q_steady'])} |")
    L.append("")
    L.append("При бюджете равного отношения кэш/активации:\n")
    L.append("| модель | M, % | q | tok/s | вердикт |\n|---|---|---|---|---|")
    for t, r in loaded:
        i = r.get("iso_ratio_point")
        if not i:
            continue
        if i.get("clamped"):
            # the budget hit the whole model, so q = 1 trivially; reporting a
            # verdict here would say "works as intended" about caching 100 %
            L.append(f"| {t} | недостижимо (нужно {i['wanted_M_percent']} %) | — "
                     f"| — | отношение недостижимо: рабочее множество токена "
                     f"больше, чем кэш такого размера |")
        else:
            L.append(f"| {t} | {i['M_percent']} | {i['q_steady']:.3f} | "
                     f"{tok_s(i['q_steady']):.1f} | {verdict(i['q_steady'])} |")
    L.append("")
    L.append("Пересчёт в tok/s ведётся по формуле §0 для целевой конфигурации "
             "(144 активации на токен, 6.3 мс резидентное чтение, 1.10 мс промах "
             "по PCIe) и переносится на малую модель только как условная шкала.")
    L.append("")

    L.append("\n## 4. Churn при M = 5 %, EPOCH\n")
    L.append("| модель | churn | серия промахов, медиана | p90 | максимум |\n"
             "|---|---|---|---|---|")
    for t, r in loaded:
        e = r["q_table"]["5"]["EPOCH"]
        L.append(f"| {t} | {e['churn']:.3f} | {e['miss_run_p50']:.1f} | "
                 f"{e['miss_run_p90']:.1f} | {e['miss_run_max']} |")
    L.append("")
    L.append("При `churn → 1` кэш перестраивается целиком каждую эпоху, и по PCIe "
             "это не пролезет независимо от q.")
    L.append("")
    for t, r in loaded:
        e = r["q_table"]["5"]["EPOCH"]
        cold = EPOCH_LEN * r["trace"]["mean_active_per_token"]
        if e["miss_run_max"] > 0.5 * cold:
            L.append(f"Максимальная серия промахов у {t} ({e['miss_run_max']}) — "
                     f"это холодный старт: EPOCH начинает с пустым кэшем и не "
                     f"заполняет его до первой границы эпохи, то есть "
                     f"{EPOCH_LEN} токенов × "
                     f"{r['trace']['mean_active_per_token']:.0f} активаций ≈ "
                     f"{cold:.0f}. Отчётные величины — медиана и p90.")
            break
    L.append("")

    L.append("\n## 5. Длина персистентности маршрута (§6.1)\n")
    for t, r in loaded:
        L.append(sec_persistence(r, t))
    L.append("\n## 6. Условная энтропия и N_eff (§6.2)\n")
    for t, r in loaded:
        L.append(sec_entropy(r, t))
    L.append("\n## 7. Насыщение (§6.3)\n")
    for t, r in loaded:
        L.append(sec_saturation(r, t))
    L.append("\n## 8. Концентрация продолжений (§6.4)\n")
    for t, r in loaded:
        L.append(sec_continuation(r, t))
    L.append("\n## 9. Зазор у среза роутера (§7)\n")
    for t, r in loaded:
        L.append(sec_gap(r, t))

    L.append("\n## 10. Контроль квантования (§8)\n")
    qc = OUT / "quantisation_control.md"
    L.append(qc.read_text(encoding="utf-8") if qc.exists()
             else "_не выполнялся_\n")

    L.append("\n## 11. Плотный путь\n")
    for t, r in loaded:
        if r["trace"]["kind"] == "dense":
            L.append(sec_dense_extra(r, t))
            L.append(sec_concentration(t))

    L.append("\n## Что было реально прогнано\n")
    L.append("| тег | модель | квантование | токенов в трассе | единиц | "
             "активно на токен |\n|---|---|---|---|---|---|")
    for t, r in loaded:
        tr = r["trace"]
        L.append(f"| {t} | `{tr['repo']}` | {tr['quant']} | {tr['tokens']} | "
                 f"{tr['units_total']} | {tr['mean_active_per_token']} |")
    L.append("")
    hon = ROOT / "HONESTY.md"
    if hon.exists():
        L.append(hon.read_text(encoding="utf-8"))
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT/'REPORT.md'}")


def main(tags):
    FIGS.mkdir(parents=True, exist_ok=True)
    md = []
    if not HAVE_MPL:
        print("(matplotlib unavailable -- writing tables only, no figures)")
    for tag in tags:
        res = load(tag)
        if res is None:
            print(f"  (no analysis for {tag})")
            continue
        if HAVE_MPL:
            fig_q_vs_m(res, tag)
            fig_layers(res, tag)
            fig_persist(res, tag)
            fig_saturation(res, tag)
            fig_gap(res, tag)
        md.append(md_q_table(res, tag))
        row = res["q_table"]["5"]["EPOCH"]
        q5, ceil = row["q_steady"], row["q_ceiling"]
        print(f"{tag}: q(M=5%, EPOCH, установившийся) = {q5:.4f}  "
              f"-> {tok_s(q5):.1f} tok/s -> {verdict(q5)}")
        if ceil < 0.95:
            print(f"    потолок: кэш держит {row['capacity_units']} единиц, токену "
                  f"нужно {res['trace']['mean_active_per_token']}, поэтому q не может "
                  f"превысить {ceil:.3f} при любом роутинге")
        iso = res.get("iso_ratio_point")
        if iso and iso.get("clamped"):
            print(f"    равное отношение недостижимо: понадобилось бы "
                  f"M={iso['wanted_M_percent']}%, больше чем есть в модели")
        elif iso:
            print(f"    равное отношение M={iso['M_percent']}%: q={iso['q_steady']:.4f}"
                  f"  -> {tok_s(iso['q_steady']):.1f} tok/s -> {verdict(iso['q_steady'])}")
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {OUT/'tables.md'}" + (f" and figures in {FIGS}" if HAVE_MPL else ""))
    write_report(tags)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "A2", "B", "B2"])
