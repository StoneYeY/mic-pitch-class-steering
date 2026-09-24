"""Paper figures + numbers.tex + table_main.tex from result csvs (runs anywhere, no GPU).

  python w2s/scripts/make_figures.py --expA results/003_expA_trajectories --expC results/004_expC_burst \
         --expB results/006_expB_test --out paper/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.patches import Rectangle
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = 50
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"raw": "#1f4e79", "proxy": "#8c8c8c", "R": "#0e7c6b", "dcoh": "#1f4e79", "dclap": "#b8602a"}
NAMES = {"sao": "SAO (unguided)", "early": "Early (0--14)", "mid": "Mid (18--32)", "late": "Late (35--49)",
         "uniform": "Uniform", "mlsp": "MLSP-fixed (20:2:48)", "rapg_cal_f1": "Top-$K$($F_1$) + $R$-scaling",
         "rapg_cal_dc": "SCPG + $R$-scaling", "rapg_on": "$R$-threshold (online)", "topk_dc_const": "SCPG: Top-$K$($\\Delta C$)"}


def fig_trajectory(expA: Path, expC: Path, schedules: dict | None, out: Path, figsize=(3.45, 3.35), *, name="fig_trajectory.pdf",
                   model="SAO", noise="rms", noise_ticks=(0.99, 0.95, 0.75, 0.5), show_proxy=True, dclap_ylim=(-0.03, 0.03),
                   margins=(0.16, 0.80, 0.81, 0.14)):
    """Figs. 1 and 3: (a) decodability / reliability along real trajectories, (b) burst steering sensitivity,
    (c) a labelled strip of the guided steps of three schedules.  noise="rms": noise level from the stored latent rms
    (SAO, EDM latents); noise="sigma": the sampler's own flow time from the per-step sigma column (SA3)."""
    k = 3.05 / figsize[1]   # keep the gaps and the strip legible when the figure is made shorter
    fig, (a, b, c) = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.0, 1.0, 0.40 * k], hspace=1.05 * k))
    s = pd.read_csv(expA / "summary.csv")
    raw, prox = s[s.variant == "raw"].sort_values("row"), s[s.variant == "gauss_proxy"].sort_values("row")
    a.plot(raw.progress, raw.f1, color=C["raw"], lw=1.4, label="Probe $F_1$")
    a.fill_between(raw.progress, raw.f1 - 1.96 * raw.f1_sem, raw.f1 + 1.96 * raw.f1_sem, color=C["raw"], alpha=.15, lw=0)
    if show_proxy:
        a.plot(prox.progress, prox.f1, color=C["proxy"], lw=1.2, ls="--", label="Proxy")
    a2 = a.twinx(); a2.spines.right.set_visible(True)
    a2.plot(raw.progress, raw.R_entropy, color=C["R"], lw=1.4, ls="-.", label="$R_t$")
    a2.set_ylabel("$R_t$", color=C["R"]); a2.tick_params(axis="y", colors=C["R"])
    a.set_ylabel("micro-$F_1$"); a.text(-0.16, 1.70, f"(a) {model}: decodability", transform=a.transAxes, fontsize=8, ha="left", va="bottom")
    a.set_ylim(bottom=0)
    # noise level of the stored latents, as the flow-time equivalent tau = sigma/(1+sigma) with sigma = noise/signal rms
    try:
        if noise == "sigma":
            tflow = raw.sigma.values
        else:
            rms0 = float(np.mean([x["rms0"] for x in json.load(open(expA / "corr.json"))["runs"]]))
            sig = np.sqrt(np.maximum(raw.rms.values ** 2 - rms0 ** 2, 0)) / rms0; tflow = sig / (1 + sig)
        top = a.secondary_xaxis("top")
        ticks_t = list(noise_ticks)
        top.set_xticks([float(raw.progress.values[int(np.argmin(np.abs(tflow - t)))]) for t in ticks_t])
        top.set_xticklabels([f"{t:.2f}" for t in ticks_t], fontsize=6); top.tick_params(length=2, pad=1)
        top.set_xlabel("noise level $\\tau$", fontsize=6.5, labelpad=1)
    except Exception as e:  # noqa: BLE001
        print("no noise axis:", e)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.38), ncol=3, frameon=False, fontsize=6.5, handlelength=1.8, columnspacing=1.2, borderaxespad=0)
    bp = pd.read_csv(expC / "by_position.csv").sort_values("position")
    b.errorbar(bp.progress, bp.d_coherence_mlsp, yerr=[bp.d_coherence_mlsp - bp.d_coherence_mlsp_lo, bp.d_coherence_mlsp_hi - bp.d_coherence_mlsp],
               color=C["dcoh"], marker="o", ms=3, lw=1.2, capsize=2, label="$\\Delta$ coherence")
    b.axhline(0, color="k", lw=.5)
    has_clap = np.isfinite(bp.d_clap).any()
    if has_clap:
        b2 = b.twinx(); b2.spines.right.set_visible(True)
        b2.errorbar(bp.progress, bp.d_clap, yerr=[bp.d_clap - bp.d_clap_lo, bp.d_clap_hi - bp.d_clap], color=C["dclap"],
                    marker="s", ms=3, lw=1.2, capsize=2, label="$\\Delta$ CLAP")
        b2.set_ylabel("$\\Delta$ CLAP", color=C["dclap"]); b2.tick_params(axis="y", colors=C["dclap"])
        b2.set_ylim(*dclap_ylim)
    b.set_ylim(top=float(bp.d_coherence_mlsp_hi.max()) * 1.4)
    b.set_ylabel("$\\Delta$ coherence")
    b.text(-0.16, 1.30, "(b) Three-step burst sensitivity", transform=b.transAxes, fontsize=8, ha="left", va="bottom")
    if has_clap:
        h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
        b.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False, fontsize=6.5, handlelength=1.8, columnspacing=1.5, borderaxespad=0)
    else:
        b.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False)
    # (c) labelled strip of guided steps
    rows = (("early", "Early", "#666666"), ("mlsp", "MLSP", "#9a9a9a"), ("rapg_cal_dc", "SCPG", C["R"]))
    c.set_ylim(-0.6, len(rows) - 0.4); c.set_yticks(range(len(rows))); c.set_yticklabels([r[1] for r in rows], fontsize=6.5)
    c.invert_yaxis(); c.tick_params(axis="y", length=0); c.tick_params(axis="x", labelsize=6.5)
    for sp in ("top", "right", "left"):
        c.spines[sp].set_visible(False)
    for i, (k, lab, col) in enumerate(rows):
        st = (schedules or {}).get(k, {}).get("steps") or []
        for st_i in st:
            c.add_patch(Rectangle(((st_i + 0.5) / STEPS, i - 0.32), 1.0 / STEPS, 0.64, color=col, lw=0))
    c.text(-0.16, 1.15, "(c) Guided steps", transform=c.transAxes, fontsize=8, ha="left", va="bottom")
    c.set_xlim(0, 1.0); c.set_xlabel("denoising progress $s$")
    fig.subplots_adjust(left=margins[0], right=margins[1], top=margins[2], bottom=margins[3])
    # guard against clipped axis titles: every label must lie inside the figure canvas
    fig.canvas.draw()
    W, H = fig.get_size_inches() * fig.dpi
    for ax in fig.get_axes():
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            bb = lab.get_window_extent()
            if lab.get_text() and (bb.x0 < 0 or bb.y0 < 0 or bb.x1 > W or bb.y1 > H):
                print(f"WARNING clipped label {lab.get_text()!r} in {name}: {bb}")
    fig.savefig(out / name); plt.close(fig)


def fig_pareto(expB: Path, out: Path):
    tab = pd.read_csv(expB / "table.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    SHORT = {"sao": "SAO", "early": "Early", "mid": "Mid", "late": "Late", "uniform": "Uniform",
             "mlsp": "MLSP-fixed", "rapg_cal_f1": "Top-$K$($F_1$)+R", "rapg_cal_dc": "SCPG+R",
             "rapg_on": "R-threshold", "topk_dc_const": "SCPG"}
    # manual label offsets (points) to avoid overlap in the upper cluster
    OFF = {"sao": (6, -3), "early": (6, -2), "mid": (-4, 7), "late": (6, -8), "uniform": (6, -3),
           "mlsp": (-52, 6), "rapg_cal_f1": (-70, -10), "rapg_cal_dc": (4, 6), "rapg_on": (6, -3),
           "topk_dc_const": (6, 4)}
    for m in tab.index:
        r = tab.loc[m]
        col = C["R"] if m.startswith("rapg") or m == "topk_dc_const" else ("#888" if m == "sao" else C["raw"])
        ax.errorbar(r.clap, r.coherence_mlsp, xerr=[[r.clap - r.clap_lo], [r.clap_hi - r.clap]],
                    yerr=[[r.coherence_mlsp - r.coherence_mlsp_lo], [r.coherence_mlsp_hi - r.coherence_mlsp]],
                    fmt="o", ms=4, color=col, capsize=2, lw=.8, zorder=3)
        ax.annotate(SHORT.get(m, m), (r.clap, r.coherence_mlsp), textcoords="offset points",
                    xytext=OFF.get(m, (5, 3)), fontsize=6.2, color=col)
    ax.set_xlabel("CLAP score $\\uparrow$"); ax.set_ylabel("melodic coherence $\\uparrow$")
    ax.margins(0.16)
    fig.tight_layout(); fig.savefig(out / "fig_pareto.pdf"); plt.close(fig)


def fig_budget(expD: Path, out: Path, expB: Path | None = None):
    """Budget sweep: melodic coherence (and CLAP) vs number of updates K for Uniform vs Top-K(dC)-const.
    Paired subset of the test set (10 prompts x 5 melodies x seed 0). Writes fig_budget.pdf + numbers_budget.tex."""
    df = pd.read_csv(expD / "per_run.csv")
    rng = np.random.default_rng(0)
    LAB = {"uniform": "Uniform", "topk_dc": "SCPG", "late": "Late", "rapg_cal_dc": "SCPG + $R$-scaling"}
    STY = {"uniform": ("o", "-"), "topk_dc": ("s", "--"), "late": ("^", ":"), "rapg_cal_dc": ("D", "-.")}
    COL = {"uniform": C["raw"], "topk_dc": C["R"], "late": "#8c8c8c", "rapg_cal_dc": C["R"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.55))
    recs = {}
    for m in [m for m in ["uniform", "topk_dc", "late", "rapg_cal_dc"] if m in set(df.method)]:
        g = df[df.method == m].groupby("K")
        Ks = sorted(g.groups)
        for ax, key in ((a, "coherence_mlsp"), (b, "clap")):
            mu, lo, hi = [], [], []
            for K in Ks:
                x = g.get_group(K)[key].dropna().values
                if not len(x):
                    mu.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
                bs = np.array([rng.choice(x, len(x)).mean() for _ in range(2000)])
                mu.append(x.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
                rec = recs.setdefault((m, K), dict(method=m, K=K, n=len(x)))
                short = "coh" if key == "coherence_mlsp" else "clap"
                rec.update({short: x.mean(), short + "_lo": lo[-1], short + "_hi": hi[-1]})
            ax.fill_between(Ks, lo, hi, color=COL[m], alpha=.15, lw=0)
            ax.plot(Ks, mu, marker=STY[m][0], ls=STY[m][1], ms=3, lw=1, color=COL[m], label=LAB[m])
    if expB is not None and (expB / "per_run.csv").exists():   # unguided reference on the same trials
        ref = pd.read_csv(expB / "per_run.csv"); ref = ref[(ref.method == "sao") & ref.trial.isin(set(df.trial))]
        for ax, key in ((a, "coherence_mlsp"), (b, "clap")):
            ax.axhline(ref[key].mean(), color="#888", lw=.8, ls=":", zorder=1)
        a.text(max(df.K), ref.coherence_mlsp.mean(), " unguided", fontsize=6.2, color="#666", va="bottom", ha="right")
    a.set_ylabel("coherence $\\uparrow$"); b.set_ylabel("CLAP $\\uparrow$")
    for ax in (a, b):
        ax.set_xlabel("updates $K$"); ax.set_xticks(sorted(set(df.K)))
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=6.8, handlelength=2.0, columnspacing=2.0, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.88)); fig.savefig(out / "fig_budget.pdf"); plt.close(fig)
    tab = pd.DataFrame(list(recs.values())); tab.to_csv(out / "budget_summary.csv", index=False)
    # numbers for the text: the K at which Top-K(dC) matches the best uniform, and gain at K=15 / K=5
    nums = {}
    if {"uniform", "topk_dc"} <= set(tab.method):
        u = tab[tab.method == "uniform"].set_index("K").coh; t = tab[tab.method == "topk_dc"].set_index("K").coh
        nums["budgetKs"] = ",".join(str(k) for k in u.index)
        roman = {5: "v", 10: "x", 15: "xv", 20: "xx", 25: "xxv", 30: "xxx", 50: "l"}   # macro names cannot contain digits
        for K in u.index:
            nums[f"cohUniK{roman.get(int(K), 'K' + str(K))}"] = f"{u[K]:.2f}"; nums[f"cohTopK{roman.get(int(K), 'K' + str(K))}"] = f"{t[K]:.2f}"
        umax = u.max(); kmatch = next((K for K in t.index if t[K] >= umax), None)
        nums["budgetUniMax"] = f"{umax:.2f}"; nums["budgetUniMaxK"] = str(int(u.idxmax()))
        nums["budgetTopMatchK"] = str(int(kmatch)) if kmatch is not None else "--"
    lines = ["% auto-generated by make_figures.py (Exp D)"] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()]
    (out / "numbers_budget.tex").write_text("\n".join(lines) + "\n")
    print(tab.round(3).to_string()); print(nums)


def fig_sa3(expA: Path, expC: Path, out: Path, expB: Path | None = None, sigma_ticks=(0.99, 0.87, 0.5, 0.14)):
    """Transfer check on Stable Audio 3 (single column, two panels): decodability F1(t) / R_t along real
    trajectories and burst steering sensitivity dC(t).  A top axis shows the sampler's noise level t at
    selected positions (the SA3 schedule is log-SNR-shifted, so step index and noise level diverge).
    Writes fig_sa3.pdf and numbers_sa3.tex."""
    s = pd.read_csv(expA / "summary.csv"); raw = s[s.variant == "raw"].sort_values("row")
    bp = pd.read_csv(expC / "by_position.csv").sort_values("position")
    sched3 = json.load(open(expB / "schedules.json")) if expB is not None and (expB / "schedules.json").exists() else None
    fig_trajectory(expA, expC, sched3, out, figsize=(3.45, 3.1), name="fig_sa3.pdf", model="SA3", noise="sigma",
                   noise_ticks=sigma_ticks, show_proxy=False, dclap_ylim=(-0.03, 0.06), margins=(0.16, 0.80, 0.79, 0.15))
    sig = raw.set_index("row").sigma
    f1 = raw.f1.values; sat = int(np.argmax(f1 >= 0.95 * f1.max()))
    corr = json.load(open(expA / "corr.json"))["corr"]
    peak = int(bp.d_coherence_mlsp.values.argmax())
    nums = {"saThreePeakF": f"{100*(sat+1)/STEPS:.0f}\\%", "saThreePeakDC": f"{100*bp.progress.values[peak]:.0f}\\%",
            "saThreeCorrRF": f"{corr['raw']['corr(R_entropy,f1)']:.2f}", "saThreeDCpeak": f"{bp.d_coherence_mlsp.values[peak]:.3f}",
            "saThreeDCearly": f"{bp.d_coherence_mlsp.values[:3].mean():.3f}", "saThreeDClate": f"{bp.d_coherence_mlsp.values[-2:].mean():.3f}",
            "saThreeFmax": f"{f1.max():.2f}", "saThreeTpeak": f"{float(sig.values[int(bp.position.values[peak])]):.2f}"}
    if expB is not None and (expB / "table.csv").exists():
        tb = pd.read_csv(expB / "table.csv").set_index("method")
        names = {"sao": "SAO", "early": "Early", "mid": "Mid", "late": "Late", "uniform": "Uni", "mlsp": "MLSP",
                 "rapg_cal_dc": "RAPG", "topk_dc_const": "TopK"}
        for m, n in names.items():
            if m in tb.index:
                nums[f"saThreeCoh{n}"] = f"{tb.loc[m, 'coherence_mlsp']:.2f}"
                if "clap" in tb.columns and pd.notna(tb.loc[m, "clap"]):
                    nums[f"saThreeClap{n}"] = f"{tb.loc[m, 'clap']:.3f}"
        if "rapg_cal_dc" in tb.index and "p_coherence_mlsp_vs_mlsp" in tb.columns:
            pv = tb.loc["rapg_cal_dc", "p_coherence_mlsp_vs_mlsp"]
            nums["saThreePRAPGvsMLSP"] = f"{pv:.1e}".replace("e-0", "e-") if pv >= 1e-4 else "10^{-4}"
        if "topk_dc_const" in tb.index and "p_coherence_mlsp_vs_mlsp_holm" in tb.columns:
            ph = float(tb.loc["topk_dc_const", "p_coherence_mlsp_vs_mlsp_holm"])
            nums["saThreePTopKvsMLSPholm"] = f"{ph:.3f}" if ph >= 0.001 else f"{ph:.1e}"
        st = json.load(open(expB / "schedules.json")).get("rapg_cal_dc", {}).get("steps")
        if st:
            nums["saThreeRAPGsteps"] = f"{min(st)}--{max(st)}"
    (out / "numbers_sa3.tex").write_text("% auto-generated by make_figures.py (SA3 transfer)\n" +
                                        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()) + "\n")
    print(nums)


def table_and_numbers(expA: Path, expC: Path, expB: Path, out: Path, order=("sao", "early", "mid", "late", "uniform", "mlsp",
                                                                        "rapg_cal_f1", "rapg_on", "topk_dc_const", "rapg_cal_dc")):
    tab = pd.read_csv(expB / "table.csv").set_index("method")
    lines = ["\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{lccccc}", "\\toprule",
             "Method & Coh.\\ $\\uparrow$ & Chroma $\\uparrow$ & CLAP $\\uparrow$ & FAD $\\downarrow$ & \\#u.\\\\", "\\midrule"]
    def cell(r, k, nd=3):  # value with a small CI underneath-in-brackets
        return f"{r[k]:.{nd}f}\\,{{\\scriptsize[{r[k+'_lo']:.{nd}f},{r[k+'_hi']:.{nd}f}]}}"
    for m in order:
        if m not in tab.index:
            continue
        r = tab.loc[m]
        dag = ""
        if m != "mlsp" and r.get("p_coherence_mlsp_vs_mlsp_holm", 1) < 0.05:
            dag = "$^{+}$" if r["coherence_mlsp"] > tab.loc["mlsp", "coherence_mlsp"] else "$^{-}$"
        fad = f"{r['fad_clap_maestro']:.2f}" if "fad_clap_maestro" in r and pd.notna(r["fad_clap_maestro"]) else "--"
        upd = f"{r['n_updates']:.1f}" if m == "rapg_on" else f"{int(round(r['n_updates']))}"
        lines.append(f"{NAMES.get(m, m)} & {r['coherence_mlsp']:.3f}{dag} & {r['chroma_cos']:.2f} & {r['clap']:.3f} & {fad} & {upd}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")
    corr = json.load(open(expA / "corr.json"))["corr"]["raw"]
    sC = json.load(open(expC / "summary.json"))
    sA = pd.read_csv(expA / "summary.csv"); rawA = sA[sA.variant == "raw"].sort_values("row")
    f1 = rawA.f1.values; sat = int(np.argmax(f1 >= 0.95 * f1.max()))
    nums = {
        "cohMLSP": f"{tab.loc['mlsp','coherence_mlsp']:.3f}" if "mlsp" in tab.index else "0.xx",
        "cohRAPG": f"{tab.loc['rapg_cal_dc','coherence_mlsp']:.3f}" if "rapg_cal_dc" in tab.index else "0.xx",
        "cohSAO": f"{tab.loc['sao','coherence_mlsp']:.3f}" if "sao" in tab.index else "0.xx",
        "corrRF": f"{corr['corr(R_entropy,f1)']:.2f}",
        "peakDC": f"{100*(sC['argmax_dcoh_pos']+1)/STEPS:.0f}\\%",
        "peakF": f"{100*(sat+1)/STEPS:.0f}\\%",
    }
    # ---- review additions: paired tests among the top schedules, realised strength, placement share, fixed-t rho ----
    from scipy.stats import spearmanr, wilcoxon
    prB = pd.read_csv(expB / "per_run.csv"); piv = prB.pivot(index="trial", columns="method", values="coherence_mlsp")
    def paired(a, b):
        d = (piv[a] - piv[b]).dropna(); nz = d[d != 0]
        return d.mean(), wilcoxon(d).pvalue, int((nz > 0).sum()), int(len(nz))
    def ptex(p):
        return "p<10^{-6}" if p < 1e-6 else ("p<10^{-5}" if p < 1e-5 else ("p<10^{-4}" if p < 1e-4 else ("p<10^{-3}" if p < 1e-3 else (f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"))))
    if {"rapg_cal_dc", "mid", "topk_dc_const", "mlsp"} <= set(piv.columns):
        d, p, w, n = paired("rapg_cal_dc", "mid"); nums.update(dRvsMid=f"{d:+.3f}", pRvsMid=ptex(p), winsRvsMid=f"{w} of {n}")
        d, p, w, n = paired("rapg_cal_dc", "topk_dc_const"); nums.update(dRvsTopK=f"{d:+.3f}", pRvsTopK=ptex(p), winsRvsTopK=f"{w} of {n}")
        d, p, w, n = paired("topk_dc_const", "mid"); nums.update(dTopKvsMid=f"{d:+.3f}", pTopKvsMid=ptex(p), winsTopKvsMid=f"{w} of {n}")
        d, p, w, n = paired("topk_dc_const", "mlsp"); nums.update(pTopKvsMLSP=ptex(p))
        share = (tab.loc["topk_dc_const", "coherence_mlsp"] - tab.loc["mlsp", "coherence_mlsp"]) / (tab.loc["rapg_cal_dc", "coherence_mlsp"] - tab.loc["mlsp", "coherence_mlsp"])
        nums["placementShare"] = f"{100*share:.0f}\\%"
        nums["cohTopK"] = f"{tab.loc['topk_dc_const','coherence_mlsp']:.3f}"; nums["cohMid"] = f"{tab.loc['mid','coherence_mlsp']:.3f}"
        nums["lamRealised"] = f"{prB[prB.method == 'rapg_cal_dc'].mean_lam.mean():.3f}"
        nums["cohGainAbs"] = f"{tab.loc['rapg_cal_dc','coherence_mlsp'] - tab.loc['sao','coherence_mlsp']:.2f}"
    psA = pd.read_csv(expA / "per_step.csv"); psA = psA[psA.variant == "raw"]
    rhos = [spearmanr(g.R_entropy, g.f1).correlation for _, g in psA.groupby("row") if g.f1.std() > 0 and g.R_entropy.std() > 0]
    rhos_late = [spearmanr(g.R_entropy, g.f1).correlation for r_, g in psA.groupby("row") if r_ >= 35 and g.f1.std() > 0]
    nums["rhoFixedT"] = f"{np.nanmedian(rhos):.2f}"; nums["rhoFixedTlate"] = f"{np.nanmedian(rhos_late):.2f}"
    nums["rhoFixedTpos"] = f"{100*np.mean(np.array(rhos) > 0):.0f}\\%"
    st = json.load(open(expC / "stability.json")) if (expC / "stability.json").exists() else None
    if st:
        loo = sorted(set(int(v) for v in st["loo_prompt"].values())); nums["looRange"] = f"{loo[0]}--{loo[-1]}" if len(loo) > 1 else str(loo[0])
        nums["looRangeS"] = f"{(loo[0]+1)/STEPS:.2f}--{(loo[-1]+1)/STEPS:.2f}" if len(loo) > 1 else f"{(loo[0]+1)/STEPS:.2f}"
        nums["bootPeakShare"] = f"{100*st['boot_counts'][str(st['peak'])]/sum(st['boot_counts'].values()):.0f}\\%"
    # noise level (flow-time equivalent) of SAO latents at the window edges, from the stored rms
    rms0 = float(np.mean([x["rms0"] for x in json.load(open(expA / "corr.json"))["runs"]]))
    def tflow(row):
        sig = np.sqrt(max(rawA.rms.values[row] ** 2 - rms0 ** 2, 0)) / rms0; return sig, sig / (1 + sig)
    for name, row in (("Eighteen", 18), ("ThirtyTwo", 32), ("Peak", int(sC["argmax_dcoh_pos"])), ("Fourteen", 14)):
        sg, tf = tflow(row); nums[f"sigmaAt{name}"] = f"{sg:.0f}" if sg >= 10 else f"{sg:.1f}"; nums[f"tAt{name}"] = f"{tf:.2f}"
    (out / "numbers.tex").write_text("% auto-generated by make_figures.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expA", required=True); ap.add_argument("--expC", required=True); ap.add_argument("--expB")
    ap.add_argument("--expD"); ap.add_argument("--sa3A"); ap.add_argument("--sa3C"); ap.add_argument("--sa3B"); ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    sch = json.load(open(Path(a.expB) / "schedules.json")) if a.expB and (Path(a.expB) / "schedules.json").exists() else None
    fig_trajectory(Path(a.expA), Path(a.expC), sch, out)
    if a.expB:
        fig_pareto(Path(a.expB), out); table_and_numbers(Path(a.expA), Path(a.expC), Path(a.expB), out)
    if a.expD:
        fig_budget(Path(a.expD), out, Path(a.expB) if a.expB else None)
    if a.sa3A and a.sa3C:
        fig_sa3(Path(a.sa3A), Path(a.sa3C), out, Path(a.sa3B) if a.sa3B else None)
    print("figures written to", out)
