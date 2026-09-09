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
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEPS = 50
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"raw": "#1f4e79", "proxy": "#8c8c8c", "R": "#0e7c6b", "dcoh": "#1f4e79", "dclap": "#b8602a"}
NAMES = {"sao": "SAO (unguided)", "early": "Early (0--14)", "mid": "Mid (18--32)", "late": "Late (35--49)",
         "uniform": "Uniform", "mlsp": "MLSP-fixed", "rapg_cal_f1": "RAPG-cal ($F_1$)",
         "rapg_cal_dc": "RAPG-cal ($\\Delta C$)", "rapg_on": "RAPG-online", "topk_dc_const": "Top-$K$($\\Delta C$), const.\\ $\\lambda$"}


def fig_trajectory(expA: Path, expC: Path, schedules: dict | None, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.6), gridspec_kw=dict(hspace=.5))
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 3.3), sharex=True)
    s = pd.read_csv(expA / "summary.csv")
    raw, prox = s[s.variant == "raw"].sort_values("row"), s[s.variant == "gauss_proxy"].sort_values("row")
    a.plot(raw.progress, raw.f1, color=C["raw"], lw=1.4, label="probe $F_1$ (real trajectory)")
    a.fill_between(raw.progress, raw.f1 - 1.96 * raw.f1_sem, raw.f1 + 1.96 * raw.f1_sem, color=C["raw"], alpha=.15, lw=0)
    a.plot(prox.progress, prox.f1, color=C["proxy"], lw=1.2, ls="--", label="$F_1$, Gaussian proxy")
    a2 = a.twinx(); a2.spines.right.set_visible(True)
    a2.plot(raw.progress, raw.R_entropy, color=C["R"], lw=1.4, label="reliability $R_t$")
    a2.set_ylabel("$R_t$", color=C["R"]); a2.tick_params(axis="y", colors=C["R"])
    a.set_ylabel("micro-$F_1$"); a.set_title("(a) decodability along real SAO trajectories", loc="left")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False)
    bp = pd.read_csv(expC / "by_position.csv").sort_values("position")
    b.errorbar(bp.progress, bp.d_coherence_mlsp, yerr=[bp.d_coherence_mlsp - bp.d_coherence_mlsp_lo, bp.d_coherence_mlsp_hi - bp.d_coherence_mlsp],
               color=C["dcoh"], marker="o", ms=3, lw=1.2, capsize=2, label="$\\Delta$ coherence")
    b.axhline(0, color="k", lw=.5)
    b2 = b.twinx(); b2.spines.right.set_visible(True)
    b2.errorbar(bp.progress, bp.d_clap, yerr=[bp.d_clap - bp.d_clap_lo, bp.d_clap_hi - bp.d_clap], color=C["dclap"],
                marker="s", ms=3, lw=1.2, capsize=2, label="$\\Delta$ CLAP")
    b2.set_ylabel("$\\Delta$ CLAP", color=C["dclap"]); b2.tick_params(axis="y", colors=C["dclap"])
    b.set_ylabel("$\\Delta$ coherence"); b.set_xlabel("denoising progress $s=t/N$")
    b.set_title("(b) steering sensitivity of a 3-step burst at $s$", loc="left")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    if schedules:   # strip of guided steps
        y0 = b.get_ylim()[0]
        for i, (k, lab) in enumerate((("early", "Early"), ("mlsp", "MLSP"), ("rapg_cal_dc", "RAPG"))):
            st = schedules.get(k, {}).get("steps")
            if st:
                yy = y0 - (i + 1) * 0.0
                b.scatter([(t + 1) / STEPS for t in st], [yy] * len(st), marker="|", s=12, color=["#444", "#888", C["R"]][i], label=None)
    fig.tight_layout(); fig.savefig(out / "fig_trajectory.pdf"); plt.close(fig)


def fig_pareto(expB: Path, out: Path):
    tab = pd.read_csv(expB / "table.csv")
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    for _, r in tab.iterrows():
        col = C["R"] if r.method.startswith("rapg") else ("#888" if r.method == "sao" else C["raw"])
        ax.errorbar(r.clap, r.coherence_mlsp, xerr=[[r.clap - r.clap_lo], [r.clap_hi - r.clap]],
                    yerr=[[r.coherence_mlsp - r.coherence_mlsp_lo], [r.coherence_mlsp_hi - r.coherence_mlsp]],
                    fmt="o", ms=4, color=col, capsize=2, lw=.8)
        ax.annotate(NAMES.get(r.method, r.method).replace("\\ ", " "), (r.clap, r.coherence_mlsp), textcoords="offset points",
                    xytext=(4, 3), fontsize=6.5)
    ax.set_xlabel("CLAP score $\\uparrow$"); ax.set_ylabel("melodic coherence $\\uparrow$")
    fig.tight_layout(); fig.savefig(out / "fig_pareto.pdf"); plt.close(fig)


def table_and_numbers(expA: Path, expC: Path, expB: Path, out: Path, order=("sao", "early", "mid", "late", "uniform", "mlsp",
                                                                        "rapg_cal_f1", "rapg_cal_dc", "rapg_on", "topk_dc_const")):
    tab = pd.read_csv(expB / "table.csv").set_index("method")
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Method & Coh.\\ $\\uparrow$ & Chroma $\\uparrow$ & CLAP $\\uparrow$ & FAD $\\downarrow$ & \\#upd.\\\\", "\\midrule"]
    def cell(r, k, nd=3):
        return f"{r[k]:.{nd}f} {{\\scriptsize[{r[k+'_lo']:.{nd}f},{r[k+'_hi']:.{nd}f}]}}"
    for m in order:
        if m not in tab.index:
            continue
        r = tab.loc[m]
        dag = "$^\\dagger$" if m != "mlsp" and r.get("p_coherence_mlsp_vs_mlsp_holm", 1) < 0.05 else ""
        fad = f"{r['fad_clap_maestro']:.2f}" if "fad_clap_maestro" in r and pd.notna(r["fad_clap_maestro"]) else "--"
        upd = f"{r['n_updates']:.1f}" if m == "rapg_on" else f"{int(round(r['n_updates']))}"
        lines.append(f"{NAMES.get(m, m)} & {cell(r,'coherence_mlsp')}{dag} & {cell(r,'chroma_cos',2)} & {cell(r,'clap',3)} & {fad} & {upd}\\\\")
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
    (out / "numbers.tex").write_text("% auto-generated by make_figures.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items()))
    print(json.dumps(nums, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--expA", required=True); ap.add_argument("--expC", required=True); ap.add_argument("--expB")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    sch = json.load(open(Path(a.expB) / "schedules.json")) if a.expB and (Path(a.expB) / "schedules.json").exists() else None
    fig_trajectory(Path(a.expA), Path(a.expC), sch, out)
    if a.expB:
        fig_pareto(Path(a.expB), out); table_and_numbers(Path(a.expA), Path(a.expC), Path(a.expB), out)
    print("figures written to", out)
