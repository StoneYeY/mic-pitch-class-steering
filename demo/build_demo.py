"""Build the self-contained results / audio-examples page (docs/index.html and the claude.ai artifact body).

Everything on the page comes from files in this repository:
  * clips ............ results/013_collect_demo_v2 (+ results/029_sao_windows_demo for the fixed windows),
                       transcoded to mp3 and embedded as data URIs
  * per-clip metrics . results/009_expB_full/per_run.csv  (the paper's Exp B run; nothing is re-scored here)
  * the summary table  results/009_expB_full/table.csv    (= Table 1 of the paper, test set, n = 225)
  * figures .......... paper/fig_trajectory.pdf, paper/fig_sa3.pdf (Fig. 1 and Fig. 3 of the paper)
  * SA3 block ........ the same for results/021_sa3_expB, results/027_sa3_demo, results/030_sa3_windows_demo

Usage (from the repository root, after `ffmpeg -i x.demo.wav -ac 1 -b:a 128k x.mp3` for every demo wav):

  python demo/build_demo.py --res results/013_collect_demo_v2 --mp3 DIR [--win-mp3 DIR] --expb results/009_expB_full \
      --sa3-res results/027_sa3_demo --sa3-mp3 DIR [--sa3-win-mp3 DIR] --sa3-expb results/021_sa3_expB \
      --paper paper --out-body BODY.html --out-full docs/index.html

  --out-body : page content without doctype/html/head/body (what the claude.ai Artifact tool expects)
  --out-full : the same content wrapped in a complete document (GitHub Pages / local viewing)
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from w2s import data  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--res", required=True, help="demo manifest dir (results/013_collect_demo_v2)")
ap.add_argument("--mp3", required=True, help="dir with <trial>__<cond>.mp3 and target_<melody>.mp3")
ap.add_argument("--win-mp3", default=None, help="dir with the Early/Mid/Late mp3s (from results/029_sao_windows_demo)")
ap.add_argument("--expb", required=True, help="Exp B results dir with per_run.csv, table.csv, schedules.json")
ap.add_argument("--sa3-res", default=None)
ap.add_argument("--sa3-mp3", default=None)
ap.add_argument("--sa3-win-mp3", default=None)
ap.add_argument("--sa3-expb", default=None)
ap.add_argument("--paper", default=str(HERE.parent / "paper"), help="dir with fig_trajectory.pdf / fig_sa3.pdf")
ap.add_argument("--out-body", required=True)
ap.add_argument("--out-full", default=None)
args = ap.parse_args()


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def steps_desc(steps: list[int]) -> str:
    """'19–33', '20–48, every 2nd', or an explicit list."""
    if not steps:
        return "none"
    st = sorted(int(s) for s in steps)
    if len(st) == 1:
        return str(st[0])
    d = {b - a for a, b in zip(st, st[1:])}
    if d == {1}:
        return f"{st[0]}–{st[-1]}"
    if len(d) == 1:
        k = d.pop()
        return f"{st[0]}–{st[-1]}, every {k}{'nd' if k == 2 else 'rd' if k == 3 else 'th'}"
    return ", ".join(map(str, st))


def find_mp3(dirs: list[Path], name: str) -> Path | None:
    for d in dirs:
        if d and (d / name).exists():
            return d / name
    return None


def build_items(man, per_run, mp3dirs, conds):
    items = []
    for it in man["items"]:
        tid = f"test-p{it['prompt_id']:02d}-{it['melody']}-s0"
        mel = data.MELODIES[it["melody"]]
        clips = {}
        for cond in conds:
            rows = per_run[(per_run.trial == tid) & (per_run.method == cond)]
            src = find_mp3(mp3dirs, f"{tid}__{cond}.mp3")
            if not rows.shape[0] or src is None:
                continue
            row = rows.iloc[0]
            clap = float(row.clap) if pd.notna(row.clap) else float("nan")
            clips[cond] = dict(audio=b64(src),
                               coh=round(float(row.coherence_mlsp), 3), chroma=round(float(row.chroma_cos), 3),
                               clap=None if clap != clap else round(clap, 3), n_updates=int(row.n_updates))
        tgt = find_mp3(mp3dirs, f"target_{it['melody']}.mp3")
        if clips and tgt is not None:
            items.append(dict(id=tid, prompt_id=it["prompt_id"], prompt=it["prompt"], melody=it["melody"],
                              notes=list(mel.notes), tempo=mel.tempo, duration=mel.duration_s,
                              target=b64(tgt), clips=clips))
    return items


def fig_png(pdf: Path, dpi: int = 200) -> str | None:
    """Rasterise a one-page figure PDF to a PNG data URI (needs pdftoppm); None if unavailable."""
    if not pdf.exists() or shutil.which("pdftoppm") is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fig"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(out)], check=True)
        return "data:image/png;base64," + b64(out.with_suffix(".png"))


def table_rows(table: pd.DataFrame, sched: dict, conds: dict, hl: set[str]) -> list[dict]:
    rows = []
    for cond, label in conds.items():
        r = table[table.method == cond]
        if not r.shape[0]:
            continue
        r = r.iloc[0]
        rows.append(dict(cond=cond, label=label, steps=steps_desc(sched[cond]["steps"]) if cond in sched else "—",
                         coh=f"{float(r.coherence_mlsp):.3f}", clap=f"{float(r.clap):.3f}" if pd.notna(r.clap) else "—",
                         hl=cond in hl))
    return rows


# ----------------------------------------------------------------------------- Stable Audio Open 1.0
res, mp3 = Path(args.res), Path(args.mp3)
expb = Path(args.expb)
man = json.load(open(res / "manifest.json"))
sched = json.load(open(expb / "schedules.json"))
per_run = pd.read_csv(expb / "per_run.csv")
table = pd.read_csv(expb / "table.csv")
mp3dirs = [mp3, Path(args.win_mp3) if args.win_mp3 else None]

BLIND = {"sao": "Unguided", "mlsp": "MLSP window (steps 20–48, every 2nd)", "rapg_cal_dc": "SCPG + R-scaling (steps 19–33)"}
LABELED = {"sao": "Unguided",
           "early": "Early window (steps 0–14)", "mid": "Mid window (steps 18–32)", "late": "Late window (steps 35–49)",
           "mlsp": "MLSP window (steps 20–48, every 2nd)", "rapg_cal_dc": "SCPG + R-scaling (steps 19–33)"}
GROUPS = [dict(title="Reference", conds=["sao"]),
          dict(title="Fixed windows, same budget of 15 updates (Table 1 ablation)", conds=["early", "mid", "late"]),
          dict(title="Prior schedule vs. sensitivity-calibrated placement, 15 updates", conds=["mlsp", "rapg_cal_dc"])]

items = build_items(man, per_run, mp3dirs, LABELED)
n_test = int(table[table.method == "sao"].n.iloc[0]) if "n" in table else len(per_run[per_run.method == "sao"])


def num(method: str, col: str = "coherence_mlsp") -> float:
    return float(table[table.method == method][col].iloc[0])


claim = (f"<b>Where the probe is most accurate is not where it steers best.</b> Along real Stable Audio Open trajectories the probe's "
         f"decodability keeps rising until denoising is nearly complete, whereas a single guidance step changes the output most "
         f"around the middle of sampling (Fig. 1). Spending the same budget of 15 updates at the steps of highest measured "
         f"steering sensitivity (SCPG, steps 19–33) raises melodic coherence from <b>{num('mlsp'):.3f}</b> (the fixed window of "
         f"the prior work) to <b>{num('topk_dc_const'):.3f}</b>, and to <b>{num('rapg_cal_dc'):.3f}</b> with reliability-scaled step "
         f"sizes, with CLAP unchanged; the early window (steps 0–14) barely moves the melody at this step size. "
         f"On Stable Audio 3 the sensitive window lies at the start of the trajectory instead (steps 4–18), again far from where the "
         f"probe is most reliable (Fig. 3): the window is model-specific, but one burst sweep locates it in either model.")

figures = []
png = fig_png(Path(args.paper) / "fig_trajectory.pdf")
if png:
    figures.append(dict(src=png, alt="Probe decodability and steering sensitivity along Stable Audio Open trajectories",
                        caption="<b>Fig. 1 (Stable Audio Open).</b> (a) Probe micro-F<sub>1</sub> and target-free reliability R<sub>t</sub> along 50 real trajectories. (b) Change in coherence and CLAP from a 3-step burst at one position (mean, 95% CI). Sensitivity peaks mid-trajectory; decodability keeps rising."))
glance = dict(claim=claim, figures=figures, tables=[
    dict(title=f"Stable Audio Open 1.0 — test-set means (Table 1 of the paper; {n_test} generations per row, 15 updates each)",
         rows=table_rows(table, sched, {**LABELED, "topk_dc_const": "SCPG: Top-K(ΔC), constant step size"}, hl={"rapg_cal_dc", "topk_dc_const"}))],
    foot="Coherence = pitch-class match of the pYIN melody inside target-active regions (higher is better); CLAP = text–audio cosine (prompt adherence). Per-clip values below are for the single generation you hear; the means above are over the whole held-out test set.")

payload = dict(conditions=BLIND, labeled_conditions=LABELED, labeled_groups=GROUPS,
               schedules={c: sched[c]["steps"] for c in LABELED if c in sched},
               adaptive={c: bool(sched[c]["adaptive_strength"]) for c in LABELED if c in sched},
               items=items,
               model_name="Stable Audio Open 1.0",
               model_note="50 EDM-DPM-solver steps, CFG 4.0, 5-s clips; probe of the paper's prior work (125k parameters). Metrics per clip: melodic coherence (pYIN pitch-class match inside target-active regions), chroma cosine, CLAP text–audio score.",
               selection_note="Six prompt–melody pairs were fixed before any listening, to cover all five target melodies (ascending twice) and a range of prompt styles: test prompts 0, 2, 4, 6, 10 and 12 with seed 0. Nothing was cherry-picked; per-run metrics for every one of the 225 test trials are in the linked CSVs.",
               glance=glance, models=[])
print(f"SAO block: {len(items)} items; conditions per item: {[len(it['clips']) for it in items]}")

# ----------------------------------------------------------------------------- Stable Audio 3 (optional)
if args.sa3_res:
    sa3_res, sa3_mp3 = Path(args.sa3_res), Path(args.sa3_mp3)
    sa3_expb = Path(args.sa3_expb or "results/021_sa3_expB")
    man3 = json.load(open(sa3_res / "manifest.json")); sched3 = json.load(open(sa3_expb / "schedules.json"))
    pr3 = pd.read_csv(sa3_expb / "per_run.csv"); tb3 = pd.read_csv(sa3_expb / "table.csv")
    LABELED3 = {"sao": "Unguided",
                "early": "Early window (steps 0–14)", "mid": "Mid window (steps 18–32)", "late": "Late window (steps 35–49)",
                "mlsp": "MLSP window (steps 20–48, every 2nd)", "rapg_cal_dc": "SCPG + R-scaling (steps 4–18)"}
    items3 = build_items(man3, pr3, [sa3_mp3, Path(args.sa3_win_mp3) if args.sa3_win_mp3 else None], LABELED3)
    n3 = int(tb3[tb3.method == "sao"].n.iloc[0]) if "n" in tb3 else len(pr3[pr3.method == "sao"])
    png3 = fig_png(Path(args.paper) / "fig_sa3.pdf")
    if png3:
        glance["figures"].append(dict(src=png3, alt="The same analysis on Stable Audio 3 medium base",
                                      caption="<b>Fig. 3 (Stable Audio 3 medium, base checkpoint).</b> (a) Decodability and reliability along 50 real trajectories; (b) change in coherence and CLAP from a 3-step burst at one position. Here the sensitive window is at the start of the trajectory (noise level t ≥ 0.95) while decodability peaks late."))
    glance["tables"].append(dict(title=f"Stable Audio 3 medium base — test-set means (§4.5 of the paper; {n3} generations per row, 15 updates each)",
                                 rows=table_rows(tb3, sched3, LABELED3, hl={"rapg_cal_dc"})))
    payload["models"].append(dict(id="sa3", name="Stable Audio 3 medium (base checkpoint)", short="SA3",
                                  note="Same prompts, melodies and probe recipe on the 50-step base model (rectified flow, 256-d latents at 10.8 frames/s, CFG 7). Here the sensitivity-calibrated window is at the start of the trajectory (steps 4–18); the fixed windows and the MLSP window are the same step indices as above.",
                                  conditions=LABELED3, groups=GROUPS,
                                  schedules={c: sched3[c]["steps"] for c in LABELED3 if c in sched3},
                                  adaptive={c: bool(sched3[c]["adaptive_strength"]) for c in LABELED3 if c in sched3}, items=items3))
    print(f"SA3 block: {len(items3)} items; conditions per item: {[len(it['clips']) for it in items3]}")

blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
tpl = (HERE / "template.html").read_text()
body = tpl.replace("__DATA__", blob)
out_body = Path(args.out_body); out_body.write_text(body)
print(f"wrote {out_body} ({out_body.stat().st_size/1e6:.2f} MB)")
if args.out_full:
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    out_full = Path(args.out_full); out_full.write_text(full)
    print(f"wrote {out_full} ({out_full.stat().st_size/1e6:.2f} MB)")
