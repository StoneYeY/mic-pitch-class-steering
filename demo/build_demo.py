"""Build the self-contained listening-test / demo page from results/013_collect_demo_v2.

Usage: python demo/build_demo.py <results_dir> <mp3_dir> <out_body.html> [out_full.html] [expB_dir] [sa3_results_dir] [sa3_mp3_dir] [sa3_expB_dir]
  out_body.html : page content without doctype/html/head/body (for the claude.ai Artifact tool)
  out_full.html : the same content wrapped in a complete document (for GitHub Pages / local viewing)
  sa3_*         : optional second block (Stable Audio 3 medium base) shown in the labeled demo
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from w2s import data  # noqa: E402

res, mp3dir, out_body = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out_full = Path(sys.argv[4]) if len(sys.argv) > 4 else None
expb = Path(sys.argv[5]) if len(sys.argv) > 5 else res.parent / "009_expB_full"

man = json.load(open(res / "manifest.json"))
sched = json.load(open(expb / "schedules.json"))
per_run = pd.read_csv(expb / "per_run.csv")


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


CONDS = {"sao": "Unguided", "mlsp": "MLSP window (steps 20–48, every 2nd)", "rapg_cal_dc": "SCPG + R-scaling (steps 19–33)"}


def build_items(man, per_run, mp3dir, conds):
    items = []
    for it in man["items"]:
        tid = f"test-p{it['prompt_id']:02d}-{it['melody']}-s0"
        mel = data.MELODIES[it["melody"]]
        clips = {}
        for cond in conds:
            rows = per_run[(per_run.trial == tid) & (per_run.method == cond)]
            if not rows.shape[0] or not (mp3dir / f"{tid}__{cond}.mp3").exists():
                continue
            row = rows.iloc[0]
            clap = float(row.clap) if pd.notna(row.clap) else float("nan")
            clips[cond] = dict(audio=b64(mp3dir / f"{tid}__{cond}.mp3"),
                               coh=round(float(row.coherence_mlsp), 3), chroma=round(float(row.chroma_cos), 3),
                               clap=None if clap != clap else round(clap, 3), n_updates=int(row.n_updates))
        if clips:
            items.append(dict(id=tid, prompt_id=it["prompt_id"], prompt=it["prompt"], melody=it["melody"],
                              notes=list(mel.notes), tempo=mel.tempo, duration=mel.duration_s,
                              target=b64(mp3dir / f"target_{it['melody']}.mp3"), clips=clips))
    return items


items = build_items(man, per_run, mp3dir, CONDS)
payload = dict(conditions=CONDS,
               schedules={c: sched[c]["steps"] for c in CONDS},
               adaptive={c: bool(sched[c]["adaptive_strength"]) for c in CONDS},
               items=items,
               model_name="Stable Audio Open 1.0",
               model_note="50 EDM-DPM-solver steps, CFG 4.0, 5-s clips; probe of the paper's prior work (125k parameters). Metrics per clip: melodic coherence (pYIN pitch-class match inside target-active regions), chroma cosine, CLAP text–audio score.",
               selection_note="Six prompt–melody pairs were fixed before any listening, to cover all five target melodies (ascending twice) and a range of prompt styles: test prompts 0, 2, 4, 6, 10 and 12 with seed 0. Nothing was cherry-picked; per-run metrics for every one of the 225 test trials are in the linked CSVs.",
               models=[])
if len(sys.argv) > 6:
    sa3_res, sa3_mp3 = Path(sys.argv[6]), Path(sys.argv[7])
    sa3_expb = Path(sys.argv[8]) if len(sys.argv) > 8 else Path("results/021_sa3_expB")
    man3 = json.load(open(sa3_res / "manifest.json")); sched3 = json.load(open(sa3_expb / "schedules.json"))
    pr3 = pd.read_csv(sa3_expb / "per_run.csv")
    CONDS3 = {"sao": "Unguided", "mlsp": "MLSP window (steps 20–48, every 2nd)", "rapg_cal_dc": "SCPG + R-scaling (steps 4–18)"}
    items3 = build_items(man3, pr3, sa3_mp3, CONDS3)
    payload["models"].append(dict(id="sa3", name="Stable Audio 3 medium (base checkpoint)", short="SA3",
                                  note="Same prompts, melodies and probe recipe on the 50-step base model (rectified flow, 256-d latents at 10.8 frames/s, CFG 7). Here the sensitivity-calibrated window is at the start of the trajectory (steps 4–18); the MLSP window is the same fixed schedule as above.",
                                  conditions=CONDS3, schedules={c: sched3[c]["steps"] for c in CONDS3},
                                  adaptive={c: bool(sched3[c]["adaptive_strength"]) for c in CONDS3}, items=items3))
    print(f"SA3 block: {len(items3)} items")
blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

tpl = (HERE / "template.html").read_text()
body = tpl.replace("__DATA__", blob)
out_body.write_text(body)
print(f"wrote {out_body} ({out_body.stat().st_size/1e6:.2f} MB)")
if out_full:
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<style>body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style>\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    out_full.write_text(full)
    print(f"wrote {out_full} ({out_full.stat().st_size/1e6:.2f} MB)")
