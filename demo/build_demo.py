"""Build the self-contained listening-test / demo page from results/013_collect_demo_v2.

Usage: python demo/build_demo.py <results_dir> <mp3_dir> <out_body.html> [out_full.html]
  out_body.html : page content without doctype/html/head/body (for the claude.ai Artifact tool)
  out_full.html : the same content wrapped in a complete document (for GitHub Pages / local viewing)
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


CONDS = {"sao": "SAO (unguided)", "mlsp": "MLSP-fixed", "rapg_cal_dc": "RAPG-cal(ΔC)"}
items = []
for it in man["items"]:
    tid = f"test-p{it['prompt_id']:02d}-{it['melody']}-s0"
    mel = data.MELODIES[it["melody"]]
    clips = {}
    for cond in CONDS:
        row = per_run[(per_run.trial == tid) & (per_run.method == cond)].iloc[0]
        clips[cond] = dict(audio=b64(mp3dir / f"{tid}__{cond}.mp3"),
                           coh=round(float(row.coherence_mlsp), 3), chroma=round(float(row.chroma_cos), 3),
                           clap=round(float(row.clap), 3), n_updates=int(row.n_updates))
    items.append(dict(id=tid, prompt_id=it["prompt_id"], prompt=it["prompt"], melody=it["melody"],
                      notes=list(mel.notes), tempo=mel.tempo, duration=mel.duration_s,
                      target=b64(mp3dir / f"target_{it['melody']}.mp3"), clips=clips))

payload = dict(conditions=CONDS,
               schedules={c: sched[c]["steps"] for c in CONDS},
               adaptive={c: bool(sched[c]["adaptive_strength"]) for c in CONDS},
               items=items)
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
