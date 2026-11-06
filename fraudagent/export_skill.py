"""Package the fraud review as a Claude Skill that a non-technical analyst can use in the Claude app.

Usage (after training, so models/ holds the model and the agent assets):
    python -m fraudagent.export_skill --model-dir models --out dist

Writes dist/fraud-case-review/ and dist/fraud-case-review.zip (upload the zip under Customize > Skills).
The skill carries the model as plain numbers (no scikit-learn needed), a weighted sample of past transactions for
the "similar cases" check, and the demo cases without their labels.
"""
import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import joblib
import numpy as np

from fraudservice.features import RAW_COLUMNS

NAME = "fraud-case-review"
TEMPLATE = Path(__file__).resolve().parent / "skill_template"
GENUINE_SAMPLE = 0.25  # keep every past fraud and a quarter of genuine transactions, each standing for 4


def export(model_dir: str | Path, out: str | Path, seed: int = 11) -> Path:
    model_dir, out = Path(model_dir), Path(out)
    skill = out / NAME
    if skill.exists():
        shutil.rmtree(skill)
    shutil.copytree(TEMPLATE, skill, ignore=shutil.ignore_patterns("__pycache__"))
    assets = skill / "assets"
    assets.mkdir(exist_ok=True)

    pipe = joblib.load(model_dir / "model.joblib")
    meta = json.loads((model_dir / "metadata.json").read_text())
    scaler, lr = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    ref = np.load(model_dir / "agent_reference.npz")
    X, y = ref["X"], ref["y"]
    m = meta["test_metrics"]
    (assets / "model.json").write_text(json.dumps({
        "model_version": meta["model_version"],
        "features": meta["features"],
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "threshold": float(meta["threshold"]),
        "overall_fraud_rate": round(float(y.mean()), 4),
        "test_metrics": {"recall": round(m["recall"], 3), "precision": round(m["precision"], 3),
                         "pr_auc": round(m["pr_auc"], 3)},
    }, indent=1))

    rng = np.random.default_rng(seed)
    keep = (y == 1) | (rng.random(len(y)) < GENUINE_SAMPLE)
    w = np.where(y == 1, 1.0, 1.0 / GENUINE_SAMPLE)
    np.savez_compressed(assets / "reference.npz", X=X[keep].astype(np.float16), y=y[keep].astype(np.int8),
                        w=w[keep].astype(np.float16))

    cases = json.loads((model_dir / "demo_cases.json").read_text())
    with open(assets / "sample_cases.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case_id", "note"] + RAW_COLUMNS)
        for c in cases:  # labels are deliberately left out
            wr.writerow([c["case_id"], c.get("note", "")] + [c["transaction"][k] for k in RAW_COLUMNS])

    zip_path = out / f"{NAME}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(skill.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    return zip_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    path = export(args.model_dir, args.out)
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
