"""The Claude Skill must give the same score and reasons as the deployed agent's tools, using only NumPy."""
import json
import subprocess
import sys
import zipfile

import pytest

from fraudagent.export_skill import NAME, export
from fraudagent.tools import FraudTools


@pytest.fixture(scope="module")
def skill(model_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    zip_path = export(model_dir, out)
    return out / NAME, zip_path


def run(skill_dir, *args):
    res = subprocess.run([sys.executable, str(skill_dir / "scripts" / "review_evidence.py"), *args],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_zip_has_skill_folder_at_top_level(skill):
    _, zip_path = skill
    names = zipfile.ZipFile(zip_path).namelist()
    assert f"{NAME}/SKILL.md" in names
    assert all(n.startswith(f"{NAME}/") for n in names)
    assert f"{NAME}/assets/model.json" in names and f"{NAME}/scripts/review_evidence.py" in names


def test_skill_md_frontmatter(skill):
    text = (skill[0] / "SKILL.md").read_text()
    head = text.split("---")[1]
    assert f"name: {NAME}" in head
    desc = [line for line in head.splitlines() if line.startswith("description:")][0]
    assert len(desc) - len("description: ") <= 1024


def test_sample_cases_have_no_labels(skill):
    header = (skill[0] / "assets" / "sample_cases.csv").read_text().splitlines()[0]
    assert "label" not in header and "group" not in header and "Class" not in header


def test_scores_and_reasons_match_the_agent_tools(skill, model_dir):
    tools = FraudTools(model_dir)
    ids = list(tools.cases)
    out = run(skill[0], "--case", *ids)
    assert [o["case_id"] for o in out] == ids
    for o in out:
        cid = o["case_id"]
        assert o["fraud_probability"] == tools.score_transaction(cid)["fraud_probability"]
        assert o["alert_threshold"] == pytest.approx(tools.meta["threshold"], abs=1e-4)
        mine = [(f["feature"], f["contribution_log_odds"]) for f in o["top_factors"]]
        theirs = [(f["feature"], f["contribution"]) for f in tools.explain_score(cid)["top_factors"]]
        assert mine == theirs
        assert o["policy_minimum"] in ("escalate", "monitor", "clear")
        if o["model_alert"]:
            assert o["policy_minimum"] == "escalate"


def test_injection_note_is_carried_and_marked_untrusted(skill):
    listed = run(skill[0], "--list")
    noted = [c["case_id"] for c in listed if c["has_note"]]
    assert noted
    o = run(skill[0], "--case", noted[0])[0]
    assert "UNTRUSTED" in o["note"]["source"] and o["note"]["text"]


def test_own_csv_and_missing_columns(skill, tmp_path):
    src = (skill[0] / "assets" / "sample_cases.csv").read_text().splitlines()
    good = tmp_path / "mine.csv"
    good.write_text("\n".join(src[:3]) + "\n")
    assert len(run(skill[0], "--csv", str(good))) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("Time,Amount\n0,10\n")
    res = subprocess.run([sys.executable, str(skill[0] / "scripts" / "review_evidence.py"), "--csv", str(bad)],
                         capture_output=True, text=True)
    assert res.returncode != 0 and "missing columns" in res.stderr
