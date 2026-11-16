"""Checks of the published benchmark contract, separate from synthetic inference."""
import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def full_results():
    return json.loads((RESULTS / "full_dataset_results.json").read_text(encoding="utf-8"))


def test_full_dataset_protocol_and_units():
    data = full_results()
    assert data["schema_version"] == 1
    p = data["protocol"]
    assert p["dataset"] == "UTD-MHAD"
    assert (p["usable_paired_recordings"], p["actions"], p["people"]) == (861, 27, 8)
    assert (p["folds"], p["training_people_per_fold"], p["test_people_per_fold"]) == (8, 7, 1)
    assert p["seeds"] == [17, 29, 43]
    assert "within person" in p["aggregation"].lower()
    assert "equal weight" in p["aggregation"].lower()
    assert data["metric_units"]["accuracy"] == data["metric_units"]["macro_f1"] == "fraction"
    assert data["branch_parameters"] == {"sensor_cnn": 29883, "hof_gru": 39259, "hof_transformer": 40763}
    scope = data["evidence_scope"].lower()
    assert "synthetic" in scope and "checkpoint" in scope


@pytest.mark.parametrize("model,fusion,displayed", [
    ("hof_gru", "arithmetic", (85.92, 83.75, .541258, .239410, 69142)),
    ("hof_gru", "product", (89.17, 87.52, .694219, .180640, 69142)),
    ("hof_transformer", "arithmetic", (87.85, 86.19, .488315, .212298, 70646)),
    ("hof_transformer", "product", (91.68, 90.64, .597152, .148871, 70646)),
])
def test_benchmark_values_csv_and_readme_agree(model, fusion, displayed):
    systems = full_results()["systems"]
    assert len(systems) == 4
    keys = [(r["model"], r["fusion"]) for r in systems]
    assert len(set(keys)) == 4
    row = next(r for r in systems if (r["model"], r["fusion"]) == (model, fusion))
    actual = (round(100 * row["accuracy"], 2), round(100 * row["macro_f1"], 2),
              round(row["nll"], 6), round(row["brier"], 6), row["parameters"])
    assert actual == displayed
    with (RESULTS / "classification_summary.csv").open(newline="", encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4
    csv_row = next(r for r in csv_rows if (r["model"], r["fusion"]) == (model, fusion))
    for metric in ("accuracy", "macro_f1", "nll", "brier", "parameters"):
        assert float(csv_row[metric]) == pytest.approx(row[metric], abs=1e-12, rel=0)
    # Parse actual Markdown table cells, rather than accepting a stray prose number.
    lines = [line for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
             if line.lstrip().startswith("|")]
    name = "GRU" if model == "hof_gru" else "Transformer"
    candidates = [line for line in lines if name.lower() in line.lower() and fusion in line.lower()]
    assert len(candidates) == 1
    cells = [re.sub(r"[*`%]", "", x.strip()) for x in candidates[0].strip("|").split("|")]
    numbers = [float(x.replace(",", "")) for x in cells if re.fullmatch(r"[\d,.]+", x)]
    assert numbers[:4] == list(displayed[:4])


def test_legacy_result_is_preserved_and_distinct():
    with (RESULTS / "legacy_eight_action_classification.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    r = rows[0]
    assert float(r["accuracy_percent"]) == 98.96
    assert float(r["macro_f1_percent"]) == 98.86
    assert int(r["parameters"]) == 66672
    assert int(r["unique_known_clips"]) == 191
    assert "eight_action" in r["scope"] and "six_subject" in r["scope"]
    # Better closed-set accuracy must not be described as better log loss.
    systems = {(r["model"], r["fusion"]): r for r in full_results()["systems"]}
    for model in ("hof_gru", "hof_transformer"):
        a, p = systems[model, "arithmetic"], systems[model, "product"]
        assert p["accuracy"] > a["accuracy"]
        assert p["nll"] > a["nll"]
        assert p["brier"] < a["brier"]
