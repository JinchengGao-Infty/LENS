"""Table IV: Recall@15 and AP over the twelve functions with known interactions,
read from the ranking aggregates (results/ranking, or the directory given)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results/ranking"


def load(name):
    return json.loads((ROOT / name).read_text())


family = load("teacher_family_stage1_aggregate.json")["per_teacher"]
reference = load("stage1_reference_aggregate.json")["macro_over_nonnull_dgps"]
h2_ap = load("friedman_h2_full_ap_replay_aggregate.json")["macro_over_dgp_means"]["mean"]
official = load("official_stage1_six_method_aggregate.json")["macro_over_nonnull_dgps"]
hstats = load("official_r_baselines_aggregate.json")["stage1"]["macro_over_nonnull_dgps"]
irf = load("official_irf_aggregate.json")["macro_over_nonnull_dgps"]


def pair(block):
    return block["recall_at_k"]["mean"], block["average_precision"]["mean"]


xgb, mlp = family["xgboost"]["macro_non_null"], family["softplus_mlp"]["macro_non_null"]
rows = [
    ("LENS", "XGBoost", *pair(xgb["LENS-raw"])),
    ("TreeSHAP", "XGBoost", *pair(xgb["TreeSHAP"])),
    ("$H^2$", "XGBoost", reference["Friedman-H2"]["recall_at_k"]["mean"], h2_ap),
    None,
    ("LENS", "MLP", *pair(mlp["LENS-raw"])),
    ("Integrated Hessians", "MLP", *pair(mlp["Integrated-Hessians-same-MLP"])),
    ("NID", "MLP", *pair(mlp["NID-weight-same-MLP"])),
    None,
    *[(m, "-", *pair(official[m])) for m in ("TreeSHAP-IQ", "Archipelago", "SHAP-IQ", "Faith-Shap")],
    ("hstats", "-", *pair(hstats)),
    ("iRF", "-", *pair(irf)),
]
print(f"{'Method':22s} {'Model':8s} Recall@15   AP")
for row in rows:
    if row is None:
        print("-" * 44)
        continue
    name, model, recall, ap = row
    print(f"{name:22s} {model:8s} {recall:9.2f} {ap:6.2f}")
