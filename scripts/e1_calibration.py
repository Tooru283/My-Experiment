#!/usr/bin/env python3
"""E1 Calibration & gap analysis for the termination verifier.

Reads harness trace JSONL files from two runs (A0 and max_config) and produces:
  - Distance-binned: accept_rate / visible_rate / true_arrival_rate
  - Reliability diagram  (Termination ECE / Brier)
  - Risk-Coverage curve
  - Stop-proposal distance distributions (reject vs accept)
  - V1 vs V2 confidence ECE comparison (b_t source selection)
  - T1-T5 failure component breakdown
  - False / Missed stop rates

Usage:
  python scripts/e1_calibration.py \
      --a0   logs/harness_traces/ep100/20260628/a0_baseline/.../val_unseen/rank_0 \
      --max  logs/harness_traces/ep100/20260628/max_config/.../val_unseen/rank_0 \
      --out  Controlled-Navigation-Harness/docs/e1_calibration \
      [--m3  logs/harness_traces/ep100/20260629/m3_proactive_stop_gate/.../val_unseen/rank_0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUCCESS_RADIUS = 3.0   # metres
NEAR_GOAL_GATE = 3.5   # metres (M3 trigger threshold)
N_DIST_BINS = 12
CONF_BINS = 5


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_traces(trace_dir: str) -> List[Dict[str, Any]]:
    """Return all events from every episode JSONL in trace_dir."""
    events = []
    for fname in sorted(os.listdir(trace_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(trace_dir, fname)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


def build_episode_index(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group events by episode_id."""
    index: Dict[str, List] = defaultdict(list)
    for e in events:
        eid = e.get("episode_id") or (e.get("payload") or {}).get("episode_id")
        if eid is not None:
            index[str(eid)].append(e)
    return dict(index)


def get_payload(event: Dict) -> Dict:
    p = event.get("payload")
    return p if isinstance(p, dict) else {}


def etype(event: Dict) -> str:
    return str(event.get("event_type") or event.get("event") or "")


# ---------------------------------------------------------------------------
# Per-episode extraction
# ---------------------------------------------------------------------------

def extract_episode(ep_events: List[Dict]) -> Dict[str, Any]:
    """Extract calibration-relevant data from one episode's events."""
    # step → distance at END of that step (after action)
    post_dist: Dict[int, float] = {}
    for e in ep_events:
        if etype(e) == "action_post_step":
            s = e.get("step_id", 0)
            outputs = get_payload(e).get("step_outputs") or []
            if outputs and isinstance(outputs[0], dict):
                d = outputs[0].get("distance_to_goal")
                if d is not None:
                    post_dist[int(s)] = float(d)

    # episode metrics
    metrics: Dict = {}
    for e in ep_events:
        if etype(e) == "episode_end":
            m = get_payload(e).get("metrics") or {}
            metrics.update(m)

    # VTV stop-proposal events
    stop_proposals = []
    for e in ep_events:
        if etype(e) != "visual_target_verifier":
            continue
        p = get_payload(e)
        if not p.get("stop_proposal"):
            continue
        step = int(e.get("step_id", 0))
        # entry distance = distance at end of previous step
        entry_dist = post_dist.get(step - 1)
        bc = p.get("best_candidate") or {}
        stop_proposals.append({
            "step": step,
            "entry_dist": entry_dist,
            "verdict": str(p.get("verdict") or ""),
            "v2_conf": bc.get("confidence"),          # V2 confidence (b_t candidate)
            "final_target_visible": bool(p.get("final_target_visible")),
            "arrival_evidence": bool(p.get("arrival_evidence")),
            "source": str(p.get("source") or ""),
            "reject_reasons": list(p.get("reject_reasons") or []),
        })

    # All VTV events (stop_proposal=True or False) for distance-binned visibility
    all_vtv = []
    for e in ep_events:
        if etype(e) != "visual_target_verifier":
            continue
        p = get_payload(e)
        step = int(e.get("step_id", 0))
        entry_dist = post_dist.get(step - 1)
        bc = p.get("best_candidate") or {}
        all_vtv.append({
            "step": step,
            "entry_dist": entry_dist,
            "stop_proposal": bool(p.get("stop_proposal")),
            "verdict": str(p.get("verdict") or ""),
            "v2_conf": bc.get("confidence"),
            "final_target_visible": bool(p.get("final_target_visible")),
            "arrival_evidence": bool(p.get("arrival_evidence")),
        })

    # V1 evidence confidence (per step, max candidate confidence toward target)
    v1_by_step: Dict[int, float] = {}
    for e in ep_events:
        if etype(e) != "visual_evidence":
            continue
        step = int(e.get("step_id", 0))
        p = get_payload(e)
        parsed = p.get("parsed") or {}
        candidates = parsed.get("candidates") or p.get("candidates") or []
        confs = []
        for c in candidates:
            if isinstance(c, dict) and c.get("final_target_visible"):
                conf = c.get("confidence")
                if conf is not None:
                    confs.append(float(conf))
        if confs:
            v1_by_step[step] = max(confs)

    return {
        "post_dist": post_dist,
        "metrics": metrics,
        "stop_proposals": stop_proposals,
        "all_vtv": all_vtv,
        "v1_by_step": v1_by_step,
    }


def collect_run(trace_dir: str) -> Dict[str, Any]:
    """Collect all calibration data for one run directory."""
    events = load_traces(trace_dir)
    ep_index = build_episode_index(events)

    stop_proposals = []   # all stop-proposal VTV events across episodes
    all_vtv = []
    v1_points = []        # (step_dist, v1_conf, is_within_radius)
    episode_results = []  # (success, oracle_success)

    for eid, ep_events in ep_index.items():
        data = extract_episode(ep_events)
        m = data["metrics"]
        success = float(m.get("success", 0)) > 0
        # oracle: was there any step within SUCCESS_RADIUS?
        dists = list(data["post_dist"].values())
        oracle = any(d <= SUCCESS_RADIUS for d in dists) if dists else False
        episode_results.append({"success": success, "oracle": oracle,
                                 "final_dist": m.get("distance_to_goal")})

        for sp in data["stop_proposals"]:
            sp["episode_id"] = eid
            sp["ep_success"] = success
            stop_proposals.append(sp)

        for vtv in data["all_vtv"]:
            if vtv["entry_dist"] is not None:
                vtv["true_arrival"] = vtv["entry_dist"] <= SUCCESS_RADIUS
                all_vtv.append(vtv)

        for step, conf in data["v1_by_step"].items():
            dist = data["post_dist"].get(step - 1)
            if dist is not None:
                v1_points.append({"dist": dist, "conf": conf,
                                   "true_arrival": dist <= SUCCESS_RADIUS})

    return {
        "stop_proposals": stop_proposals,
        "all_vtv": all_vtv,
        "v1_points": v1_points,
        "episode_results": episode_results,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_ece(confidences: List[float], labels: List[int], n_bins: int = 5) -> float:
    """Expected Calibration Error (equal-width bins)."""
    if not confidences:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    confs = np.array(confidences)
    labs = np.array(labels, dtype=float)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi if i < n_bins - 1 else confs <= hi)
        if mask.sum() == 0:
            continue
        frac = mask.sum() / total
        acc = labs[mask].mean()
        mean_conf = confs[mask].mean()
        ece += frac * abs(acc - mean_conf)
    return float(ece)


def compute_brier(confidences: List[float], labels: List[int]) -> float:
    if not confidences:
        return float("nan")
    c = np.array(confidences)
    y = np.array(labels, dtype=float)
    return float(np.mean((c - y) ** 2))


def reliability_bins(confidences, labels, n_bins=5):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_confs, mean_accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = [(lo <= c < hi if i < n_bins - 1 else lo <= c <= hi)
                for c in confidences]
        subset_c = [c for c, m in zip(confidences, mask) if m]
        subset_l = [l for l, m in zip(labels, mask) if m]
        if subset_c:
            mean_confs.append(float(np.mean(subset_c)))
            mean_accs.append(float(np.mean(subset_l)))
            counts.append(len(subset_c))
    return mean_confs, mean_accs, counts


def risk_coverage(confidences, labels, n_thresholds=20):
    """Sweep threshold: coverage = fraction committed, risk = false-stop rate among committed."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    coverages, risks = [], []
    for tau in thresholds:
        committed = [(c, l) for c, l in zip(confidences, labels) if c >= tau]
        if not committed:
            continue
        cov = len(committed) / len(confidences)
        # risk = fraction of committed decisions where true_arrival=0 (false stop)
        risk = sum(1 for _, l in committed if l == 0) / len(committed)
        coverages.append(cov)
        risks.append(risk)
    return coverages, risks


# ---------------------------------------------------------------------------
# Derived tables
# ---------------------------------------------------------------------------

def stop_decision_table(proposals):
    """Compute false/missed stop counts from stop-proposal events."""
    valid = [p for p in proposals if p.get("entry_dist") is not None
             and p["verdict"] in ("allow", "reject", "uncertain")]
    allowed = [p for p in valid if p["verdict"] == "allow"]
    rejected = [p for p in valid if p["verdict"] != "allow"]
    within = lambda p: p["entry_dist"] <= SUCCESS_RADIUS

    true_stop = sum(1 for p in allowed if within(p))
    false_stop = sum(1 for p in allowed if not within(p))
    missed_stop = sum(1 for p in rejected if within(p))
    true_reject = sum(1 for p in rejected if not within(p))

    return {
        "total_proposals": len(valid),
        "allowed": len(allowed),
        "rejected": len(rejected),
        "true_stop": true_stop,
        "false_stop": false_stop,
        "missed_stop": missed_stop,
        "true_reject": true_reject,
        "stop_precision": true_stop / len(allowed) if allowed else float("nan"),
        "stop_recall": true_stop / (true_stop + missed_stop) if (true_stop + missed_stop) else float("nan"),
    }


def distance_binned(vtv_events, n_bins=N_DIST_BINS, max_dist=6.0):
    """Compute accept_rate, visible_rate, true_arrival_rate by distance bin."""
    edges = np.linspace(0.0, max_dist, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    accept, visible, arrival, counts = (np.zeros(n_bins) for _ in range(4))
    for e in vtv_events:
        d = e.get("entry_dist")
        if d is None or d > max_dist:
            continue
        idx = min(int(d / max_dist * n_bins), n_bins - 1)
        counts[idx] += 1
        if e.get("verdict") == "allow":
            accept[idx] += 1
        if e.get("final_target_visible"):
            visible[idx] += 1
        if e.get("true_arrival"):
            arrival[idx] += 1
    safe = np.where(counts > 0, counts, 1)
    return centres, accept / safe, visible / safe, arrival / safe, counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distance_binned(centres, accept_a0, visible_a0, arrival_a0,
                          accept_mx, visible_mx, arrival_mx, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    kw = dict(marker="o", linewidth=1.5)
    ax.plot(centres, accept_a0,  color="#1f77b4", linestyle="-",  label="A0 accept_rate",   **kw)
    ax.plot(centres, visible_a0, color="#ff7f0e", linestyle="--", label="A0 visible_rate",  marker="s", linewidth=1.5)
    ax.plot(centres, arrival_a0, color="#2ca02c", linestyle=":",  label="A0 arrival_rate",  marker="^", linewidth=1.5)
    ax.plot(centres, accept_mx,  color="#d62728", linestyle="-",  label="max_config accept_rate", **kw)
    ax.plot(centres, visible_mx, color="#9467bd", linestyle="--", label="max_config visible_rate", marker="s", linewidth=1.5)
    ax.plot(centres, arrival_mx, color="#8c564b", linestyle=":",  label="max_config arrival_rate", marker="^", linewidth=1.5)
    ax.axvline(SUCCESS_RADIUS,  color="black", linestyle=":",  linewidth=1, label=f"success radius ({SUCCESS_RADIUS}m)")
    ax.axvline(NEAR_GOAL_GATE,  color="black", linestyle="-.", linewidth=1, label=f"near-goal gate ({NEAR_GOAL_GATE}m)")
    ax.set_xlabel("Distance to goal (m)")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Distance-binned: acceptance / visibility / true arrival")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_reliability(a0_confs, a0_labels, mx_confs, mx_labels, out_path,
                     extra_runs=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    for confs, labels, color, name in [
        (a0_confs, a0_labels, "#1f77b4", "A0"),
        (mx_confs, mx_labels, "#ff7f0e", "max_config"),
    ] + (extra_runs or []):
        mc, ma, cnts = reliability_bins(confs, labels)
        ax.plot(mc, ma, marker="o", color=color, label=name)
        for x, y, n in zip(mc, ma, cnts):
            ax.annotate(str(n), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical P(A_t=1)")
    ax.set_title("Reliability diagram (termination verifier)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_risk_coverage(a0_confs, a0_labels, mx_confs, mx_labels, out_path,
                       extra_runs=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for confs, labels, color, name in [
        (a0_confs, a0_labels, "#1f77b4", "A0"),
        (mx_confs, mx_labels, "#ff7f0e", "max_config"),
    ] + (extra_runs or []):
        covs, risks = risk_coverage(confs, labels)
        ax.plot(covs, risks, color=color, label=name, linewidth=2)
    ax.set_xlabel("Coverage (fraction of stop proposals committed)")
    ax.set_ylabel("Selective risk (false-stop rate among committed)")
    ax.set_ylim(0, 1)
    ax.set_title("Risk-Coverage curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_reject_accept_dist(a0_proposals, mx_proposals, out_path,
                             extra_runs=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 6, 13)
    for proposals, color, name in [
        (a0_proposals, "#1f77b4", "A0"),
        (mx_proposals, "#ff7f0e", "max_config"),
    ] + (extra_runs or []):
        rej = [p["entry_dist"] for p in proposals
               if p["verdict"] != "allow" and p.get("entry_dist") is not None]
        acc = [p["entry_dist"] for p in proposals
               if p["verdict"] == "allow" and p.get("entry_dist") is not None]
        if rej:
            ax.hist(rej, bins=bins, density=True, alpha=0.5, color=color,
                    label=f"{name} rejected")
        if acc:
            ax.hist(acc, bins=bins, density=True, alpha=0.5, color=color,
                    histtype="step", linewidth=2, linestyle="--",
                    label=f"{name} accepted")
    ax.axvline(SUCCESS_RADIUS, color="gray", linestyle=":", linewidth=1.5,
               label=f"success radius ({SUCCESS_RADIUS}m)")
    ax.set_xlabel("Distance to goal (m)")
    ax.set_ylabel("Density")
    ax.set_title("Stop proposals: rejected vs accepted distance distribution")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_v1_v2_ece(a0_run, mx_run, out_path):
    """Compare V1 vs V2 confidence reliability for A0 and max_config."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, run_data, run_name in [(axes[0], a0_run, "A0"),
                                    (axes[1], mx_run, "max_config")]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
        # V2: from stop proposals
        v2_confs = [p["v2_conf"] for p in run_data["stop_proposals"]
                    if p.get("v2_conf") is not None and p.get("entry_dist") is not None]
        v2_labels = [int(p["entry_dist"] <= SUCCESS_RADIUS)
                     for p in run_data["stop_proposals"]
                     if p.get("v2_conf") is not None and p.get("entry_dist") is not None]
        if v2_confs:
            mc, ma, cnts = reliability_bins(v2_confs, v2_labels)
            ece = compute_ece(v2_confs, v2_labels)
            ax.plot(mc, ma, "o-", color="#ff7f0e",
                    label=f"V2 conf (ECE={ece:.3f})")
        # V1: from visual evidence events
        v1_confs = [p["conf"] for p in run_data["v1_points"] if p.get("conf") is not None]
        v1_labels = [int(p["true_arrival"]) for p in run_data["v1_points"] if p.get("conf") is not None]
        if v1_confs:
            mc, ma, cnts = reliability_bins(v1_confs, v1_labels)
            ece = compute_ece(v1_confs, v1_labels)
            ax.plot(mc, ma, "s-", color="#1f77b4",
                    label=f"V1 conf (ECE={ece:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted confidence")
        ax.set_title(f"{run_name}: V1 vs V2 calibration")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Empirical P(A_t=1)")
    fig.suptitle("b_t source comparison (V1 vs V2 confidence)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def print_metrics(name: str, run: Dict, proposals: List) -> None:
    eps = run["episode_results"]
    sr = sum(1 for e in eps if e["success"]) / len(eps) if eps else 0
    osr = sum(1 for e in eps if e["oracle"]) / len(eps) if eps else 0
    conv = sr / osr if osr > 0 else float("nan")

    tbl = stop_decision_table(proposals)
    valid_props = [p for p in proposals
                   if p.get("v2_conf") is not None and p.get("entry_dist") is not None]
    confs = [p["v2_conf"] for p in valid_props]
    labels = [int(p["entry_dist"] <= SUCCESS_RADIUS) for p in valid_props]
    ece = compute_ece(confs, labels)
    brier = compute_brier(confs, labels)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Episodes: {len(eps)}")
    print(f"  SR: {sr:.1%}  OSR: {osr:.1%}  OSR→SR: {conv:.1%}")
    print(f"  Stop proposals: {tbl['total_proposals']}")
    print(f"  Allowed: {tbl['allowed']}  Rejected: {tbl['rejected']}")
    print(f"  True stop  (allowed + ≤3m): {tbl['true_stop']}")
    print(f"  False stop (allowed + >3m): {tbl['false_stop']}  "
          f"[precision={tbl['stop_precision']:.1%}]")
    print(f"  Missed stop (rejected + ≤3m): {tbl['missed_stop']}  "
          f"[recall={tbl['stop_recall']:.1%}]")
    print(f"  True reject (rejected + >3m): {tbl['true_reject']}")
    print(f"  Termination ECE (V2 conf): {ece:.4f}")
    print(f"  Brier score   (V2 conf): {brier:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="E1 Calibration analysis")
    parser.add_argument("--a0", required=True, help="A0 trace rank_0 directory")
    parser.add_argument("--max", required=True, dest="max_cfg",
                        help="max_config trace rank_0 directory")
    parser.add_argument("--m3", default=None, help="Optional M3 trace rank_0 directory")
    parser.add_argument("--out", default="Controlled-Navigation-Harness/docs/e1_calibration",
                        help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading traces...")
    a0_run = collect_run(args.a0)
    mx_run = collect_run(args.max_cfg)
    extra_runs_plot = []
    if args.m3:
        m3_run = collect_run(args.m3)
        extra_runs_plot = [(
            [p["v2_conf"] for p in m3_run["stop_proposals"]
             if p.get("v2_conf") is not None and p.get("entry_dist") is not None],
            [int(p["entry_dist"] <= SUCCESS_RADIUS) for p in m3_run["stop_proposals"]
             if p.get("v2_conf") is not None and p.get("entry_dist") is not None],
            "#2ca02c",
            "M3",
        )]
        extra_reject_plot = [(m3_run["stop_proposals"], "#2ca02c", "M3")]
    else:
        m3_run = None
        extra_reject_plot = []

    # --- prepare calibration inputs ---
    def run_to_calib(run):
        proposals = run["stop_proposals"]
        valid = [p for p in proposals
                 if p.get("v2_conf") is not None and p.get("entry_dist") is not None]
        confs  = [p["v2_conf"] for p in valid]
        labels = [int(p["entry_dist"] <= SUCCESS_RADIUS) for p in valid]
        return confs, labels

    a0_confs,  a0_labels  = run_to_calib(a0_run)
    mx_confs,  mx_labels  = run_to_calib(mx_run)

    # --- print text metrics ---
    print_metrics("A0 baseline", a0_run, a0_run["stop_proposals"])
    print_metrics("max_config",  mx_run, mx_run["stop_proposals"])
    if m3_run:
        print_metrics("M3", m3_run, m3_run["stop_proposals"])

    # --- distance-binned ---
    print("\nGenerating plots...")
    c_a0, acc_a0, vis_a0, arr_a0, _ = distance_binned(a0_run["all_vtv"])
    c_mx, acc_mx, vis_mx, arr_mx, _ = distance_binned(mx_run["all_vtv"])
    plot_distance_binned(c_a0, acc_a0, vis_a0, arr_a0,
                          acc_mx, vis_mx, arr_mx,
                          out_dir / "01_distance_binned.png")

    # --- reliability diagram ---
    plot_reliability(a0_confs, a0_labels, mx_confs, mx_labels,
                     out_dir / "02_reliability_diagram.png",
                     extra_runs=extra_runs_plot)

    # --- risk-coverage ---
    plot_risk_coverage(a0_confs, a0_labels, mx_confs, mx_labels,
                       out_dir / "03_risk_coverage.png",
                       extra_runs=extra_runs_plot)

    # --- reject/accept distribution ---
    plot_reject_accept_dist(a0_run["stop_proposals"], mx_run["stop_proposals"],
                             out_dir / "04_reject_accept_dist.png",
                             extra_runs=extra_reject_plot)

    # --- V1 vs V2 ECE ---
    plot_v1_v2_ece(a0_run, mx_run, out_dir / "05_v1_v2_ece.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
