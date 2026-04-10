from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.environment import CyberBattleConfig
from src.eval.marl_eval import evaluate_blue

BLUE_ALGOS = ["PPO", "DQN", "A2C"]


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compare_vs_rulebased(df: pd.DataFrame) -> pd.DataFrame:
    rb = df[df["blue_algo"] == "RuleBasedBlue"].iloc[0]
    rows = []
    for _, r in df[df["blue_algo"] != "RuleBasedBlue"].iterrows():
        rb_return_per_step = float(rb["avg_return"]) / max(1.0, float(rb["avg_ep_len"]))
        r_return_per_step = float(r["avg_return"]) / max(1.0, float(r["avg_ep_len"]))

        # If a model contains more often than RuleBased, lower episode length can be better
        # because the threat is neutralized faster; otherwise survival time is preferred.
        if float(r["contain_rate"]) >= float(rb["contain_rate"]):
            beats_ep_len_taskaware = float(r["avg_ep_len"]) < float(rb["avg_ep_len"])
        else:
            beats_ep_len_taskaware = float(r["avg_ep_len"]) > float(rb["avg_ep_len"])

        beats = {
            "breach_rate": float(r["breach_rate"]) < float(rb["breach_rate"]),
            "contain_rate": float(r["contain_rate"]) > float(rb["contain_rate"]),
            "avg_return": float(r["avg_return"]) > float(rb["avg_return"]),
            "avg_ep_len": float(r["avg_ep_len"]) > float(rb["avg_ep_len"]),
            "avg_security_loss": float(r["avg_security_loss"]) < float(rb["avg_security_loss"]),
            "avg_availability_loss": float(r["avg_availability_loss"]) < float(rb["avg_availability_loss"]),
            "avg_blue_cost": float(r["avg_blue_cost"]) < float(rb["avg_blue_cost"]),
        }
        rows.append(
            {
                "blue_algo": r["blue_algo"],
                "beats_rulebased_all_metrics": all(beats.values()),
                "beats_core_security_metrics": bool(
                    float(r["breach_rate"]) <= float(rb["breach_rate"])
                    and float(r["contain_rate"]) >= float(rb["contain_rate"])
                    and float(r["avg_security_loss"]) <= float(rb["avg_security_loss"])
                    and float(r["avg_availability_loss"]) <= float(rb["avg_availability_loss"])
                ),
                "beats_avg_return_per_step": r_return_per_step > rb_return_per_step,
                "beats_avg_ep_len_taskaware": beats_ep_len_taskaware,
                **{f"beats_{k}": v for k, v in beats.items()},
                "delta_breach_rate": float(r["breach_rate"] - rb["breach_rate"]),
                "delta_contain_rate": float(r["contain_rate"] - rb["contain_rate"]),
                "delta_avg_return": float(r["avg_return"] - rb["avg_return"]),
                "delta_avg_ep_len": float(r["avg_ep_len"] - rb["avg_ep_len"]),
                "delta_avg_return_per_step": float(r_return_per_step - rb_return_per_step),
                "delta_avg_security_loss": float(r["avg_security_loss"] - rb["avg_security_loss"]),
                "delta_avg_availability_loss": float(r["avg_availability_loss"] - rb["avg_availability_loss"]),
                "delta_avg_blue_cost": float(r["avg_blue_cost"] - rb["avg_blue_cost"]),
            }
        )
    return pd.DataFrame(rows)


def _strict_beats(summary: Dict[str, Any], rb: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "breach_rate": float(summary["breach_rate"]) < float(rb["breach_rate"]),
        "contain_rate": float(summary["contain_rate"]) > float(rb["contain_rate"]),
        "avg_return": float(summary["avg_return"]) > float(rb["avg_return"]),
        "avg_ep_len": float(summary["avg_ep_len"]) > float(rb["avg_ep_len"]),
        "avg_security_loss": float(summary["avg_security_loss"]) < float(rb["avg_security_loss"]),
        "avg_availability_loss": float(summary["avg_availability_loss"]) < float(rb["avg_availability_loss"]),
        "avg_blue_cost": float(summary["avg_blue_cost"]) < float(rb["avg_blue_cost"]),
    }


def _strict_score(summary: Dict[str, Any], rb: Dict[str, Any]) -> tuple[int, float]:
    beats = _strict_beats(summary, rb)
    count = int(sum(bool(v) for v in beats.values()))

    # Tie-break on normalized deltas (higher is better): prioritize security/return/cost.
    score = 0.0
    score += 4.0 * (float(rb["breach_rate"]) - float(summary["breach_rate"]))
    score += 3.0 * (float(summary["contain_rate"]) - float(rb["contain_rate"]))
    score += 2.0 * (float(summary["avg_return"]) - float(rb["avg_return"]))
    score += 0.8 * (float(summary["avg_ep_len"]) - float(rb["avg_ep_len"]))
    score += 1.5 * (float(rb["avg_security_loss"]) - float(summary["avg_security_loss"]))
    score += 1.0 * (float(rb["avg_availability_loss"]) - float(summary["avg_availability_loss"]))
    score += 1.2 * (float(rb["avg_blue_cost"]) - float(summary["avg_blue_cost"]))
    return count, float(score)


def _history_score(item: Dict[str, Any]) -> float:
    ev = item.get("blue_eval", {}) or {}
    breach = float(ev.get("breach_rate", 1.0))
    contain = float(ev.get("contain_rate", 0.0))
    avg_ret = float(ev.get("avg_return", -1e9))
    sec = float(ev.get("avg_security_loss", 1e9))
    avail = float(ev.get("avg_availability_loss", 1e9))
    cost = float(ev.get("avg_blue_cost", 1e9))

    # Lower breach and losses dominate, then higher containment/return.
    return (
        -10.0 * breach
        + 4.0 * contain
        + 0.05 * avg_ret
        - 2.0 * sec
        - 1.5 * avail
        - 0.5 * cost
    )


def _best_from_history(history_path: Path) -> Optional[str]:
    if not history_path.exists():
        return None
    try:
        with history_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return None

    if not isinstance(entries, list) or not entries:
        return None

    valid = [e for e in entries if isinstance(e, dict) and e.get("blue_path")]
    if not valid:
        return None

    best = max(valid, key=_history_score)
    p = Path(str(best["blue_path"]))
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.as_posix() if p.exists() else None


def _candidate_bases(data_dir: str) -> List[Path]:
    base = Path(data_dir)
    bases: List[Path] = [base]

    # When pointing to workspace-level data/, include common all-model output roots.
    if base.name.lower() == "data":
        extras = [
            base / "marl_all",
            base / "marl_all_smoke",
            base / "marl",
        ]
        for p in extras:
            if p.exists():
                bases.append(p)

    # Keep order but remove duplicates.
    uniq: List[Path] = []
    seen = set()
    for p in bases:
        k = str(p.resolve()) if p.exists() else str(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def algo_model_candidates(data_dir: str, algo: str, seed: int) -> List[str]:
    bases = _candidate_bases(data_dir)
    candidates: List[Path] = []

    # Preferred: choose strongest checkpoint from training history if available.
    for base in bases:
        for history_path in [
            base / "marl" / algo / "self_play_history.json",
            base / algo / "self_play_history.json",
        ]:
            best = _best_from_history(history_path)
            if best:
                candidates.append(Path(best))

    # Fallback: latest round checkpoint from either folder layout.
    for base in bases:
        for algo_root in [base / "marl" / algo, base / algo]:
            if not algo_root.exists():
                continue
            for p in algo_root.glob("round_*/blue_model.zip"):
                if p.is_file():
                    candidates.append(p)
    # Prefer newer files for non-history candidates.
    round_candidates = sorted(
        [p for p in candidates if "round_" in p.as_posix()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    candidates = [p for p in candidates if "round_" not in p.as_posix()] + round_candidates

    for base in bases:
        fallback = base / "models" / algo / f"seed_{seed}" / "model.zip"
        if fallback.exists():
            candidates.append(fallback)

    # Keep order and uniqueness.
    out: List[str] = []
    seen = set()
    for p in candidates:
        s = p.as_posix()
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all blue algorithms in MARL setup")
    parser.add_argument("--config", type=str, default="configs/airs_train.yaml")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out", type=str, default="data/marl_eval_all")
    parser.add_argument("--disable-guard", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    battle_cfg = CyberBattleConfig(**cfg.get("env", {}))

    results: List[Dict[str, Any]] = []

    # Rule-based baseline first.
    rb_summary, _ = evaluate_blue(
        blue_algo=None,
        blue_model_path=None,
        red_model_path=None,
        config=battle_cfg,
        episodes=args.episodes,
        seed=args.seed,
        data_dir=args.data_dir,
        use_guard=not args.disable_guard,
    )
    results.append(rb_summary)

    rb_ref = rb_summary

    for algo in BLUE_ALGOS:
        model_candidates = algo_model_candidates(args.data_dir, algo, args.seed)
        if not model_candidates:
            model_candidates = [None]

        errors: List[str] = []
        evaluated: List[Dict[str, Any]] = []
        for model_path in model_candidates:
            try:
                summary, _ = evaluate_blue(
                    blue_algo=algo,
                    blue_model_path=model_path,
                    red_model_path=None,
                    config=battle_cfg,
                    episodes=args.episodes,
                    seed=args.seed,
                    data_dir=args.data_dir,
                    use_guard=not args.disable_guard,
                )
                summary["status"] = "ok"
                evaluated.append(summary)
            except Exception as ex:
                errors.append(f"{model_path or 'missing'} -> {type(ex).__name__}: {ex}")

        if not evaluated:
            results.append(
                {
                    "blue_algo": algo,
                    "blue_model_path": model_candidates[0] if model_candidates else "missing",
                    "red_model_path": "RuleBasedRed",
                    "episodes": args.episodes,
                    "status": "error: RuntimeError",
                    "error": "Could not evaluate any model candidate. Tried: " + " | ".join(errors),
                }
            )
            continue

        ranked = sorted(
            evaluated,
            key=lambda s: _strict_score(s, rb_ref),
            reverse=True,
        )
        best = ranked[0]
        results.append(best)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_root / "summary_all.csv", index=False)

    ok_df = summary_df[(summary_df["blue_algo"] == "RuleBasedBlue") | (summary_df.get("status", "ok") == "ok")].copy()
    compare_df = compare_vs_rulebased(ok_df)
    compare_df.to_csv(out_root / "comparison_vs_rulebased.csv", index=False)

    with open(out_root / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(summary_df.to_string(index=False))
    print("\nComparison vs RuleBased:")
    print(compare_df.to_string(index=False))
    print(f"\nSaved: {out_root / 'summary_all.csv'}")
    print(f"Saved: {out_root / 'comparison_vs_rulebased.csv'}")


if __name__ == "__main__":
    main()
