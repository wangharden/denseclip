import argparse
import json
import math
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Stage3 training outputs for early contract failures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank-min", type=float, default=1e-4)
    parser.add_argument("--min-rank-epochs", type=int, default=2)
    parser.add_argument("--min-gate-mean", type=float, default=0.003)
    parser.add_argument("--min-residual-mean", type=float, default=0.001)
    parser.add_argument("--min-activity-epochs", type=int, default=3)
    parser.add_argument("--identity-tolerance", type=float, default=1e-6)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--stall-seconds", type=float, default=900.0)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def collect_alerts(
    output_dir: Path,
    rank_min: float,
    min_rank_epochs: int,
    min_gate_mean: float,
    min_residual_mean: float,
    min_activity_epochs: int,
    identity_tolerance: float,
) -> dict[str, object]:
    alerts: list[str] = []
    notes: list[str] = []

    identity = load_json(output_dir / "identity_check.json")
    if identity is None:
        notes.append("identity_check_missing")
    else:
        if "max_abs_logit_diff" in identity:
            logit_drift = float(identity.get("max_abs_logit_diff", 0.0))
            if logit_drift > identity_tolerance:
                alerts.append(f"identity_drift logit={logit_drift:.6g} tol={identity_tolerance:.6g}")
        else:
            image_drift = float(identity.get("max_abs_image_diff", 0.0))
            pixel_drift = float(identity.get("max_abs_pixel_diff", 0.0))
            if image_drift > identity_tolerance or pixel_drift > identity_tolerance:
                alerts.append(
                    f"identity_drift image={image_drift:.6g} pixel={pixel_drift:.6g} tol={identity_tolerance:.6g}"
                )

    rank_check = load_json(output_dir / "rank_check.json")
    stats = load_json(output_dir / "map_stats.json")
    if stats is None:
        stats = load_json(output_dir / "rescal_stats.json")
    if stats is None:
        stats = load_json(output_dir / "calibrator_stats.json")
    if rank_check is not None:
        initial_rank = float(rank_check.get("rank_smooth", 0.0))
        if initial_rank <= rank_min:
            alerts.append(f"initial_rank_inactive rank_smooth={initial_rank:.6g} rank_min={rank_min:.6g}")
    elif stats is not None:
        initial_rank = float(((stats.get("initial_rank_activity") or {}).get("rank_smooth", 0.0)))
        if initial_rank <= rank_min:
            alerts.append(f"initial_rank_inactive rank_smooth={initial_rank:.6g} rank_min={rank_min:.6g}")
    else:
        notes.append("rank_check_missing")

    rows = load_jsonl(output_dir / "train_metrics.jsonl")
    latest_epoch = int(rows[-1].get("global_step", rows[-1].get("epoch", 0))) if rows else 0
    if not rows:
        notes.append("train_metrics_missing")
    else:
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    alerts.append(f"non_finite_metric epoch={row.get('epoch')} key={key} value={value}")
        recent = rows[-min_rank_epochs:]
        if len(recent) >= min_rank_epochs:
            rank_values = [abs(float(row.get("rank", 0.0))) for row in recent]
            if all(value <= rank_min for value in rank_values):
                alerts.append(
                    "rank_inactive_recent "
                    + ",".join(f"epoch{int(row['epoch'])}={float(row.get('rank', 0.0)):.6g}" for row in recent)
                )
        activity_recent = rows[-min_activity_epochs:]
        if len(activity_recent) >= min_activity_epochs:
            gate_values = [float(row.get("gate_mean", 0.0)) for row in activity_recent]
            residual_values = [float(row.get("residual_abs_mean", 0.0)) for row in activity_recent]
            if all(value <= min_gate_mean for value in gate_values) and all(
                value <= min_residual_mean for value in residual_values
            ):
                alerts.append(
                    "activity_collapsed_recent "
                    + ",".join(
                        (
                            f"step{int(row.get('global_step', row.get('epoch', 0)))}"
                            f":gate={float(row.get('gate_mean', 0.0)):.6g}"
                            f":residual={float(row.get('residual_abs_mean', 0.0)):.6g}"
                        )
                        for row in activity_recent
                    )
                )

    completed = (output_dir / "train_history.json").exists()
    status = "alert" if alerts else "ok"
    if not completed and not rows and identity is None:
        status = "waiting"
    return {
        "status": status,
        "output_dir": str(output_dir),
        "completed": completed,
        "latest_epoch": latest_epoch,
        "alerts": alerts,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    start_time = time.monotonic()
    last_epoch = -1
    last_progress_time = start_time

    while True:
        summary = collect_alerts(
            output_dir=output_dir,
            rank_min=args.rank_min,
            min_rank_epochs=args.min_rank_epochs,
            min_gate_mean=args.min_gate_mean,
            min_residual_mean=args.min_residual_mean,
            min_activity_epochs=args.min_activity_epochs,
            identity_tolerance=args.identity_tolerance,
        )
        print(json.dumps(summary, ensure_ascii=False))

        if not args.follow:
            raise SystemExit(2 if summary["status"] == "alert" else 0)

        latest_epoch = int(summary["latest_epoch"])
        if latest_epoch > last_epoch:
            last_epoch = latest_epoch
            last_progress_time = time.monotonic()

        if summary["status"] == "alert":
            raise SystemExit(2)
        if bool(summary["completed"]):
            raise SystemExit(0)
        if time.monotonic() - start_time > args.timeout_seconds:
            raise SystemExit(3)
        if time.monotonic() - last_progress_time > args.stall_seconds:
            raise SystemExit(4)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
