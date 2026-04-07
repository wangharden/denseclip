import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.data import build_shared_split_manifest, load_shared_split_manifest, save_shared_split_manifest


MVTEC_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)

MODE_SCREENING = "screening"
MODE_FINALIST = "finalist"

EXPERIMENT_BASELINE = "baseline"
EXPERIMENT_SUBSPACE_D8 = "subspace_d8"
EXPERIMENT_CHOICES = (EXPERIMENT_BASELINE, EXPERIMENT_SUBSPACE_D8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage2 P4 full-15-category visual evaluations.")
    parser.add_argument("--mode", choices=(MODE_SCREENING, MODE_FINALIST), default=MODE_SCREENING)
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--manifests-dir", default="outputs/split_manifests/stage2")
    parser.add_argument("--output-dir", default="outputs/stage2/p4_full15")
    parser.add_argument("--categories", nargs="+", default=list(MVTEC_CATEGORIES))
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENT_CHOICES)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--support-normal-k", type=int, default=16)
    parser.add_argument("--support-defect-k", type=int, default=4)
    parser.add_argument("--cache-root", default="outputs/feature_cache")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--pretrained", default="pretrained/RN50.pt")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--feature-layer", default="layer4")
    parser.add_argument("--score-mode", default="neg-normal")
    parser.add_argument("--aggregation-mode", default="max")
    parser.add_argument("--aggregation-stage", default="patch")
    parser.add_argument("--topk-ratio", type=float, default=0.01)
    parser.add_argument("--reference-topk", type=int, default=3)
    parser.add_argument("--subspace-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def slug(value: str) -> str:
    return value.replace("-", "_").replace(".", "p")


def ratio_tag(value: float) -> str:
    return f"{int(round(value * 1000)):03d}"


def default_experiments(mode: str) -> list[str]:
    if mode == MODE_SCREENING:
        return [EXPERIMENT_BASELINE, EXPERIMENT_SUBSPACE_D8]
    return [EXPERIMENT_BASELINE]


def default_seeds(mode: str) -> list[int]:
    if mode == MODE_SCREENING:
        return [42]
    return [42, 43, 44]


def manifest_filename(category: str, support_normal_k: int, support_defect_k: int, seed: int) -> str:
    return f"{category}_sn{support_normal_k}_sd{support_defect_k}_seed{seed}.json"


def ensure_manifest(
    manifests_dir: Path,
    data_root: Path,
    category: str,
    support_normal_k: int,
    support_defect_k: int,
    seed: int,
) -> Path:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / manifest_filename(
        category=category,
        support_normal_k=support_normal_k,
        support_defect_k=support_defect_k,
        seed=seed,
    )
    if manifest_path.is_file():
        manifest = load_shared_split_manifest(manifest_path)
        if manifest.category != category:
            raise ValueError(f"Manifest category mismatch at {manifest_path}: {manifest.category} != {category}")
        return manifest_path

    manifest = build_shared_split_manifest(
        root=data_root,
        category=category,
        support_normal_k=support_normal_k,
        support_defect_k=support_defect_k,
        seed=seed,
    )
    save_shared_split_manifest(manifest, manifest_path)
    return manifest_path


def experiment_config(args: argparse.Namespace, experiment_name: str) -> dict[str, object]:
    if experiment_name == EXPERIMENT_BASELINE:
        return {"method": "baseline"}
    if experiment_name == EXPERIMENT_SUBSPACE_D8:
        return {"method": "subspace", "subspace_dim": int(args.subspace_dim)}
    raise ValueError(f"Unsupported experiment: {experiment_name}")


def build_run_name(args: argparse.Namespace, experiment_name: str) -> str:
    config = experiment_config(args, experiment_name)
    parts = [
        "a1",
        f"sn{int(args.support_normal_k):03d}",
        f"sd{int(args.support_defect_k):03d}",
        slug(args.feature_layer),
        slug(args.score_mode),
        slug(str(config["method"])),
        f"refk{int(args.reference_topk):03d}",
        slug(args.aggregation_mode),
        slug(args.aggregation_stage),
        f"topk{ratio_tag(args.topk_ratio)}",
    ]
    if config["method"] == "subspace":
        parts.append(f"dim{int(config['subspace_dim']):03d}")
    return "_".join(parts)


def scope_name(mode: str, seed: int) -> str:
    if mode == MODE_SCREENING:
        return f"seed{seed}"
    return f"finalist_seed{seed}"


def build_command(
    args: argparse.Namespace,
    category: str,
    manifest_path: Path,
    seed: int,
    scope_dir: Path,
    experiment_names: list[str],
) -> list[str]:
    configs = [experiment_config(args, name) for name in experiment_names]
    methods: list[str] = []
    subspace_dims: list[int] = []
    for config in configs:
        method = str(config["method"])
        if method not in methods:
            methods.append(method)
        if method == "subspace":
            dim = int(config["subspace_dim"])
            if dim not in subspace_dims:
                subspace_dims.append(dim)
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "eval_structure_ablation.py"),
        "--category",
        category,
        "--split-manifest",
        str(manifest_path),
        "--output-dir",
        str(scope_dir),
        "--cache-root",
        str(args.cache_root),
        "--image-size",
        str(args.image_size),
        "--pretrained",
        str(args.pretrained),
        "--device",
        args.device,
        "--feature-layer",
        args.feature_layer,
        "--score-mode",
        args.score_mode,
        "--aggregation-mode",
        args.aggregation_mode,
        "--aggregation-stage",
        args.aggregation_stage,
        "--topk-ratio",
        str(args.topk_ratio),
        "--reference-topk",
        str(args.reference_topk),
        "--methods",
        *methods,
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
    ]
    if subspace_dims:
        command.extend(["--subspace-dims", *(str(dim) for dim in subspace_dims)])
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def read_metrics(metrics_path: Path) -> dict[str, object] | None:
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.experiments is None:
        args.experiments = default_experiments(args.mode)
    if args.seeds is None:
        args.seeds = default_seeds(args.mode)
    if args.mode == MODE_SCREENING and len(args.seeds) != 1:
        raise ValueError("screening mode expects exactly one seed.")
    if args.mode == MODE_FINALIST and len(args.experiments) != 1:
        raise ValueError("finalist mode expects exactly one experiment.")

    output_root = Path(args.output_dir)
    manifests_dir = Path(args.manifests_dir)
    data_root = Path(args.data_root)
    all_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        current_scope = output_root / scope_name(args.mode, seed)
        current_scope.mkdir(parents=True, exist_ok=True)
        experiment_names = [build_run_name(args, name) for name in args.experiments]
        for category in args.categories:
            manifest_path = ensure_manifest(
                manifests_dir=manifests_dir,
                data_root=data_root,
                category=category,
                support_normal_k=args.support_normal_k,
                support_defect_k=args.support_defect_k,
                seed=seed,
            )
            command = build_command(
                args=args,
                category=category,
                manifest_path=manifest_path,
                seed=seed,
                scope_dir=current_scope,
                experiment_names=args.experiments,
            )
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "seed": seed,
                        "category": category,
                        "experiments": experiment_names,
                        "command": command,
                    }
                )
            )
            completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
            for exp_name in experiment_names:
                expected_metrics_path = current_scope / exp_name / category / "metrics.json"
                metrics = read_metrics(expected_metrics_path)
                row = {
                    "mode": args.mode,
                    "seed": seed,
                    "scope": current_scope.name,
                    "category": category,
                    "experiment": exp_name,
                    "status": "completed" if completed.returncode == 0 and metrics is not None else "failed",
                    "returncode": completed.returncode,
                    "manifest_path": str(manifest_path),
                    "metrics_path": str(expected_metrics_path),
                    "command": json.dumps(command, ensure_ascii=False),
                }
                if metrics is not None:
                    row.update(
                        {
                            "image_auroc": metrics.get("image_auroc"),
                            "pixel_auroc": metrics.get("pixel_auroc"),
                            "pro": metrics.get("pro"),
                        }
                    )
                all_rows.append(row)

    run_manifest = {
        "mode": args.mode,
        "seeds": args.seeds,
        "experiments": [build_run_name(args, name) for name in args.experiments],
        "categories": list(args.categories),
        "rows": all_rows,
    }
    scope_suffix = "__".join(scope_name(args.mode, seed) for seed in args.seeds)
    manifest_stem = f"{args.mode}_{scope_suffix}_run_manifest"
    (output_root / f"{manifest_stem}.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_rows_csv(output_root / f"{manifest_stem}.csv", all_rows)
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "num_rows": len(all_rows),
                "num_failures": sum(1 for row in all_rows if row["status"] != "completed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
