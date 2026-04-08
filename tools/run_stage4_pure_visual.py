import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fewshot.data import build_shared_split_manifest, load_shared_split_manifest, save_shared_split_manifest


DEFAULT_WEAK5_CATEGORIES = ("bottle", "carpet", "grid", "leather", "screw", "zipper")
DEFAULT_FULL15_CATEGORIES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage4 pure-visual structure screening/finalist evaluations.")
    parser.add_argument("--mode", choices=(MODE_SCREENING, MODE_FINALIST), default=MODE_SCREENING)
    parser.add_argument("--subset", choices=("weak5_bottle", "full15"), required=True)
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--manifests-dir", default="outputs/split_manifests/stage4_pure_visual")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--categories", nargs="+")
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
    parser.add_argument("--methods", nargs="+", default=["baseline", "fastref_lite", "subspace"])
    parser.add_argument("--coreset-ratios", nargs="+", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--fastref-select-ratios", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--fastref-blend-alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--fastref-steps", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--match-ks", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--spatial-windows", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--subspace-dims", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def default_categories(subset: str) -> tuple[str, ...]:
    if subset == "weak5_bottle":
        return DEFAULT_WEAK5_CATEGORIES
    return DEFAULT_FULL15_CATEGORIES


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
) -> list[str]:
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
        *args.methods,
        "--fastref-select-ratios",
        *(str(value) for value in args.fastref_select_ratios),
        "--coreset-ratios",
        *(str(value) for value in args.coreset_ratios),
        "--fastref-blend-alphas",
        *(str(value) for value in args.fastref_blend_alphas),
        "--fastref-steps",
        *(str(value) for value in args.fastref_steps),
        "--match-ks",
        *(str(value) for value in args.match_ks),
        "--spatial-windows",
        *(str(value) for value in args.spatial_windows),
        "--subspace-dims",
        *(str(value) for value in args.subspace_dims),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def main() -> None:
    args = parse_args()
    if args.categories is None:
        args.categories = list(default_categories(args.subset))
    if args.seeds is None:
        args.seeds = default_seeds(args.mode)
    if args.mode == MODE_SCREENING and len(args.seeds) != 1:
        raise ValueError("screening mode expects exactly one seed.")
    if args.mode == MODE_FINALIST and len(args.seeds) < 2:
        raise ValueError("finalist mode expects multiple seeds.")

    output_root = Path(args.output_dir)
    manifests_dir = Path(args.manifests_dir)
    data_root = Path(args.data_root)
    completed_runs: list[dict[str, object]] = []

    for seed in args.seeds:
        current_scope = output_root / scope_name(args.mode, seed)
        current_scope.mkdir(parents=True, exist_ok=True)
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
            )
            completed_runs.append(
                {
                    "scope": current_scope.name,
                    "seed": seed,
                    "category": category,
                    "command": command,
                }
            )
            subprocess.run(command, check=True, cwd=REPO_ROOT)

    aggregate_output = output_root / ("summary_seed42" if args.mode == MODE_SCREENING else "summary_finalist")
    aggregate_inputs = [str(output_root / scope_name(args.mode, seed)) for seed in args.seeds]
    aggregate_command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "aggregate_structure_ablation.py"),
        "--input-dirs",
        *aggregate_inputs,
        "--output-dir",
        str(aggregate_output),
    ]
    subprocess.run(aggregate_command, check=True, cwd=REPO_ROOT)

    payload = {
        "mode": args.mode,
        "subset": args.subset,
        "output_dir": str(output_root),
        "aggregate_output": str(aggregate_output),
        "num_completed_runs": len(completed_runs),
        "completed_runs": completed_runs,
        "aggregate_command": aggregate_command,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
