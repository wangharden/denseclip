import argparse
import json
from pathlib import Path

from fewshot.data import build_shared_split_manifest, save_shared_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a shared split manifest for Stage A1/A2/B comparisons.")
    parser.add_argument("--data-root", default="data/mvtec_anomaly_detection")
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--support-normal-k", type=int, default=8)
    parser.add_argument("--support-defect-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = build_shared_split_manifest(
        root=args.data_root,
        category=args.category,
        support_normal_k=args.support_normal_k,
        support_defect_k=args.support_defect_k,
        seed=args.seed,
    )
    output_path = Path(args.output) if args.output else Path("outputs") / "split_manifests" / f"{args.category}.json"
    save_shared_split_manifest(
        manifest=split,
        path=output_path,
        metadata={
            "data_root": args.data_root,
            "seed": args.seed,
            "requested_support_normal_k": args.support_normal_k,
            "requested_support_defect_k": args.support_defect_k,
        },
    )
    summary = {
        "manifest_path": str(output_path.resolve()),
        "category": split.category,
        "num_support_normal": len(split.support_normal),
        "num_support_defect": len(split.support_defect),
        "num_query_eval": len(split.query_eval),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
