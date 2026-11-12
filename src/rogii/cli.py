from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rogii import __version__


def _predict(args) -> None:
    from rogii import pipeline

    pipeline.run(
        profile=args.profile,
        structural=args.structural,
        data_dir=args.data,
        out_dir=args.out,
        artifacts_dir=args.artifacts,
        fast=args.fast,
        n_jobs=args.jobs,
        train_missing=args.train_missing,
    )


def _audit(args) -> None:
    from rogii import audit, paths

    data = paths.competition_dir(args.data)
    out = paths.work_dir(args.out)
    print(audit.write_audit(out, data / "sample_submission.csv"))


def _blend(args) -> None:
    from rogii import blend
    from rogii.io import read_submission, write_submission

    merged = blend.weighted(read_submission(args.first), read_submission(args.second), args.weight)
    write_submission(merged, Path(args.output))
    print(f"wrote {args.output} ({len(merged)} rows, {args.weight:g}*first + {1 - args.weight:g}*second)")


def _list(args) -> None:
    from rogii.config import profile_names
    from rogii.structural.variants import load_variant, variant_names

    print("profiles:")
    for name in profile_names():
        print(f"  {name}")
    print("structural variants:")
    for name in variant_names():
        print(f"  {name:<8} {load_variant(name).description}")


def _cv(args) -> None:
    from rogii import paths
    from rogii.diagnostics import cv

    data = paths.competition_dir(args.data)
    out = paths.work_dir(args.out)
    cv.run(args.kind, data, out, n_wells=args.wells, seed=args.seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rogii", description="ROGII wellbore TVT prediction")
    parser.add_argument("--version", action="version", version=f"rogii {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="run the full pipeline and write submission.csv")
    p.add_argument("--profile", default="vp_balanced_modelpkg_005")
    p.add_argument("--structural", help="structural blend variant applied after the audit (see `rogii list`)")
    p.add_argument("--data", help="competition folder (default: auto-detect)")
    p.add_argument("--out", help="output folder (default: /kaggle/working or ./outputs)")
    p.add_argument("--artifacts", help="folder holding the artifact datasets (default: ./artifacts)")
    p.add_argument("--jobs", type=int, help="worker threads for feature building")
    p.add_argument("--fast", action="store_true", help="shrink Monte-Carlo budgets; smoke runs only")
    p.add_argument(
        "--train-missing", action="store_true", help="train the ridge stack if its artifacts are absent"
    )
    p.set_defaults(func=_predict)

    p = sub.add_parser("audit", help="validate out/submission.csv against the sample submission")
    p.add_argument("--data")
    p.add_argument("--out")
    p.set_defaults(func=_audit)

    p = sub.add_parser("blend", help="weighted mix of two submission files")
    p.add_argument("first")
    p.add_argument("second")
    p.add_argument("-w", "--weight", type=float, default=0.5, help="weight of the first file")
    p.add_argument("-o", "--output", default="blended.csv")
    p.set_defaults(func=_blend)

    p = sub.add_parser("list", help="show available profiles and structural variants")
    p.set_defaults(func=_list)

    p = sub.add_parser("cv", help="leave-well-out diagnostics on the train wells")
    p.add_argument("kind", choices=["oracle", "selector", "heel", "struct"])
    p.add_argument("--data")
    p.add_argument("--out")
    p.add_argument("--wells", type=int, default=250, help="number of train wells to score")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cv)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
