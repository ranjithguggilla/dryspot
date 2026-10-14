"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data, score, study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dryspot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch the water point csv into data/raw")

    t = sub.add_parser("train", help="run the validation study and fit the shipped model")
    t.add_argument("--max-rows", type=int, default=None, help="subsample for a quick run")
    t.add_argument("--folds", type=int, default=4)
    t.add_argument("--bootstrap", type=int, default=200)
    t.add_argument("--out", type=Path, default=config.OUTPUT_DIR)

    s = sub.add_parser("score", help="rank the rows of a csv by risk")
    s.add_argument("csv", type=Path)
    s.add_argument("--model", type=Path, default=None)
    s.add_argument("--out", type=Path, default=Path("ranked_water_points.csv"))
    s.add_argument("--visit-share", type=float, default=config.DEFAULT_VISIT_SHARE)

    sub.add_parser("app", help="launch the web app")

    args = ap.parse_args(argv)
    if args.cmd == "download":
        print(data.download())
    elif args.cmd == "train":
        if not config.RAW_FILE.exists():
            print(
                "raw data missing, using the committed sample; `dryspot download` gets the full set"
            )
        study.main(
            max_rows=args.max_rows, n_splits=args.folds, n_boot=args.bootstrap, out_dir=args.out
        )
    elif args.cmd == "score":
        bundle = score.load_bundle(args.model)
        scored = score.triage_split(
            score.score_frame(bundle, pd.read_csv(args.csv)), args.visit_share
        )
        scored.to_csv(args.out, index=False)
        print(f"wrote {len(scored)} rows to {args.out}")
    elif args.cmd == "app":
        from . import app

        app.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
