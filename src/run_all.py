"""Entry point: reproduces every report and figure in outputs/.

Usage:
    python -m src.run_all [--data-dir PATH] [--out-dir PATH]
"""

import argparse
from pathlib import Path

from . import audit, config, data, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None,
                        help="Folder containing the Kaggle CSVs (default: ./data, then ../data)")
    parser.add_argument("--out-dir", default="outputs",
                        help="Where reports and figures are written (default: outputs)")
    args = parser.parse_args()

    data_dir = config.resolve_data_dir(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {data_dir.resolve()}")
    primary = data.load(data_dir, config.PRIMARY_CSV)
    biased = data.load(data_dir, config.BIASED_CSV)

    print("[1/3] Validating data against its documentation...")
    data.validate(primary, biased, out_dir)

    print("[2/3] Training and evaluating the mid-semester at-risk model (primary file)...")
    model.run(primary, out_dir)

    print("[3/3] Auditing the biased file...")
    audit.run(biased, out_dir)

    print(f"Done. Reports and figures are in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
