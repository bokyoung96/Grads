from __future__ import annotations

import sys
from pathlib import Path

SP500_DIR = Path(__file__).resolve().parents[2]
REPO_PARENT = SP500_DIR.parent
for cand in (REPO_PARENT, SP500_DIR):
    if str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from SP500.preprocess.univ.factory.build import UnivBuild
from SP500.root import DATA_DIR


def register(
    *,
    monthly_path: Path | None = None,
    whole_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Build sedol daily universe parquet.

    Defaults:
      - monthly_path: latest DATA_DIR/SPY_sedol_monthly_*.csv
      - whole_path:   latest DATA_DIR/SPY_sedol_whole_*.csv
      - out_path:     DATA_DIR/univ/sedol_daily.parquet
    """

    builder = UnivBuild(
        monthly_path=monthly_path,
        whole_path=whole_path,
        out_path=out_path or (DATA_DIR / "univ" / "sedol_daily.parquet"),
    )
    return builder.run()


if __name__ == "__main__":
    print(register())
