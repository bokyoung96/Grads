from __future__ import annotations

import logging
import sys
from pathlib import Path

PREPROCESS_DIR = Path(__file__).resolve().parent
GRADS_DIR = PREPROCESS_DIR.parent
if str(GRADS_DIR) not in sys.path:
    sys.path.insert(0, str(GRADS_DIR))

from preprocess.base.align import Align
from preprocess.base.load import Load
from preprocess.base.reg import FE
from preprocess.factory.build import Build
from preprocess.factory.data import Data
import preprocess.price.feat_price as feat_price
import preprocess.fundamentals.feat_fund as feat_fund
import preprocess.consensus.feat_cons as feat_cons
import preprocess.sector.feat_sector as feat_sector

_feature_modules = (feat_price, feat_fund, feat_cons, feat_sector)

fe = list(FE)

# NOTE: BM_MODE: "excess" or "none"
BM_MODE = "excess"
BM_TKR = "IKS200"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
loader = Load()
aligner = Align()
builder = Build(fe)

data = Data(loader, aligner, builder)
data.make(bm_mode=BM_MODE, bm_tkr=BM_TKR)
