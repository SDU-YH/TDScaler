from __future__ import annotations

import sys
from pathlib import Path

DTQN_SOURCE_ROOT = Path(__file__).resolve().parent / "vendor"
source_root = str(DTQN_SOURCE_ROOT)
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from dtqn.agents.dtqn import DtqnAgent as ReferenceDtqnAgent
from dtqn.networks.dtqn import DTQN as ReferenceDTQN
from utils.epsilon_anneal import LinearAnneal as ReferenceLinearAnneal
