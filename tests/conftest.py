import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def stopping():
    df = pd.read_csv(DATA / "stopping_distance_60_values.csv")
    df.columns = ["obs", "v_kmh", "v_ms", "d_theo", "grade_factor", "d_sim", "sigma", "split"]
    return df


@pytest.fixture(scope="session")
def vapour():
    df = pd.read_csv(DATA / "vapour_pressure_60_values.csv", encoding="latin-1")
    df.columns = ["obs", "T_C", "T_K", "P_ref", "P_sim", "sigma", "split"]
    return df
