"""Shared setup. app.py runs the whole Streamlit page when imported, so the tests
execute only the part above its UI section, which is all functions and imports."""

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def app():
    source = (ROOT / "app.py").read_text().split("# 6. STREAMLIT APPLICATION INTERFACE")[0]
    namespace = {"__file__": str(ROOT / "app.py"), "__name__": "app_functions"}
    exec(compile(source, "app.py", "exec"), namespace)
    return namespace
