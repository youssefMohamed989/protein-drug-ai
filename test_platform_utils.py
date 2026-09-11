import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

openmm = pytest.importorskip("openmm", reason="OpenMM not installed")

from src.md.platform_utils import list_available_platforms, resolve_platform


def test_list_available_platforms_includes_cpu():
    available = list_available_platforms()
    assert "CPU" in available  # always available in any OpenMM build


def test_resolve_platform_cpu_case_insensitive():
    platform, props = resolve_platform("cpu")
    assert platform.getName() == "CPU"
    assert props is None


def test_resolve_platform_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="Unknown platform"):
        resolve_platform("NotARealPlatform")


def test_resolve_platform_unavailable_platform_raises_runtime_error():
    available = list_available_platforms()
    if "CUDA" in available:
        pytest.skip("CUDA is actually available on this machine; can't test the unavailable path")
    with pytest.raises(RuntimeError, match="not available"):
        resolve_platform("CUDA")


def test_resolve_platform_rejects_unsupported_property():
    with pytest.raises(ValueError, match="does not support"):
        resolve_platform("CPU", properties={"NotARealProperty": "value"})
