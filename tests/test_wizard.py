"""The wizard's table and its scaling to other GPU sizes (no GPU, standard library).

    python3 -m pytest -q tests/test_wizard.py
"""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "wizard", Path(__file__).resolve().parent.parent / "scripts" / "wizard.py")
wizard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wizard)


def test_table_is_keyed_by_gpu_size():
    tables = wizard.load_table()
    assert wizard.REF_GPU_GIB in tables
    rows = tables[wizard.REF_GPU_GIB]
    assert rows[("gpu", "dspark", True, 8, 2048)] == 267520


def test_unmeasured_size_falls_back_to_reference():
    tables = {24: {"a": 1}}
    assert wizard.pick_table(tables, 22.0) == ({"a": 1}, 24)
    tables[22] = {"b": 2}
    assert wizard.pick_table(tables, 22.0) == ({"b": 2}, 22)


def test_memory_shift():
    assert wizard.memory_shift(24, 24, "dspark") == 0
    # 2 GiB less at 0.95 utilization, 18 KiB per token with DSpark.
    assert wizard.memory_shift(22, 24, "dspark") == -int(2 * 0.95 * 2**20 / 18)
    assert wizard.memory_shift(22, 24, "off") < wizard.memory_shift(22, 24, "dspark")


def test_turing_estimate_lands_on_the_measured_length():
    tables = wizard.load_table()
    key, gib, measured = wizard.TURING_MEASURED
    rows, ref = wizard.pick_table(tables, gib)
    estimate = wizard.expected(rows, *key) + wizard.memory_shift(gib, ref, key[1])
    assert estimate + wizard.turing_shift(tables) == measured
