"""Bất biến chống leakage: tập bệnh nhân giữa các split phải RỜI NHAU.

Chạy sau make_split.py:  pytest tests/test_leakage.py -q
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import pytest

_CANDIDATES = [
    "/kaggle/working/splits/lits_v1.json",
    "splits/lits_v1.json",
    "data/splits/lits_v1.json",
]


def _load():
    for p in _CANDIDATES:
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    found = glob.glob("**/lits_v1.json", recursive=True)
    if found:
        return json.load(open(found[0], encoding="utf-8"))
    pytest.skip("Không thấy split.json — chạy scripts/make_split.py trước.")


def test_test_disjoint_from_all_folds():
    s = _load()
    test = set(s["test"])
    for i, f in enumerate(s["folds"]):
        assert test.isdisjoint(f["train"]), f"fold {i}: test ∩ train != ∅"
        assert test.isdisjoint(f["val"]), f"fold {i}: test ∩ val != ∅"


def test_train_val_disjoint_each_fold():
    s = _load()
    for i, f in enumerate(s["folds"]):
        assert set(f["train"]).isdisjoint(f["val"]), f"fold {i}: train ∩ val != ∅"


def test_val_folds_pairwise_disjoint():
    s = _load()
    vals = [set(f["val"]) for f in s["folds"]]
    for i, (a, b) in enumerate(itertools.combinations(vals, 2)):
        assert a.isdisjoint(b), "hai fold val trùng bệnh nhân"


def test_no_patient_id_empty():
    s = _load()
    assert len(s["test"]) > 0
    for f in s["folds"]:
        assert f["train"] and f["val"]