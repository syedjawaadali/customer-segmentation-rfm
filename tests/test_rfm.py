"""Invariant tests for RFM scoring and K-Means segmentation.

Guard the contracts the README relies on: quintile scores stay in range, the
segment labels are ordered correctly, clustering returns the requested number of
groups with a valid silhouette, and the metric aggregates reconcile.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import generate_data  # noqa: E402
import rfm as rfm_mod  # noqa: E402


@pytest.fixture(scope="module")
def tx():
    return generate_data.generate(n_customers=800)


@pytest.fixture(scope="module")
def scored(tx):
    r = rfm_mod.build_rfm(tx)
    r, sil = rfm_mod.cluster(r)
    r["segment"] = r["rfm_score"].map(rfm_mod.label)
    return r, sil


def test_transactions_schema(tx):
    assert {"customer_id", "order_date", "amount"}.issubset(tx.columns)
    assert (tx["amount"] > 0).all()


def test_rfm_quintiles_in_range(scored):
    r, _ = scored
    for col in ("R", "F", "M"):
        assert r[col].between(1, 5).all()
    assert r["rfm_score"].between(3, 15).all()
    assert (r["recency"] >= 0).all()
    assert (r["frequency"] >= 1).all()


def test_label_boundaries():
    assert rfm_mod.label(15) == "Champions"
    assert rfm_mod.label(13) == "Champions"
    assert rfm_mod.label(12) == "Loyal"
    assert rfm_mod.label(7) == "Potential"
    assert rfm_mod.label(5) == "At Risk"
    assert rfm_mod.label(3) == "Hibernating"


def test_labels_are_score_monotonic(scored):
    r, _ = scored
    # Mean RFM score must strictly decrease from Champions to Hibernating.
    means = (r.groupby("segment")["rfm_score"].mean()
             .reindex(rfm_mod.SEG_ORDER).dropna())
    vals = means.tolist()
    assert vals == sorted(vals, reverse=True)
    assert len(set(vals)) == len(vals)   # strictly, no ties


def test_cluster_count_and_silhouette(scored):
    r, sil = scored
    assert r["cluster"].nunique() == 4
    assert -1.0 <= sil <= 1.0
    assert sil > 0            # clusters are structured, not noise


def test_metrics_reconcile(scored):
    r, sil = scored
    m = rfm_mod.metrics(r, n_tx=5000, sil=sil, k=4)
    assert sum(m["segment_sizes"].values()) == m["n_customers"] == len(r)
    assert abs(sum(m["revenue_by_segment"].values()) - m["total_monetary"]) < 1.0
    assert 0 <= m["champions_pct_of_revenue"] <= 100
    assert len(m["cluster_profiles"]) == 4
