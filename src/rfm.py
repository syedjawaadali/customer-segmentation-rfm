"""RFM scoring + K-Means segmentation of a customer base.

Two complementary views of the same customers: an interpretable RFM quintile
model that names segments for marketing, and an unsupervised K-Means clustering
validated with a silhouette score. Run after ``generate_data.py``. Outputs land
in ``outputs/``; committed copies for the README live in ``docs/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transactions.csv"
OUT = ROOT / "outputs"
DOCS = ROOT / "docs"

SEG_ORDER = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]

# ---- House style -----------------------------------------------------------
INK = "#0f172a"
GRID = "#e2e8f0"
ACCENT = "#1E90FF"
ACCENT_2 = "#00C2A8"
PALETTE = ["#1E90FF", "#00C2A8", "#F5A524", "#7C5CFF", "#F2647C"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": GRID,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

_MONEY = FuncFormatter(lambda x, _: f"${x/1e3:.0f}K" if abs(x) >= 1e3 else f"${x:.0f}")


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda s: (snapshot - s.max()).days),
        frequency=("order_date", "count"),
        monetary=("amount", "sum"),
    )
    # 1-5 quintile scores (recency reversed: recent = high score)
    rfm["R"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm[["R", "F", "M"]].sum(axis=1)
    return rfm


def label(score: int) -> str:
    if score >= 13:
        return "Champions"
    if score >= 10:
        return "Loyal"
    if score >= 7:
        return "Potential"
    if score >= 5:
        return "At Risk"
    return "Hibernating"


def cluster(rfm: pd.DataFrame, k: int = 4) -> tuple[pd.DataFrame, float]:
    """K-Means on log-scaled, standardised RFM; returns the silhouette score."""
    X = StandardScaler().fit_transform(
        np.log1p(rfm[["recency", "frequency", "monetary"]]))
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["cluster"] = km.fit_predict(X)
    sil = float(silhouette_score(X, rfm["cluster"]))
    return rfm, sil


def metrics(rfm: pd.DataFrame, n_tx: int, sil: float, k: int) -> dict:
    seg_sizes = rfm["segment"].value_counts().reindex(SEG_ORDER).fillna(0).astype(int)
    seg_rev = rfm.groupby("segment")["monetary"].sum().reindex(SEG_ORDER).fillna(0)
    total_rev = float(rfm["monetary"].sum())
    champ_share = round(100 * seg_rev["Champions"] / total_rev, 1)
    return {
        "n_customers": int(len(rfm)),
        "n_transactions": int(n_tx),
        "k_clusters": k,
        "silhouette_score": round(sil, 3),
        "total_monetary": round(total_rev, 2),
        "avg_orders_per_customer": round(float(rfm["frequency"].mean()), 2),
        "champions_pct_of_customers": round(100 * seg_sizes["Champions"] / len(rfm), 1),
        "champions_pct_of_revenue": champ_share,
        "segment_sizes": seg_sizes.to_dict(),
        "revenue_by_segment": {k2: round(float(v), 2) for k2, v in seg_rev.items()},
        "cluster_profiles": {
            int(c): {m: round(float(v), 1) for m, v in row.items()}
            for c, row in rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
            .mean().iterrows()
        },
    }


def dashboard(rfm: pd.DataFrame, m: dict) -> None:
    """One-page segmentation dashboard: sizes, clusters, revenue, profiles."""
    OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Customer Segmentation — RFM + K-Means", fontsize=16,
                 fontweight="bold", color=INK, x=0.5, y=0.99)

    # 1) RFM segment sizes
    ax = axes[0, 0]
    sizes = pd.Series(m["segment_sizes"]).reindex(SEG_ORDER)
    ax.bar(sizes.index, sizes.values, color=PALETTE)
    ax.set_title("Segment Sizes (rule-based RFM)")
    ax.set_ylabel("Customers")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(sizes.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2) K-Means clusters in frequency x monetary space (log)
    ax = axes[0, 1]
    for c in sorted(rfm["cluster"].unique()):
        sub = rfm[rfm["cluster"] == c]
        ax.scatter(sub["frequency"], sub["monetary"], s=14, alpha=0.55,
                   color=PALETTE[c % len(PALETTE)], label=f"Cluster {c}")
    ax.set_yscale("log")
    ax.set_title(f"K-Means Clusters (silhouette = {m['silhouette_score']:.3f})")
    ax.set_xlabel("Frequency (orders)")
    ax.set_ylabel("Monetary (total spend)")
    ax.yaxis.set_major_formatter(_MONEY)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # 3) Revenue by segment
    ax = axes[1, 0]
    rev = pd.Series(m["revenue_by_segment"]).reindex(SEG_ORDER)[::-1]
    ax.barh(rev.index, rev.values, color=ACCENT_2)
    ax.set_title("Revenue by Segment")
    ax.xaxis.set_major_formatter(_MONEY)
    ax.grid(axis="y", visible=False)

    # 4) Cluster profile heatmap (min-max normalised per metric, annotated w/ means)
    ax = axes[1, 1]
    prof = rfm.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    norm = (prof - prof.min()) / (prof.max() - prof.min())
    ax.imshow(norm.values, cmap="Blues", aspect="auto")
    ax.set_title("Cluster Profiles (mean R / F / M)")
    ax.set_xticks(range(3)); ax.set_xticklabels(["Recency", "Frequency", "Monetary"])
    ax.set_yticks(range(len(prof))); ax.set_yticklabels([f"Cluster {c}" for c in prof.index])
    ax.grid(False)
    for i in range(len(prof)):
        for j, col in enumerate(["recency", "frequency", "monetary"]):
            val = prof.iloc[i][col]
            txt = f"{val:,.0f}" if col != "frequency" else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if norm.values[i, j] > 0.55 else INK,
                    fontsize=10, fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for t in (OUT / "dashboard.png", DOCS / "dashboard.png"):
        fig.savefig(t, dpi=130)
    plt.close(fig)

    # Standalone cluster scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in sorted(rfm["cluster"].unique()):
        sub = rfm[rfm["cluster"] == c]
        ax.scatter(sub["recency"], sub["monetary"], s=16, alpha=0.55,
                   color=PALETTE[c % len(PALETTE)], label=f"Cluster {c}")
    ax.set_yscale("log")
    ax.set_title("Customer Clusters — Recency vs Monetary")
    ax.set_xlabel("Recency (days since last order)")
    ax.set_ylabel("Monetary (total spend)")
    ax.yaxis.set_major_formatter(_MONEY)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for t in (OUT / "clusters.png", DOCS / "clusters.png"):
        fig.savefig(t, dpi=120)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python src/generate_data.py first")
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    rfm = build_rfm(df)
    rfm, sil = cluster(rfm)
    rfm["segment"] = rfm["rfm_score"].map(label)

    m = metrics(rfm, n_tx=len(df), sil=sil, k=4)
    dashboard(rfm, m)

    print("=== Segment sizes (rule-based RFM) ===")
    print(rfm["segment"].value_counts().reindex(SEG_ORDER).to_string())
    print(f"\n=== K-Means (k=4) silhouette score: {sil:.3f} ===")
    print("Cluster profiles (mean values):")
    print(rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
          .mean().round(1).to_string())
    print(f"\nChampions: {m['champions_pct_of_customers']}% of customers, "
          f"{m['champions_pct_of_revenue']}% of revenue.")

    OUT.mkdir(exist_ok=True); DOCS.mkdir(exist_ok=True)
    rfm.to_csv(OUT / "rfm_segments.csv")
    rfm.head(10).to_csv(DOCS / "sample_segments.csv")
    for t in (OUT / "metrics.json", DOCS / "metrics.json"):
        t.write_text(json.dumps(m, indent=2))
    print(f"\nSaved segment table -> {OUT / 'rfm_segments.csv'}")
    print(f"Dashboard + clusters + metrics.json written to {OUT}/ and {DOCS}/")


if __name__ == "__main__":
    main()
