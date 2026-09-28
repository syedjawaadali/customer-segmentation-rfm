"""RFM scoring + K-Means segmentation of customers."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"


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


def cluster(rfm: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    X = StandardScaler().fit_transform(
        np.log1p(rfm[["recency", "frequency", "monetary"]]))
    rfm["cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    return rfm


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run: python src/generate_data.py first")
    df = pd.read_csv(DATA, parse_dates=["order_date"])
    rfm = cluster(build_rfm(df))
    rfm["segment"] = rfm["rfm_score"].map(label)

    print("=== Segment sizes (rule-based RFM) ===")
    print(rfm["segment"].value_counts().to_string())
    print("\n=== Cluster profiles (K-Means, mean values) ===")
    print(rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
          .mean().round(1).to_string())
    out = DATA.parent / "outputs"
    out.mkdir(exist_ok=True)
    rfm.to_csv(out / "rfm_segments.csv")
    print(f"\nSaved segment table -> {out / 'rfm_segments.csv'}")


if __name__ == "__main__":
    main()
