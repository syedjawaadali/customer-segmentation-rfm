"""Generate synthetic customer transaction history."""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(23)
DATA = Path(__file__).resolve().parents[1] / "data"


def generate(n_customers: int = 2000) -> pd.DataFrame:
    rows = []
    end = pd.Timestamp("2026-01-01")
    for cid in range(1, n_customers + 1):
        n_orders = RNG.poisson(4) + 1
        for _ in range(n_orders):
            days_ago = int(RNG.exponential(120))
            rows.append((cid, end - pd.Timedelta(days=days_ago),
                         round(RNG.gamma(2.5, 40), 2)))
    return pd.DataFrame(rows, columns=["customer_id", "order_date", "amount"])


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    df = generate()
    out = DATA / "transactions.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} transactions for "
          f"{df.customer_id.nunique():,} customers -> {out}")
