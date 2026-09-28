# Customer Segmentation (RFM + K-Means)

Segments a customer base two complementary ways: an interpretable **RFM**
(Recency, Frequency, Monetary) quintile scoring model and an unsupervised
**K-Means** clustering, then labels customers as Champions, Loyal, Potential,
At Risk or Hibernating for targeted marketing.

## Run it
```bash
pip install -r requirements.txt
python src/generate_data.py
python src/rfm.py
```
Outputs a per-customer segment table to `outputs/rfm_segments.csv`.

## Stack
`scikit-learn` (KMeans, StandardScaler) · `pandas` · `numpy`

---
Part of my data analytics portfolio — [github.com/syedjawaadali](https://github.com/syedjawaadali)
