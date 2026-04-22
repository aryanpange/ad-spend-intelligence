# AdIntel — Ad Spend Intelligence Platform

A production-style analytics dashboard built in Python/Streamlit for 
multi-channel digital advertising intelligence.

**Live demo → [adintel.streamlit.app]([https://your-url.streamlit.app](https://ad-spend-intelligence.streamlit.app/))**

## What it does
- **Executive Overview** — KPI scorecards, revenue/spend trends, blended ROAS
- **Attribution Analysis** — 5 attribution models (First Touch, Last Touch, 
  Linear, Time Decay, Position Based) with misattribution gap quantification
- **Anomaly Monitor** — Isolation Forest ML model detects budget spikes, 
  CVR collapses, CPC anomalies across 5 channels in real time
- **Budget Pacing** — Monthly spend vs target with channel-level drill-down

## Tech stack
Python · Streamlit · Pandas · Plotly · Scikit-learn · Isolation Forest

## Dataset
18 months · 5 channels · 2,740 rows · ₹12Cr+ total spend simulated

## Key analytical methods
- Unsupervised anomaly detection (Isolation Forest, contamination tuning)
- Multi-touch attribution modeling & misattribution gap analysis  
- Rolling window feature engineering (7-day, 14-day)
- Budget pacing with seasonality-adjusted targets
