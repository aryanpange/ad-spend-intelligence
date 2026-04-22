import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

np.random.seed(42)

CHANNEL_CONFIG = {
    'Google Search': {
        'daily_spend': (4000, 8000),
        'ctr': (0.04, 0.08),
        'cvr': (0.03, 0.06),
        'cpc_base': 45,
        'impression_multiplier': 22
    },
    'Meta Ads': {
        'daily_spend': (3000, 6000),
        'ctr': (0.015, 0.04),
        'cvr': (0.015, 0.035),
        'cpc_base': 28,
        'impression_multiplier': 55
    },
    'Programmatic Display': {
        'daily_spend': (2000, 4000),
        'ctr': (0.003, 0.008),
        'cvr': (0.005, 0.015),
        'cpc_base': 12,
        'impression_multiplier': 280
    },
    'YouTube': {
        'daily_spend': (1500, 3500),
        'ctr': (0.008, 0.02),
        'cvr': (0.01, 0.025),
        'cpc_base': 20,
        'impression_multiplier': 90
    },
    'Affiliate': {
        'daily_spend': (1000, 2500),
        'ctr': (0.02, 0.05),
        'cvr': (0.04, 0.08),
        'cpc_base': 0,
        'impression_multiplier': 40
    }
}

CHANNELS = list(CHANNEL_CONFIG.keys())
START_DATE = datetime(2023, 1, 1)
NUM_DAYS = 548
AVG_ORDER_VALUE = (800, 2500)


def get_seasonality_multiplier(date):
    seasonality_map = {
        1: 0.80,
        2: 0.82,
        3: 0.95,
        4: 0.98,
        5: 1.00,
        6: 0.93,
        7: 0.90,
        8: 0.95,
        9: 1.15,
        10: 1.35,
        11: 1.60,
        12: 1.40,
    }
    return seasonality_map[date.month]


def get_day_of_week_multiplier(date):
    if date.weekday() == 6:
        return 0.70
    elif date.weekday() == 5:
        return 0.78
    else:
        return 1.00


def inject_anomaly(spend, channel, date, anomaly_log):
    anomaly_type = None
    multiplier = 1.0
    rand = np.random.random()

    if rand < 0.010:
        multiplier = np.random.uniform(2.5, 4.0)
        anomaly_type = 'Budget Spike'
    elif rand < 0.018:
        multiplier = np.random.uniform(0.05, 0.20)
        anomaly_type = 'Budget Crash'
    elif rand < 0.023:
        multiplier = 0.0
        anomaly_type = 'Zero Traffic'

    adjusted_spend = spend * multiplier

    if anomaly_type:
        anomaly_log.append({
            'date': date.strftime('%Y-%m-%d'),
            'channel': channel,
            'anomaly_type': anomaly_type,
            'original_spend': round(spend, 2),
            'actual_spend': round(adjusted_spend, 2),
            'multiplier': round(multiplier, 2)
        })

    return adjusted_spend, anomaly_type


def generate_channel_day_record(date, channel, anomaly_log):
    cfg = CHANNEL_CONFIG[channel]

    base_spend = np.random.uniform(*cfg['daily_spend'])
    seasonal_factor = get_seasonality_multiplier(date)
    dow_factor = get_day_of_week_multiplier(date)

    spend = base_spend * seasonal_factor * dow_factor
    spend *= np.random.normal(loc=1.0, scale=0.05)
    spend = max(spend, 0)

    spend, anomaly_type = inject_anomaly(spend, channel, date, anomaly_log)

    if channel == 'Affiliate':
        impressions = int(spend * cfg['impression_multiplier'])
    else:
        impressions = int((spend / cfg['cpc_base']) * cfg['impression_multiplier']) \
                      if cfg['cpc_base'] > 0 else int(spend * 80)

    impressions = max(impressions, 0)

    ctr = np.random.uniform(*cfg['ctr'])

    if anomaly_type in ['Budget Crash', 'Zero Traffic']:
        ctr *= 0.1

    clicks = int(impressions * ctr)
    clicks = min(clicks, impressions)
    clicks = max(clicks, 0)

    cvr = np.random.uniform(*cfg['cvr'])
    conversions = int(clicks * cvr)
    conversions = max(conversions, 0)

    if conversions > 0:
        avg_order = np.random.uniform(*AVG_ORDER_VALUE)
        revenue = conversions * avg_order
    else:
        revenue = 0.0

    actual_ctr = (clicks / impressions) if impressions > 0 else 0
    actual_cvr = (conversions / clicks) if clicks > 0 else 0
    cpc = (spend / clicks) if clicks > 0 else 0
    roas = (revenue / spend) if spend > 0 else 0
    cost_per_conv = (spend / conversions) if conversions > 0 else 0

    return {
        'date': date.strftime('%Y-%m-%d'),
        'channel': channel,
        'spend': round(spend, 2),
        'impressions': impressions,
        'clicks': clicks,
        'conversions': conversions,
        'revenue': round(revenue, 2),
        'ctr': round(actual_ctr, 5),
        'cvr': round(actual_cvr, 5),
        'cpc': round(cpc, 2),
        'roas': round(roas, 3),
        'cost_per_conversion': round(cost_per_conv, 2),
        'seasonality_factor': round(seasonal_factor, 3),
        'day_of_week': date.strftime('%A'),
        'month': date.month,
        'year': date.year,
        'week_number': date.isocalendar()[1],
        'is_weekend': int(date.weekday() >= 5),
        'anomaly_type': anomaly_type if anomaly_type else 'None'
    }


def generate_dataset():
    print("Starting data simulation...")
    print(f"Date range: {START_DATE.strftime('%d %b %Y')} to "
          f"{(START_DATE + timedelta(days=NUM_DAYS-1)).strftime('%d %b %Y')}")
    print(f"Channels: {', '.join(CHANNELS)}")
    print(f"Expected rows: {NUM_DAYS * len(CHANNELS):,}")
    print("-" * 50)

    records = []
    anomaly_log = []

    for day_offset in range(NUM_DAYS):
        date = START_DATE + timedelta(days=day_offset)

        for channel in CHANNELS:
            record = generate_channel_day_record(date, channel, anomaly_log)
            records.append(record)

        if day_offset % 90 == 0:
            print(f"  Generated through {date.strftime('%b %Y')}...")

    df = pd.DataFrame(records)
    anomaly_df = pd.DataFrame(anomaly_log)

    df['date'] = pd.to_datetime(df['date'])
    df['spend'] = df['spend'].astype(float)
    df['impressions'] = df['impressions'].astype(int)
    df['clicks'] = df['clicks'].astype(int)
    df['conversions'] = df['conversions'].astype(int)
    df['revenue'] = df['revenue'].astype(float)
    df['is_weekend'] = df['is_weekend'].astype(int)

    print("\nRunning data validation checks...")

    issues = []

    invalid_ctr = df[df['clicks'] > df['impressions']]
    if len(invalid_ctr) > 0:
        issues.append(f"  FAIL: {len(invalid_ctr)} rows where clicks > impressions")
    else:
        print("  PASS: No rows where clicks > impressions")

    neg_spend = df[df['spend'] < 0]
    if len(neg_spend) > 0:
        issues.append(f"  FAIL: {len(neg_spend)} rows with negative spend")
    else:
        print("  PASS: No negative spend values")

    invalid_cvr = df[df['conversions'] > df['clicks']]
    if len(invalid_cvr) > 0:
        issues.append(f"  FAIL: {len(invalid_cvr)} rows where conversions > clicks")
    else:
        print("  PASS: No rows where conversions > clicks")

    daily_channel_count = df.groupby('date')['channel'].count()
    if (daily_channel_count != len(CHANNELS)).any():
        issues.append("  FAIL: Some dates missing channel records")
    else:
        print(f"  PASS: All {len(CHANNELS)} channels present for all {NUM_DAYS} days")

    extreme_roas = df[df['roas'] > 50]
    print(f"  INFO: {len(extreme_roas)} rows with ROAS > 50x (review these)")

    if issues:
        print("\nValidation issues found:")
        for issue in issues:
            print(issue)
    else:
        print("\nAll validation checks passed.")

    return df, anomaly_df


def save_outputs(df, anomaly_df):
    os.makedirs('data', exist_ok=True)

    df.to_csv('data/campaign_data.csv', index=False)
    print(f"\nSaved: data/campaign_data.csv ({len(df):,} rows, {len(df.columns)} columns)")

    anomaly_df.to_csv('data/ground_truth_anomalies.csv', index=False)
    print(f"Saved: data/ground_truth_anomalies.csv ({len(anomaly_df):,} anomalies logged)")

    summary = {
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'total_rows': len(df),
        'date_range_start': df['date'].min().strftime('%Y-%m-%d'),
        'date_range_end': df['date'].max().strftime('%Y-%m-%d'),
        'channels': CHANNELS,
        'total_spend_inr': round(df['spend'].sum(), 2),
        'total_revenue_inr': round(df['revenue'].sum(), 2),
        'total_conversions': int(df['conversions'].sum()),
        'anomalies_injected': len(anomaly_df),
        'anomaly_rate_pct': round(len(anomaly_df) / len(df) * 100, 2)
    }

    with open('data/dataset_metadata.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved: data/dataset_metadata.json")

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"  Total rows:          {summary['total_rows']:,}")
    print(f"  Date range:          {summary['date_range_start']} to {summary['date_range_end']}")
    print(f"  Total spend (INR):   Rs.{summary['total_spend_inr']:,.0f}")
    print(f"  Total revenue (INR): Rs.{summary['total_revenue_inr']:,.0f}")
    print(f"  Total conversions:   {summary['total_conversions']:,}")
    print(f"  Blended ROAS:        {summary['total_revenue_inr'] / summary['total_spend_inr']:.2f}x")
    print(f"  Anomalies injected:  {summary['anomalies_injected']} ({summary['anomaly_rate_pct']}%)")
    print("=" * 50)

    return summary


def print_channel_snapshot(df):
    print("\nCHANNEL PERFORMANCE SNAPSHOT")
    print("-" * 70)

    snapshot = df.groupby('channel').agg(
        Total_Spend=('spend', 'sum'),
        Total_Revenue=('revenue', 'sum'),
        Conversions=('conversions', 'sum'),
        Avg_ROAS=('roas', 'mean'),
        Avg_CPC=('cpc', 'mean')
    ).round(2)

    snapshot['Cost_Per_Conv'] = (snapshot['Total_Spend'] / snapshot['Conversions']).round(0)
    snapshot['Total_Spend'] = snapshot['Total_Spend'].apply(lambda x: f"Rs.{x:,.0f}")
    snapshot['Total_Revenue'] = snapshot['Total_Revenue'].apply(lambda x: f"Rs.{x:,.0f}")

    print(snapshot.sort_values('Avg_ROAS', ascending=False).to_string())
    print("\nRead this table top to bottom:")
    print("  -> Which channel earns the most per rupee spent?")
    print("  -> Which is burning budget with low ROAS?")


if __name__ == '__main__':
    df, anomaly_df = generate_dataset()
    summary = save_outputs(df, anomaly_df)
    print_channel_snapshot(df)
    print("\nPhase 1 complete.")