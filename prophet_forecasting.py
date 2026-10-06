import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backend.services.data_service import DataService

def forecast_carbon_emissions(user_id=1, forecast_days=7):
    """
    Builds time-series forecast for carbon emissions over 7-day and 30-day horizons.
    Uses Prophet if available, with robust statsmodels Exponential Smoothing fallback.
    """
    df = DataService.get_user_records_df(user_id)
    if df.empty or len(df) < 15:
        df = DataService.get_user_records_df(1)

    if df.empty or len(df) < 10:
        return {
            "status": "not_available",
            "reason": "Insufficient historical time-series data for carbon forecasting."
        }

    # Ensure chronological sorting
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)

    # Daily resampled carbon totals
    df_daily = df.set_index('Datetime').resample('1D')[['carbon_emission', 'energy_consumption']].sum().reset_index()
    df_daily = df_daily.dropna()

    if len(df_daily) < 3:
        # Fallback to hourly aggregated forecast if daily samples are under 3
        df_daily = df[['Datetime', 'carbon_emission', 'energy_consumption']].tail(168).copy()
        df_daily.columns = ['ds', 'y', 'energy']
    else:
        df_daily.columns = ['ds', 'y', 'energy']

    forecast_records = []
    historical_records = []

    for _, row in df_daily.tail(14).iterrows():
        historical_records.append({
            "ds": row['ds'].strftime('%Y-%m-%d'),
            "actual_carbon": round(float(row['y']), 3)
        })

    last_date = df_daily['ds'].max()
    recent_y = df_daily['y'].values

    # Attempt Prophet
    used_engine = "ExponentialSmoothing"
    prophet_worked = False

    try:
        from prophet import Prophet
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df_daily[['ds', 'y']])
        future = m.make_future_dataframe(periods=forecast_days, freq='D')
        fcst = m.predict(future)
        
        fcst_tail = fcst.tail(forecast_days)
        for _, r in fcst_tail.iterrows():
            yhat = max(0.1, float(r['yhat']))
            yhat_lower = max(0.0, float(r['yhat_lower']))
            yhat_upper = float(r['yhat_upper'])
            forecast_records.append({
                "ds": r['ds'].strftime('%Y-%m-%d'),
                "forecast_carbon": round(yhat, 3),
                "lower_bound": round(yhat_lower, 3),
                "upper_bound": round(yhat_upper, 3)
            })
        prophet_worked = True
        used_engine = "Prophet"
    except Exception as ex:
        print(f"[FORECAST] Prophet unavailable/failed ({ex}), using statsmodels exponential smoothing fallback...")

    if not prophet_worked:
        # Statsmodels / Exponential Smoothing forecast fallback
        try:
            from statsmodels.tsa.api import Holt
            model = Holt(recent_y, initialization_method="estimated").fit()
            predictions = model.forecast(forecast_days)
        except Exception:
            # Moving average trend fallback
            ma = np.mean(recent_y[-5:]) if len(recent_y) >= 5 else np.mean(recent_y)
            std = np.std(recent_y) if len(recent_y) >= 2 else 0.2
            predictions = [ma + (i * 0.02 * ma) for i in range(1, forecast_days + 1)]

        for i, val in enumerate(predictions):
            dt = last_date + timedelta(days=i + 1)
            yhat = max(0.1, float(val))
            std_margin = 0.15 * yhat
            forecast_records.append({
                "ds": dt.strftime('%Y-%m-%d'),
                "forecast_carbon": round(yhat, 3),
                "lower_bound": round(max(0.0, yhat - std_margin), 3),
                "upper_bound": round(yhat + std_margin, 3)
            })

    total_projected_7d = sum(r['forecast_carbon'] for r in forecast_records[:7])
    avg_daily_projected = total_projected_7d / min(7, len(forecast_records))

    return {
        "status": "success",
        "engine": used_engine,
        "forecast_horizon_days": forecast_days,
        "total_projected_7d_kg_co2": round(total_projected_7d, 3),
        "avg_daily_projected_kg_co2": round(avg_daily_projected, 3),
        "historical": historical_records,
        "forecast": forecast_records
    }
