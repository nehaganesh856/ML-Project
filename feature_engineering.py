import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Extracts time-based fields from the Datetime column.
    """
    df = df.copy()
    if 'Datetime' not in df.columns:
        return df
        
    df['hour'] = df['Datetime'].dt.hour
    df['day'] = df['Datetime'].dt.day
    df['month'] = df['Datetime'].dt.month
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['is_weekend'] = df['Datetime'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Hour category: Morning (6-12), Afternoon (12-17), Evening (17-22), Night (22-6)
    def get_hour_category(h):
        if 6 <= h < 12:
            return 0  # Morning
        elif 12 <= h < 17:
            return 1  # Afternoon
        elif 17 <= h < 22:
            return 2  # Evening
        else:
            return 3  # Night
            
    df['hour_category'] = df['hour'].apply(get_hour_category)
    return df

def create_lag_features(df, target_col='energy_consumption', lags=[1, 2, 3]):
    """
    Creates historical lags for the target column.
    """
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col='energy_consumption', windows=[3, 6]):
    """
    Creates moving averages of the target column.
    """
    df = df.copy()
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'{target_col}_roll_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
    return df

def run_feature_engineering(df, target_col='energy_consumption', is_training=True):
    """
    Executes all feature engineering steps. Fills NaNs created by shift/rolling.
    """
    # 1. Time features
    df = create_time_features(df)
    
    # 2. Lag features
    df = create_lag_features(df, target_col=target_col, lags=[1, 2, 3, 24])
    
    # 3. Rolling features
    df = create_rolling_features(df, target_col=target_col, windows=[3, 6])
    
    # Fill any NaNs created by shift/rolling with the mean or forward/backfill
    # To prevent leakage during test inference, we ffill/bfill row-wise
    feature_cols = [col for col in df.columns if 'lag' in col or 'roll' in col]
    df[feature_cols] = df[feature_cols].bfill().ffill().fillna(0.0)
    
    return df
