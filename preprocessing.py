import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from backend.config import Config

def load_csv(filepath, sample_size=None):
    """
    Load CSV dataset from file, support both semicolon and comma separation.
    Handles '?' strings as missing values.
    Uses memory-efficient chunked loading for large CSV datasets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect delimiter
    delimiter = ','
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if ';' in first_line:
            delimiter = ';'

    expected_cols = [
        'Date', 'Time', 'Global_active_power', 'Global_reactive_power', 
        'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    expected_cols_lower = [c.lower() for c in expected_cols]

    # Validate header first
    header_df = pd.read_csv(filepath, sep=delimiter, nrows=0)
    header_cols_stripped = [col.strip().lower() for col in header_df.columns]
    for col in expected_cols_lower:
        if col not in header_cols_stripped:
            raise KeyError(f"Missing required column: {col}")

    # Read CSV in chunks
    chunk_list = []
    chunk_size = 50000
    rows_loaded = 0
    max_raw_rows = sample_size * 60 if sample_size else None

    for chunk in pd.read_csv(filepath, sep=delimiter, na_values=['?'], low_memory=False, chunksize=chunk_size):
        if chunk.empty:
            continue
            
        # Clean column names
        chunk.columns = chunk.columns.str.strip()
        
        # Case-insensitive column mapping
        col_mapping = {}
        for col in chunk.columns:
            if col.lower() in expected_cols_lower:
                orig_col = expected_cols[expected_cols_lower.index(col.lower())]
                col_mapping[col] = orig_col
        chunk.rename(columns=col_mapping, inplace=True)
        
        # Select only required columns
        chunk = chunk[[col for col in expected_cols if col in chunk.columns]].copy()
        
        # Combine Date and Time
        try:
            chunk['Datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], dayfirst=True, errors='coerce')
            if chunk['Datetime'].isnull().all():
                chunk['Datetime'] = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], errors='coerce')
            
            chunk.dropna(subset=['Datetime'], inplace=True)
        except Exception as e:
            raise ValueError(f"Error parsing date/time columns: {str(e)}")
            
        chunk_list.append(chunk)
        rows_loaded += len(chunk)
        if max_raw_rows and rows_loaded >= max_raw_rows:
            break

    if not chunk_list:
        raise ValueError("The uploaded CSV file is empty or has no valid rows.")
        
    df = pd.concat(chunk_list, ignore_index=True)
    
    # Sort chronologically
    df.sort_values(by='Datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Optional Sampling for efficiency
    if sample_size and sample_size < len(df):
        df = df.iloc[:sample_size].copy()
        
    return df

def validate_data(df):
    """
    Validates required columns and checks for data quality.
    """
    required_cols = [
        'Datetime', 'Global_active_power', 'Global_reactive_power', 
        'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing required column: {col}"
    
    if len(df) == 0:
        return False, "DataFrame has 0 records."
        
    return True, "Data is valid."

def handle_missing_values(df):
    """
    Fills numeric missing values using forward-fill, then backward-fill.
    Drops any remaining rows that cannot be filled.
    """
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    original_rows = len(df)
    
    # Cast to numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Forward fill then backward fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # Drop rows if they still have NaN in numeric cols
    df.dropna(subset=numeric_cols, inplace=True)
    removed_rows = original_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows due to unfillable NaN values.")
        
    return df

def handle_invalid_values(df):
    """
    Removes invalid negative values and clips voltage to normal household range (200V - 250V).
    """
    numeric_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage', 
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    
    # Replace negative values with 0
    for col in numeric_cols:
        if col != 'Voltage':
            df.loc[df[col] < 0, col] = 0.0

    # Clip voltage to reasonable range (200-250V)
    df['Voltage'] = df['Voltage'].clip(lower=200.0, upper=250.0)
    
    return df

def calculate_energy_consumption(df):
    """
    Computes energy consumption in kWh.
    If data is minutely: Active Power (kW) * (1 / 60) hours = energy in kWh.
    """
    # Global_active_power in the UCI dataset is already in kilowatts (kW)
    # So for 1-minute intervals, energy (kWh) = power (kW) * 1/60 hour.
    df['energy_consumption'] = df['Global_active_power'] * (1.0 / 60.0)
    return df

def calculate_carbon_emission(df, emission_factor=0.5):
    """
    Calculates carbon emissions: energy_consumption (kWh) * emission_factor (kg CO2 per kWh).
    """
    if 'energy_consumption' not in df.columns:
        df = calculate_energy_consumption(df)
    df['carbon_emission'] = df['energy_consumption'] * emission_factor
    return df

def aggregate_data(df, freq='h'):
    """
    Resample the minutely dataset by frequency (default Hourly 'H').
    - Power values are averaged over the hour, then total energy is sum of sub-minute energies.
    - Voltage is averaged.
    - Carbon and energy consumption are summed.
    """
    df = df.copy()
    df.set_index('Datetime', inplace=True)
    
    agg_rules = {
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'mean', # sub-metering raw values are in Wh active energy, average works as power indicator
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean',
        'energy_consumption': 'sum',
        'carbon_emission': 'sum'
    }
    
    # Filter only available columns
    rules = {col: agg_rules[col] for col in df.columns if col in agg_rules}
    
    df_resampled = df.resample(freq).agg(rules)
    df_resampled.dropna(inplace=True) # Remove hours with no records
    df_resampled.reset_index(inplace=True)
    
    return df_resampled

def normalize_features(df, exclude_cols=['Datetime', 'Date', 'Time'], name='default'):
    """
    Normalizes numeric columns to 0-1 scale.
    Saves scaler file for future inference.
    """
    df = df.copy()
    features = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Save the scaler
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    scaler_path = os.path.join(Config.MODEL_SAVE_PATH, 'scalers', f'{name}_scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    return df, scaler

def split_train_test(df, test_size=0.2, time_series=True):
    """
    Splits the dataframe into train and test sets.
    If time_series=True, splits chronologically (first 80% train, last 20% test).
    Otherwise splits randomly.
    """
    if time_series:
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        # Shuffle randomly
        shuffled = df.sample(frac=1, random_state=42).copy()
        split_idx = int(len(shuffled) * (1 - test_size))
        train_df = shuffled.iloc[:split_idx].copy()
        test_df = shuffled.iloc[split_idx:].copy()
        
    return train_df, test_df

def prepare_lstm_sequences(df, feature_cols, target_col='energy_consumption', sequence_length=24):
    """
    Prepares sequences of historical records to train LSTM.
    For sequence_length L: inputs X are shapes (N, L, F), outputs y are shape (N, 1).
    """
    X, y, timestamps = [], [], []
    
    data_feat = df[feature_cols].values
    data_target = df[target_col].values
    time_values = df['Datetime'].values if 'Datetime' in df.columns else np.arange(len(df))
    
    for i in range(len(df) - sequence_length):
        X.append(data_feat[i : i + sequence_length])
        y.append(data_target[i + sequence_length])
        timestamps.append(time_values[i + sequence_length])
        
    X_arr, y_arr, t_arr = np.array(X), np.array(y), np.array(timestamps)
    if len(X_arr) == 0:
        raise ValueError("Not enough sequential data to create LSTM training sequences.")
    return X_arr, y_arr, t_arr
