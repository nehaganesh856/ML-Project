import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from backend.config import Config

# Set TF logging to minimum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape, output_steps=1):
    """
    Build a Sequential LSTM model with specified layers.
    Note: Standard output_steps is 1 for next-step prediction, 
    but we can configure it for multi-step if needed.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(25, return_sequences=False),
        Dropout(0.2),
        Dense(output_steps)
    ])
    
    # Compile with Adam and MSE loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, user_id=None):
    """
    Train the LSTM model and save it as lstm.keras.
    """
    try:
        print("[LSTM] Training started")
        print(f"[LSTM] Dataset records: train={len(X_train)}, val={len(X_val)}")
        print(f"[LSTM] Sequence length: {X_train.shape[1]}")
        print(f"[LSTM] Features: {X_train.shape[2]}")
        
        # Cast to float32
        X_train = X_train.astype("float32")
        X_val = X_val.astype("float32")
        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")
        
        print(f"[LSTM] X shape: {X_train.shape}")
        print(f"[LSTM] y shape: {y_train.shape}")
        print(f"[LSTM] Train shape: {X_train.shape}")
        print(f"[LSTM] X_train dtype: {X_train.dtype}")
        print(f"[LSTM] y_train dtype: {y_train.dtype}")

        # Check for NaN / Inf
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isnan(X_val)) or np.any(np.isnan(y_val)):
            print("[LSTM] Warning: NaN values found in input data. Cleaning...")
            X_train = np.nan_to_num(X_train, nan=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0)
            
        if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)) or np.any(np.isinf(X_val)) or np.any(np.isinf(y_val)):
            print("[LSTM] Warning: Inf values found in input data. Cleaning...")
            X_train = np.nan_to_num(X_train, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, posinf=0.0, neginf=0.0)
            y_val = np.nan_to_num(y_val, posinf=0.0, neginf=0.0)

        # Build model
        print("[LSTM] Building model...")
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape, output_steps=1)
        
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True
        )
        
        # Custom Epoch Logging Callback
        class EpochLoggingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                val_loss_str = f", val_loss: {logs.get('val_loss'):.4f}" if logs and 'val_loss' in logs else ""
                print(f"[LSTM] Epoch completed: epoch {epoch + 1}, loss: {logs.get('loss'):.4f}{val_loss_str}")

        print("[LSTM] Starting model.fit()...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, EpochLoggingCallback()],
            verbose=0
        )
        
        # Save model in .keras format
        print("[LSTM] Saving model...")
        save_dir = os.path.join(Config.MODEL_SAVE_PATH, f"user_{user_id}") if user_id else Config.MODEL_SAVE_PATH
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'lstm.keras')
        model.save(model_path)
        
        # Load verification
        print("[LSTM] Verifying saved model by loading it...")
        loaded_model = load_model(model_path)
        print("[LSTM] Model loaded successfully.")
        
        print("[LSTM] Training completed")
        return model, history
    except Exception as e:
        print(f"Error training LSTM: {e}")
        raise e

def evaluate_lstm(model, X_test, y_test, scaler=None):
    """
    Evaluate the LSTM model and return metrics.
    If scaler is provided, inverse-transforms prediction and target before evaluation.
    """
    y_pred = model.predict(X_test, verbose=0)
    
    if scaler is not None:
        num_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 7
        y_test_inv = []
        y_pred_inv = []
        
        for vt, vp in zip(y_test, y_pred):
            val_t = vt[0] if hasattr(vt, '__len__') else vt
            val_p = vp[0] if hasattr(vp, '__len__') else vp
            
            dummy_t = np.zeros((1, num_features))
            dummy_t[0, 0] = val_t
            inv_t = scaler.inverse_transform(dummy_t)[0, 0]
            y_test_inv.append(inv_t)
            
            dummy_p = np.zeros((1, num_features))
            dummy_p[0, 0] = val_p
            inv_p = scaler.inverse_transform(dummy_p)[0, 0]
            y_pred_inv.append(inv_p)
            
        y_test_eval = np.array(y_test_inv)
        y_pred_eval = np.array(y_pred_inv)
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred
        
    mae = mean_absolute_error(y_test_eval, y_pred_eval)
    mse = mean_squared_error(y_test_eval, y_pred_eval)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_eval, y_pred_eval)
    try:
        mape = mean_absolute_percentage_error(y_test_eval, y_pred_eval) * 100
    except Exception:
        mape = 0.0
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape)
    }

def forecast_lstm(model, X_input, steps=24):
    """
    Generates a multi-step forecast recursively.
    X_input shape must be (1, sequence_length, num_features).
    Assumes that the target is one of the features, and we update it recursively.
    If multiple features are used, recursively forecasting them requires assuming
    other features remain static or predicting them.
    To keep it simple:
    For single step prediction, we just do model.predict(X_input).
    For multi-step recursive forecasting of energy consumption:
    We take the last sequence, predict the next value, shift the sequence, 
    and overwrite the target feature in the sequence with our prediction, then repeat.
    """
    forecast_results = []
    current_seq = X_input.copy() # shape (1, seq_len, num_features)
    
    for _ in range(steps):
        # Predict next value: output shape is (1, 1)
        pred = model.predict(current_seq, verbose=0)[0][0]
        forecast_results.append(float(pred))
        
        # Shift the sequence: move everything left by 1 step
        new_row = current_seq[0, -1, :].copy() # Copy last row features
        # Update the target feature index (assumed to be 0 or energy_consumption)
        # To make it simple, we assume the first feature is the target energy consumption
        new_row[0] = pred 
        
        # Shift current sequence: drop first row, append new_row
        new_seq = np.empty_like(current_seq)
        new_seq[0, :-1, :] = current_seq[0, 1:, :]
        new_seq[0, -1, :] = new_row
        current_seq = new_seq
        
    return forecast_results
