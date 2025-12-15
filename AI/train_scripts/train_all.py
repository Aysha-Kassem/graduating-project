import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# ==============================
# 1) DATA PREPROCESSING
# ==============================
def prepare_common_data(filepath=None, time_steps=50, test_size=0.2, step_size=2):
    import os

    if filepath is None:
        # AI/train_scripts/train_all.py → AI/
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "dataset", "DataSet.csv")

    df = pd.read_csv(filepath, low_memory=False)

    # Extract wrist sensor data
    df['WristAccelerometer_x'] = pd.to_numeric(df['WristAccelerometer'], errors='coerce')
    df['WristAccelerometer_y'] = pd.to_numeric(df['Unnamed: 30'], errors='coerce')
    df['WristAccelerometer_z'] = pd.to_numeric(df['Unnamed: 31'], errors='coerce')
    df['WristAngularVelocity_x'] = pd.to_numeric(df['WristAngularVelocity'], errors='coerce')
    df['WristAngularVelocity_y'] = pd.to_numeric(df['Unnamed: 33'], errors='coerce')
    df['WristAngularVelocity_z'] = pd.to_numeric(df['Unnamed: 34'], errors='coerce')

    FEATURES = ['WristAccelerometer_x','WristAccelerometer_y','WristAccelerometer_z',
                'WristAngularVelocity_x','WristAngularVelocity_y','WristAngularVelocity_z']

    # Magnitude
    df['Acc_mag'] = np.sqrt(df['WristAccelerometer_x']**2 +
                            df['WristAccelerometer_y']**2 +
                            df['WristAccelerometer_z']**2)
    df['Gyro_mag'] = np.sqrt(df['WristAngularVelocity_x']**2 +
                             df['WristAngularVelocity_y']**2 +
                             df['WristAngularVelocity_z']**2)
    FEATURES += ['Acc_mag','Gyro_mag']

    # Define Fall Now based on Tag
    FALL_CODES = [7,8,9,10,11]
    df['fall_now'] = df['Tag'].apply(lambda x: 1 if x in FALL_CODES else 0)

    # Fall Soon
    HORIZON = 10
    fall_series = df['fall_now'].values
    df['fall_soon'] = [int(fall_series[i+1:i+HORIZON+1].max()) if i+HORIZON < len(fall_series) else 0
                       for i in range(len(fall_series))]

    df[FEATURES] = df[FEATURES].fillna(0)

    X = df[FEATURES].values
    y_now = df['fall_now'].values
    y_soon = df['fall_soon'].values

    X_train_raw, X_test_raw, y_train_now, y_test_now, y_train_soon, y_test_soon = train_test_split(
        X, y_now, y_soon, test_size=test_size, random_state=42, stratify=y_now
    )

    # Ensure folders exist
    os.makedirs("../scaler", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    joblib.dump(scaler, "../scaler/scaler_all.save")

    # Sliding window
    def create_sequences(data, target, steps, step_size=1):
        X_seq, y_seq = [], []
        for i in range(0, len(data)-steps, step_size):
            X_seq.append(data[i:i+steps])
            y_seq.append(target[i+steps-1])
        return np.array(X_seq), np.array(y_seq)

    X_train, y_train_now_seq = create_sequences(X_train_scaled, y_train_now, time_steps, step_size)
    _, y_train_soon_seq = create_sequences(X_train_scaled, y_train_soon, time_steps, step_size)
    X_test, y_test_now_seq = create_sequences(X_test_scaled, y_test_now, time_steps, step_size)
    _, y_test_soon_seq = create_sequences(X_test_scaled, y_test_soon, time_steps, step_size)

    return X_train, X_test, y_train_now_seq, y_test_now_seq, y_train_soon_seq, y_test_soon_seq

# ==============================
# 2) ATTENTION BLOCK
# ==============================
def attention_block(inputs):
    score = layers.Dense(128, activation='tanh')(inputs)
    score = layers.Dense(1, activation='sigmoid')(score)
    attention = layers.Multiply()([inputs, score])
    context = layers.GlobalAveragePooling1D()(attention)
    return context

# ==============================
# 3) MODEL BUILDERS
# ==============================
def build_lstm_attention(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Add()([x, x2])
    x = layers.LayerNormalization()(x)
    att = attention_block(x)
    shared = layers.Dense(128, activation='relu')(att)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0008),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

def build_bigru_attention(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x2 = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Add()([x, x2])
    x = layers.LayerNormalization()(x)
    att = attention_block(x)
    shared = layers.Dense(128, activation='relu')(att)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0008),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

def build_bigru_lstm_attention(time_steps, features):
    inp = Input(shape=(time_steps, features))
    x_gru = layers.Bidirectional(layers.GRU(128, return_sequences=True))(inp)
    x_gru = layers.LayerNormalization()(x_gru)
    x_gru = layers.Dropout(0.3)(x_gru)
    x_lstm = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_gru)
    x = layers.Add()([x_gru, x_lstm])
    x = layers.LayerNormalization()(x)
    att = attention_block(x)
    shared = layers.Dense(128, activation='relu')(att)
    shared = layers.Dropout(0.3)(shared)
    fall_now = layers.Dense(64, activation='relu')(shared)
    fall_now = layers.Dense(1, activation='sigmoid', name='fall_now')(fall_now)
    fall_soon = layers.Dense(64, activation='relu')(shared)
    fall_soon = layers.Dense(1, activation='sigmoid', name='fall_soon')(fall_soon)
    model = Model(inp, [fall_now, fall_soon])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0008),
        loss='binary_crossentropy',
        metrics={
            "fall_now": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
            "fall_soon": ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    return model

# ==============================
# 4) TRAIN FUNCTION
# ==============================
def train_all_models():
    X_train, X_test, y_train_now, y_test_now, y_train_soon, y_test_soon = prepare_common_data()

    models_info = [
        ("LSTM_Attention", build_lstm_attention),
        ("BiGRU_Attention", build_bigru_attention),
        ("BiGRU_Attention_LSTM", build_bigru_lstm_attention)
    ]

    for name, builder in models_info:
        print(f"\nTraining {name} model...")
        model = builder(X_train.shape[1], X_train.shape[2])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=4),
            ModelCheckpoint(f"../models/BEST_{name}.keras", save_best_only=True, monitor='val_loss')
        ]
        model.fit(
            X_train,
            {"fall_now": y_train_now, "fall_soon": y_train_soon},
            validation_data=(X_test, {"fall_now": y_test_now, "fall_soon": y_test_soon}),
            epochs=80,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )
        model.save(f"../models/FINAL_{name}.keras")
        print(f"{name} model saved successfully!")

if __name__=="__main__":
    train_all_models()
