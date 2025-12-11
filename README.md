# Fall Detection System - Overview

## 1. System Overview

This system is designed for **real-time fall detection and early fall prediction** using motion and vital sensors. The main components are:

- **Sensors**: Collect real-time motion and vital sign data.
- **Database**: Stores raw sensor data and predictions.
- **Data Preprocessing**: Converts raw sensor data into a format suitable for AI models.
- **AI Models Ensemble**: Uses multiple models to predict falls:
  - LSTM + Attention
  - BiGRU + Attention
  - BiGRU-LSTM + Attention
- **Prediction**: Detects `Fall Now` and `Fall Soon`.
- **Notification System**: Sends alerts via mobile app push notifications or SMS.

---

## 2. System Flow (Diagram)

```text
   ┌─────────────┐          ┌─────────────┐
   │ Motion      │          │ Vital       │
   │ Sensor      │          │ Sensor      │
   └───────┬─────┘          └───────┬─────┘
           │                        │
           │                        │
           ▼                        ▼
   ┌───────────────────────────────────┐
   │          Database                 │
   │  (Raw sensor & prediction data)  │
   └─────────────┬────────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Data Preprocessing│
        └─────────┬────────┘
                  │
                  ▼
        ┌─────────────────────────┐
        │ AI Models Ensemble      │
        │ (LSTM + BiGRU + Hybrid)│
        └─────────┬──────────────┘
                  │
          Prediction Output:
          ┌───────────────┐
          │ Fall Now      │
          │ Fall Soon     │
          └──────┬────────┘
                 │
                 ▼
        ┌───────────────────┐
        │ Notification System│
        │ (Mobile App / SMS) │
        └───────────────────┘

## 3. Components Description

### 3.1 Sensors
- **Motion Sensor**: Detects movement patterns in real-time.
- **Vital Sensor**: Measures heart rate, blood pressure, and other vital signs continuously.

### 3.2 Database
- Stores:
  - Raw sensor data
  - Preprocessed data
  - Model predictions with timestamps and probabilities

### 3.3 Data Preprocessing
- Steps include:
  - Scaling/normalization
  - Sequence creation for time-series data
  - Feature engineering

### 3.4 AI Models Ensemble
- Combines predictions from:
  - **LSTM + Attention**: Good for sequential patterns
  - **BiGRU + Attention**: Captures long-term dependencies
  - **BiGRU-LSTM + Attention**: Hybrid model for robust prediction
- Ensemble improves prediction accuracy.

### 3.5 Prediction
- Two outputs:
  - **Fall Now**: Immediate fall detected
  - **Fall Soon**: Early prediction of possible fall

### 3.6 Notification System
- Sends alerts to users via:
  - Mobile App push notifications
  - SMS or other messaging systems
- Ensures rapid response in case of emergencies.

---

## 4. Summary

The system continuously monitors motion and vital signs, processes the data, predicts potential or ongoing falls, stores results, and triggers notifications to prevent accidents and ensure safety.

