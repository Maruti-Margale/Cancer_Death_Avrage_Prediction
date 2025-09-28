🚀 Cancer Death Average Prediction App

🔗 Live Demo: cancerdeathavrageprediction-maruti.streamlit.app

📊 Project Overview

This Streamlit-based machine learning app predicts the average annual cancer death count (avganncount) based on 28 health, demographic, and environmental features provided by the user.

Below is the core logic of how the app works:

```mermaid
graph TD
    A[Start: User Enters 28 Feature Values] --> B[Input Data is Collected as Raw Values]
    B --> C{Load SCALER_PARAMS: Min/Max Values from Training Data}
    C --> D[Apply MinMax Scaling Formula to Each Feature]
    D --> E[Scaled Input Data (Normalized to 0_to_1 Range)]
    E --> F{Load Cancer_Regression.pkl Model}
    F --> G[Model Predicts avganncount]
    G --> H[End: Display Predicted Annual Cancer Count]
```
