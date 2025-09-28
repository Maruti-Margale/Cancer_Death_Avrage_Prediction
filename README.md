# Cancer_Death_Avrage_Prediction
Live Demo : https://cancerdeathavrageprediction-maruti.streamlit.app/#county-cancer-incidence-predictor-avganncount

```mermaid
graph TD
    A[Start: User Enters 28 Feature Values] --> B(Input Data is Collected as Raw Values)
    B --> C{Load SCALER_PARAMS: Min/Max Values from Training Data}
    C --> D(Apply MinMax Scaling Formula to Each Feature)
    D --> E[Scaled Input Data (Normalized to 0-1 Range)]
    E --> F{Load Cancer_Regression.pkl Model}
    F --> G(Model Predicts avganncount)
    G --> H[End: Display Predicted Annual Cancer Count]
```
