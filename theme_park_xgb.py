import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from feature_engineering import feature_engineering

# PATHS
path_to_weather = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/weather_data.csv")
path_to_train = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_train.csv")
path_to_validation = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_X_test_val.csv")
submission_path = Path("/Users/antoinechosson/Desktop/hackathon_eleven/submit_xgb.csv")

# Feature engineering
X, y = feature_engineering(path_to_train, path_to_weather, validation=False)
X_validation = feature_engineering(path_to_validation, path_to_weather, validation=True)

# Model definition and training
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=500,
    learning_rate=0.1,
    max_depth=8,
    random_state=42,
    verbosity=0
)
xgb_model.fit(X, y)

# Predict on validation/test set
validation_pred = xgb_model.predict(X_validation)

# Format into submission form
answer_df = pd.read_csv(path_to_validation)
answer_df["y_pred"] = validation_pred
answer_df["KEY"] = "Validation"
answer_df.drop(columns = ["ADJUST_CAPACITY","DOWNTIME","CURRENT_WAIT_TIME","TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"], inplace=True)
answer_df.to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path.resolve()}")
