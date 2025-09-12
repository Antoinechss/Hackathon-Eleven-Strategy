import pandas as pd 
pd.set_option('display.max_columns', None)
import numpy as np 
from pathlib import Path 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
from torch import nn 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATA PROCESSING ---------

path_to_weather = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/weather_data.csv")

def data_processing(path_to_dataset, validation):

    other_features = pd.read_csv(path_to_dataset)
    weather = pd.read_csv(path_to_weather)

    # Merging with weather 
    df = pd.merge(other_features, weather, on='DATETIME', how='inner')

    # dropping columns with too much missing data 
    miss_pct = df.isna().mean().sort_values(ascending=False) # Seing which column has missing values 
    df = df.drop(columns=["TIME_TO_PARADE_2"])

    # managing date and time 
    #df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    #df["DATETIME"] = df["DATETIME"].astype('int64')

    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df["hour"] = df["DATETIME"].dt.hour
    df["dow"] = df["DATETIME"].dt.dayofweek
    df["month"] = df["DATETIME"].dt.month

    # cyclical encoding for hour/dow
    df["hour_sin"]  = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"]  = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]   = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dow"]/7)

    df = df.drop(columns=["DATETIME","hour","dow"])  # keep cyclical + month if useful

    # Turning attraction name (str) into labels : one-hot encoding
    df = pd.get_dummies(df, columns=["ENTITY_DESCRIPTION_SHORT"] , drop_first=True)

    # Filling in missing NaN time to parade 1 and time to night show by the average times in the column 
    mean_time_to_parade1 = df["TIME_TO_PARADE_1"].mean(numeric_only=True)
    mean_time_to_night_show = df["TIME_TO_NIGHT_SHOW"].mean(numeric_only=True)
    df = df.fillna({"TIME_TO_PARADE_1": mean_time_to_parade1, "TIME_TO_NIGHT_SHOW": mean_time_to_night_show})
    df = df.fillna({"snow_1h": 0, "rain_1h": 0})

    if not validation:
        y = df["WAIT_TIME_IN_2H"]
        df = df.drop(columns=["WAIT_TIME_IN_2H"])
        # Data normalization 
        return df, y
    else:
        df = StandardScaler().fit_transform(df)
        return df

path = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_train.csv")
X, y = data_processing(path, False)

# --- Evaluation metric 

def RMSE(y_pred, y_true):
    mse = nn.MSELoss()
    return torch.sqrt(mse(y_pred, y_true))

# --- MODEL ---------------

# Hyperparameters 
epochs = 10000
lr = 0.003
hidden_units = 5
k = 5 # nb of folds for cross-validation 

class ThemeParkModel(nn.Module):
    def __init__(self, n_features, n_outputs=1, hidden_units=hidden_units):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(n_features, hidden_units),
            nn.ReLU(),                            
            nn.Linear(hidden_units, n_outputs),)
    
    def forward(self,x):
        y_pred = self.linear_layer_stack(x)
        return y_pred 
    
kf = KFold(n_splits = k, shuffle = True, random_state = 42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    print(f"Fold {fold+1}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_val   = scaler.transform(X.iloc[val_idx])

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y.values[train_idx], dtype=torch.float32, device=device).view(-1,1)
    X_val   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val   = torch.tensor(y.values[val_idx], dtype=torch.float32, device=device).view(-1,1)
        
    n_features = X_train.shape[1]

    theme_park_model = ThemeParkModel(n_features, n_outputs = 1,hidden_units=hidden_units).to(device)
    rmse = RMSE
    optimizer = torch.optim.Adam(theme_park_model.parameters(), lr)
    
    for epoch in range (epochs): 

        # train 
        theme_park_model.train()
        y_pred = theme_park_model(X_train)
        loss = RMSE(y_pred,y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # validation  
        theme_park_model.eval()
        with torch.inference_mode(): 
            val_pred = theme_park_model(X_val)
            val_loss = RMSE(val_pred,y_val.type(torch.float))

        if epoch % 100 == 0 : 
            print(epoch, val_loss.item())
    
    print (fold+1, val_loss)


# TEST ON VALIDATION DATASET

# Run model on validation 

save_path = "/Users/antoinechosson/Desktop/hackathon_eleven/theme_park_model.pt"
torch.save(theme_park_model.state_dict(), save_path)
model_V0 = ThemeParkModel(n_features, n_outputs = 1, hidden_units=hidden_units)
model_V0.load_state_dict(torch.load(save_path, weights_only=True))

model_V0.eval()

validation_path = "/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_X_test_val.csv"
X_validation = data_processing(validation_path, True)
X_validation = torch.tensor(X_validation, dtype=torch.float32, device=device)
with torch.inference_mode(): 
    validation_pred = theme_park_model(X_validation)

# Format into submission form 

answer_df = pd.read_csv(validation_path)
answer_df["y_pred"] = validation_pred.cpu().numpy().flatten()
answer_df["KEY"] = "Validation"
answer_df.drop(columns = ["ADJUST_CAPACITY","DOWNTIME","CURRENT_WAIT_TIME","TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"], inplace=True)

answer_df.to_csv("/Users/antoinechosson/Desktop/hackathon_eleven/submit.csv", index=False)








