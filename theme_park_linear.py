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
from feature_engineering import feature_engineering

path_to_weather = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/weather_data.csv")
path_to_train = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_train.csv")
path_to_validation = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_X_test_val.csv")
save_path = Path("/Users/antoinechosson/Desktop/hackathon_eleven/theme_park_model.pt")
validation_path = Path("/Users/antoinechosson/Desktop/hackathon_eleven/hackathon_data/waiting_times_X_test_val.csv")
submission_path = Path("/Users/antoinechosson/Desktop/hackathon_eleven/submit.csv")

# Importing data for training

X, y = feature_engineering(path_to_train, path_to_weather, validation = False)

# Evaluation metric : RMSE 

def RMSE(y_pred, y_true):
    mse = nn.MSELoss()
    return torch.sqrt(mse(y_pred, y_true))

# LINEAR MODEL DEFINITION

# Hyperparameters 

epochs = 10000
lr = 0.003
hidden_units = 5
k = 5 
kf = KFold(n_splits = k, shuffle = True, random_state = 42)


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
    
    
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    # Splitting dataset into training and valiation 

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
    rmse = RMSE # Loss function 
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


# TESTING MODEL ON TEST DATASET

torch.save(theme_park_model.state_dict(), save_path)
model_V0 = ThemeParkModel(n_features, n_outputs = 1, hidden_units=hidden_units)
model_V0.load_state_dict(torch.load(save_path, weights_only=True))

model_V0.eval()

X_validation = feature_engineering(path_to_validation, path_to_weather, validation = True)
X_validation = torch.tensor(X_validation, dtype=torch.float32, device=device)
with torch.inference_mode(): 
    validation_pred = theme_park_model(X_validation)


# Format into submission form 

answer_df = pd.read_csv(validation_path)
answer_df["y_pred"] = validation_pred.cpu().numpy().flatten()
answer_df["KEY"] = "Validation"
answer_df.drop(columns = ["ADJUST_CAPACITY","DOWNTIME","CURRENT_WAIT_TIME","TIME_TO_PARADE_1","TIME_TO_PARADE_2","TIME_TO_NIGHT_SHOW"], inplace=True)

answer_df.to_csv(submission_path, index=False)








