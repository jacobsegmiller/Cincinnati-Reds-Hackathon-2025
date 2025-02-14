# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 23:12:20 2025

@author: seggy
"""
#Cincinnati Reds Hackathon 2025 Python Code
#IAA Team
#Andrew Buelna, Landon Docherty, Brett Laderman, Danny Ryan and Jacob Segmiller

#Reading in the necessary packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from optbinning import Scorecard
from matplotlib import pyplot as plt
from optbinning.scorecard import plot_auc_roc, plot_ks
from optbinning import OptimalBinning
from optbinning import BinningProcess
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

#Reading in the datasets
batters = pd.read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/batters.csv")
battingOrder = pd.read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/battingOrder.csv")
BattingFinalStats = pd.read_csv("C:/Users/seggy/Documents/Reds Hackathon 2025/battingstats_final_age.csv")
pitches = pd.read_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\Pitcher_Stats_Full.csv")
pitches = pitches.replace([float('inf'), -float('inf')], float('nan'))  # Replace inf with NaN

#Merging Data
Hitters = pd.merge(batters, battingOrder, on = ["batter", "year"], how = "left")

#Getting rid of infinity and NA's
Hitters = Hitters.replace([float('inf'), -float('inf')], float('nan'))  # Replace inf with NaN
Hitters = Hitters.dropna()  # Drop rows with NaN
Hit = BattingFinalStats.replace([float('inf'), -float('inf')], float('nan'))  # Replace inf with NaN
Hit = BattingFinalStats.dropna()  # Drop rows with NaN

#Selecting Columns 
columns = ['games', 'H', 'Singles', 'Doubles', 'Triples', 'HR', 'SO', 
                    'BB', 'HBP', 'SF', 'SB', 'CI', 'BA', 'OBP', 'SLG', 'OPS', 
                    'ISO', 'BABIP', 'K_perc', 'BB_perc', 'HR_perc', 'XBH', 'TB', 
                    'RC', 'Contact_perc', 'BB_K_ratio', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' ,'PA', 'age']
selected_columns = ['SF', 'OPS', 'K_perc', 'BB_perc', 'TB', 
                    'Contact_perc', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' ,'PA', 'age']
selected_columns2 = ['SF', 'OPS', 'K_perc', 'BB_perc', 'TB', 
                    'Contact_perc', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' , 'age']
Hitters1 = Hitters[selected_columns]
Hitters_pred = Hit[selected_columns2]

#testing out stat prediction
selected_columns2 = ['SF', 'OPS', 'K_perc', 'BB_perc', 'TB', 
                    'Contact_perc', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' ,'PA', 'age', 'year', 'batter']
hitters_2 = Hitters[selected_columns2]
Train = hitters_2[(hitters_2["year"] == 2021) | (hitters_2["year"] == 2022)]
Test = hitters_2[(hitters_2["year"] == 2023)]
hitters_2.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\hitters_2.csv")
#outputting Train
#Train.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\Train.csv")

#pivoting wider
#pivoting on batter, and using year to go to column names 
baseball = Train.pivot(index="batter", columns="year")
#fixing column names
baseball.columns = [f"{col[0]}_{col[1]}" for col in baseball.columns]
baseball.reset_index(inplace=True)
#creating flags 
baseball["in_2021"] = baseball.filter(like="_2021").notna().any(axis=1).astype(int)
baseball["in_2022"] = baseball.filter(like="_2022").notna().any(axis=1).astype(int)
#selected colmsn 3 is 2 without the age year and batter
selected_columns3 = ['SF', 'OPS', 'K_perc', 'BB_perc', 'TB', 
                    'Contact_perc', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' ]

#trying Out All Combinations of Weights
weights = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5)]
for weight_1, weight_2 in weights:
    for var in selected_columns3:
        baseball[f"{var}_2023"] = baseball.apply(
            lambda row: (weight_1 * row[f"{var}_2021"] + weight_2 * row[f"{var}_2022"])
            if row["in_2021"] == 1 and row["in_2022"] == 1
            else row[f"{var}_2022"],
            axis=1
        )
    #test dataset (using 2023)
    test_good = Test.rename(columns={col: col + "_2023" for col in selected_columns3})
    #keeping variables with suffix of 2023 or batter in name of variable
    baseball_good = baseball.filter(regex="_2023|^batter$")
    #merging on batter id
    merged_baseball = baseball_good.merge(test_good, on="batter", suffixes=("_pred", "_actual"))
    #rmse per column
    for var in selected_columns3:
        merged_baseball[f"{var}_RMSE"] = np.sqrt(
            np.mean((merged_baseball[f"{var}_2023_pred"] - merged_baseball[f"{var}_2023_actual"]) ** 2))
    #keeping rmse columns 
    merged_baseball_filter1 = merged_baseball.filter(like="_RMSE").iloc[0]
    filename = f"C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\one_{int(weight_1*10)}_two_{int(weight_2*10)}.csv"
    merged_baseball_filter1.to_csv(filename)

#Officially projecting out 2024 stats
selected_columns3 = ['SF', 'OPS', 'K_perc', 'BB_perc', 'TB', 
                    'Contact_perc', 'RC_per_PA', 'XBH_perc', 
                    'weighted_spot_avg' ,'PA', 'age', 'year', 'batter']
hitters_pre = Hitters[selected_columns2]
#pivoting on batter, and using year as the columns
hitters_good = hitters_pre.pivot(index="batter", columns="year")
#fixing column names and indexes
hitters_good.columns = [f"{col[0]}_{col[1]}" for col in hitters_good.columns]
hitters_good.reset_index(inplace=True)
#creating flags for years played/ present in the dataset
hitters_good["in_2021"] = hitters_good.filter(like="_2021").notna().any(axis=1).astype(int)
hitters_good["in_2022"] = hitters_good.filter(like="_2022").notna().any(axis=1).astype(int)
hitters_good["in_2023"] = hitters_good.filter(like="_2023").notna().any(axis=1).astype(int)
#creating variable ind
#if they are in 21/22/23 then ind = 1
#if they are in 22/23 then ind = 2
#if they are in 23 then ind = 3
def ind(row):
    if row['in_2021'] == 1 and row['in_2022'] == 1 and row['in_2023'] == 1:
        return 1
    elif row['in_2021'] == 0 and row['in_2022'] == 1 and row['in_2023'] == 1:
        return 2
    elif row['in_2021'] == 0 and row['in_2022'] == 0 and row['in_2023'] == 1:
        return 3
    else:
        return 4
#applying above function to ind
hitters_good['year_ind'] = hitters_good.apply(ind, axis =1)
#checking the year flags
freq_data = hitters_good[['in_2021', 'in_2022', 'in_2023', 'year_ind']]
combo_counts = freq_data.groupby(['in_2021', 'in_2022', 'in_2023', 'year_ind']).size()
print(combo_counts)

#Applying year weights based on year_ind
# {0.4 -> 0.6} --> {0.2 -> 0.3 -> 0.5} {TB, weighted_spot_avg} (cols1)
# {0.3 -> 0.7} --> {0.15 -> 0.25 -> 0.6} {K_perc, Contact_perc, RC_per_PA} (cols2)
# {0.2 -> 0.8} --> {0.1 -> 0.2 -> 0.7} {SF, OPS, BB_perc, XBH_perc} (cols3)
cols1 = ['TB', 'weighted_spot_avg']
cols2 = ['K_perc', 'Contact_perc', 'RC_per_PA']
cols3 = ['SF', 'OPS', 'BB_perc', 'XBH_perc']

#Making hitters_good1
hitters_good1 = hitters_good

#cols1 loop
for var in cols1:
    hitters_good1[f"{var}_2024"] = hitters_good1.apply(
        lambda row: (
            (0.2 * row[f"{var}_2021"]) + (0.3 * row[f"{var}_2022"]) + (0.5 * row[f"{var}_2023"])  # For year_ind == 1
            if row["year_ind"] == 1 
            else (
                (0.4 * row[f"{var}_2022"]) + (0.6 * row[f"{var}_2023"])  # For year_ind == 2
                if row["year_ind"] == 2 
                else (1 * row[f"{var}_2023"])  # For other cases
            )
        ),
        axis=1
    )
    
#cols2 loop
for var in cols2:
    hitters_good1[f"{var}_2024"] = hitters_good1.apply(
        lambda row: (
            (0.15 * row[f"{var}_2021"]) + (0.25 * row[f"{var}_2022"]) + (0.6 * row[f"{var}_2023"])  # For year_ind == 1
            if row["year_ind"] == 1 
            else (
                (0.3 * row[f"{var}_2022"]) + (0.7 * row[f"{var}_2023"])  # For year_ind == 2
                if row["year_ind"] == 2 
                else (1 * row[f"{var}_2023"])  # For other cases
            )
        ),
        axis=1
    )
    
#cols3 loop
for var in cols3:
    hitters_good1[f"{var}_2024"] = hitters_good1.apply(
        lambda row: (
            (0.1 * row[f"{var}_2021"]) + (0.2 * row[f"{var}_2022"]) + (0.7 * row[f"{var}_2023"])  # For year_ind == 1
            if row["year_ind"] == 1 
            else (
                (0.2 * row[f"{var}_2022"]) + (0.8 * row[f"{var}_2023"])  # For year_ind == 2
                if row["year_ind"] == 2 
                else (1 * row[f"{var}_2023"])  # For other cases
            )
        ),
        axis=1
    )

#Getting final dataset ready
hitters_good2 = hitters_good1
#only keeping 2024 stats and the stats we want
columns_to_keep = ['batter'] + [col for col in hitters_good2.columns if col.endswith('_2024')]
hitters_good3 = hitters_good2[columns_to_keep]
#dropping _2024 from column names 
hitters_good3.rename(columns={col: col.replace('_2024', '') for col in hitters_good3.columns}, inplace=True)
#hitters_good3.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\battingstats_final.csv")

#Dropping NA's for Pitchers
pitches = pitches.dropna()  # Drop rows with NaN

#Testing out stat prediction
selected_columns2 = ['WHIP'	,'H', 'HR', 'ER', 'SO', 'BB','HBP','SOBBratio', 'pitcher', 'Year']
pitchers_2 = pitches[selected_columns2]
Train = pitchers_2[(pitchers_2["Year"] == 2021) | (pitchers_2["Year"] == 2022)]
Test = pitchers_2[(pitchers_2["Year"] == 2023)]
#outputting Train
#Train.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\Train.csv")

#Pivoting wider 
#dropping dulicate rows
Train = Train.drop_duplicates(subset=["pitcher", "Year"], keep="first")
baseball1 = Train.pivot(index="pitcher", columns="Year")
baseball1.columns = [f"{col[0]}_{col[1]}" for col in baseball1.columns]
baseball1.reset_index(inplace=True)
baseball1["in_2021"] = baseball1.filter(like="_2021").notna().any(axis=1).astype(int)
baseball1["in_2022"] = baseball1.filter(like="_2022").notna().any(axis=1).astype(int)
#selected colmsn 3 is 2 without the age year and batter
selected_columns3 = ['WHIP'	,'H', 'HR', 'ER', 'SO', 'BB','HBP','SOBBratio']

#Running Though the Combinatons
for var in selected_columns3:

    baseball1[f"{var}_2023"] = baseball1.apply(
        lambda row: (0.4 * row[f"{var}_2021"] + 0.6 * row[f"{var}_2022"])
        if row["in_2021"] == 1 and row["in_2022"] == 1
        else row[f"{var}_2022"],
        axis=1
    )
test_good = Test.rename(columns={col: col + "_2023" for col in selected_columns3})
baseball1_good = baseball1.filter(regex="_2023|^pitcher$")
merged_baseball1 = baseball1_good.merge(test_good, on="pitcher", suffixes=("_pred", "_actual"))
for var in selected_columns3:
    merged_baseball1[f"{var}_RMSE"] = np.sqrt(
        np.mean((merged_baseball1[f"{var}_2023_pred"] - merged_baseball1[f"{var}_2023_actual"]) ** 2))
merged_baseball1_filter1 = merged_baseball1.filter(like="_RMSE").iloc[0]
#merged_baseball1_filter1.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\P_baseball1.csv")

#Officially projecting out 2024 stats (starters)
selected_columns4 = ['WHIP'	,'H', 'HR', 'ER', 'SO', 'BB','HBP','SOBBratio', 'pitcher', 'Year']
pitchers_pre_pre = pitches[(pitches['role_key'] == 'SP')]
pitchers_pre = pitchers_pre_pre[selected_columns4]
#Pivoting on pitchver, and using Year for column names
pitchers_good = pitchers_pre.pivot(index="pitcher", columns="Year")
#Fixing column names
pitchers_good.columns = [f"{col[0]}_{col[1]}" for col in pitchers_good.columns]
pitchers_good.reset_index(inplace=True)
#Making indicator variable
pitchers_good["in_2021"] = pitchers_good.filter(like="_2021").notna().any(axis=1).astype(int)
pitchers_good["in_2022"] = pitchers_good.filter(like="_2022").notna().any(axis=1).astype(int)
pitchers_good["in_2023"] = pitchers_good.filter(like="_2023").notna().any(axis=1).astype(int)
#creating variable ind
#if they are in 21/22/23 then ind = 1
#if they are in 22/23 then ind = 2
#if they are in 23 then ind = 3
#Function for year indicator
def ind(row):
    if row['in_2021'] == 1 and row['in_2022'] == 1 and row['in_2023'] == 1:
        return 1
    elif row['in_2021'] == 0 and row['in_2022'] == 1 and row['in_2023'] == 1:
        return 2
    elif row['in_2021'] == 0 and row['in_2022'] == 0 and row['in_2023'] == 1:
        return 3
    else:
        return 4
#Applying above function to ind
pitchers_good['year_ind'] = pitchers_good.apply(ind, axis =1)
#Making pitchers_good1
pitchers_good1 = pitchers_good

#Loop through
cols1 = ['WHIP'	,'H', 'HR', 'ER', 'SO', 'BB','HBP','SOBBratio']
#cols1 loop
for var in cols1:
    pitchers_good1[f"{var}_2024"] = pitchers_good1.apply(
        lambda row: (
            (0.2 * row[f"{var}_2021"]) + (0.3 * row[f"{var}_2022"]) + (0.5 * row[f"{var}_2023"])  # For year_ind == 1
            if row["year_ind"] == 1 
            else (
                (0.4 * row[f"{var}_2022"]) + (0.6 * row[f"{var}_2023"])  # For year_ind == 2
                if row["year_ind"] == 2 
                else (1 * row[f"{var}_2023"])  # For other cases
            )
        ),
        axis=1
    )
pitchers_good2 = pitchers_good1
# Keep 'batter' and columns with '_2024' suffix
columns_to_keep = ['pitcher'] + [col for col in pitchers_good2.columns if col.endswith('_2024')]
#keeping only the above columns
pitchers_good3 = pitchers_good2[columns_to_keep]
#dropping the _2024 suffix
pitchers_good3.rename(columns={col: col.replace('_2024', '') for col in pitchers_good3.columns}, inplace=True)
#pitchers_good3.to_csv("C:\\Users\\dryan\\OneDrive\\Documents\\NC_state\\Spring\\baseball\\SP_final.csv")

#Setting Up X and Y
X = Hitters1.drop(columns=['PA'])  # Features
y = Hitters1['PA']  # Target (PA)

#Train/Test Split
# Split data into training (2021/2022) and test (2023)
train_set = Hitters[Hitters['year'].isin([2021, 2022])]
test_set = Hitters[Hitters['year'] == 2023]
# Prepare training and testing data
X_train = train_set[selected_columns].drop(columns=['PA'])  # Features for training
y_train = train_set['PA']  # Target for training
X_test = test_set[selected_columns].drop(columns=['PA'])  # Features for testing
y_test = test_set['PA']  # Target for testing

#Build XGBoost Model
model = xgb.XGBRegressor(n_estimators = 50,
                        subsample = 0.5,
                        random_state = 1234)
#Training the model (pre-tuning)
model.fit(X_train, y_train)
#Predictions
y_pred = model.predict(X_test)
#Root Mean Square Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE: ", rmse)
#First Tuning
# Parameters
param_grid = {
    'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'eta': [0.1, 0.15, 0.2, 0.25, 0.3],
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'subsample': [0.25, 0.5, 0.75, 1]
}
# Grid search
grid_search = GridSearchCV(estimator=model, param_grid = param_grid, cv = 10)
# Fitting the grid search
grid_search.fit(X_train, y_train)
#Best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)
#Getting the best model
best_model = grid_search.best_estimator_
#Re-Tuning 2nd time
#Parameters
param_grid2 = {
    'n_estimators': [30, 35, 40, 45, 50, 55, 60],
    'eta': [0.15, 0.18, 0.2, 0.22, 0.25],
    'max_depth': [1, 2, 3, 4, 5, 6, 7],
    'subsample': [0.5, 0.65, 0.7, 0.75, 0.8, 0.85]
}
#Grid search
grid_search2 = GridSearchCV(estimator=model, param_grid = param_grid2, cv = 10)
#Fitting the grid search
grid_search2.fit(X_train, y_train)
#Best parameters
best_params2 = grid_search2.best_params_
print("Best Parameters:", best_params2)
#Getting the best model
best_model2 = grid_search2.best_estimator_
#Re-Tuning 3rd time
#Parameters
param_grid3 = {
    'n_estimators': [45, 50, 55, 60, 65, 70],
    'eta': [0.18, 0.2, 0.22, 0.23, 0.24, 0.25],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.5, 0.65, 0.7, 0.75, 0.8, 0.85]
}
#Grid search
grid_search3 = GridSearchCV(estimator=model, param_grid = param_grid3, cv = 10)
#Fitting the grid search
grid_search3.fit(X_train, y_train)
#Best parameters
best_params3 = grid_search3.best_params_
print("Best Parameters:", best_params3)
# Getting the best model
best_model3 = grid_search3.best_estimator_
#Re-Tuning 4th time
#Parameters
param_grid4 = {
    'n_estimators': [55, 60, 65, 70, 75, 80],
    'eta': [0.2, 0.21, 0.22, 0.23, 0.24],
    'max_depth': [2, 3, 4, 5, 6],
    'subsample': [0.65, 0.7, 0.75, 0.8, 0.85]
}
#Grid search
grid_search4 = GridSearchCV(estimator=model, param_grid = param_grid4, cv = 10)
#Fitting the grid search
grid_search4.fit(X_train, y_train)
#Best parameters
best_params4 = grid_search4.best_params_
print("Best Parameters:", best_params4)
#Getting the best model
best_model4 = grid_search4.best_estimator_
#Variable Importance and Selection
xgb_mlb = xgb.XGBRegressor(n_estimators = 80,
                        subsample = 0.75,
                        eta = 0.22,
                        max_depth = 4,
                        random_state = 1234)
xgb_mlb.fit(X_train, y_train)
forest_importances = pd.Series(xgb_mlb.feature_importances_, index = xgb_mlb.feature_names_in_)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
xgb.plot_importance(xgb_mlb, importance_type = 'cover')
plt.show()
X_train_r = X_train
X_train_r['random'] = np.random.normal(0, 1, 1344)
xgb_mlb = xgb.XGBRegressor(n_estimators = 80,
                        subsample = 0.75,
                        eta = 0.22,
                        max_depth = 4,
                        random_state = 1234)

xgb_mlb.fit(X_train_r, y_train)
forest_importances = pd.Series(xgb_mlb.feature_importances_, index = xgb_mlb.feature_names_in_)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
#XGBoost Evaluation
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse:.2f}")
# First Try: Root Mean Squared Error: 14.95
# Best Parameters: {'eta': 0.2, 'max_depth': 4, 'n_estimators': 50, 'subsample': 0.75}
# Evaluation
y_pred2 = best_model2.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
print(f"Root Mean Squared Error: {rmse:.2f}")
#Second Try: Root Mean Squared Error: 14.96
#Best Parameters: {'eta': 0.22, 'max_depth': 4, 'n_estimators': 60, 'subsample': 0.75}
#Evaluation
y_pred3 = best_model3.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred3))
print(f"Root Mean Squared Error: {rmse:.2f}")
#Third Try: Root Mean Squared Error: 14.58
#Best Parameters: {'eta': 0.22, 'max_depth': 4, 'n_estimators': 70, 'subsample': 0.75}
#Evaluation
y_pred4 = best_model4.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred4))
print(f"Root Mean Squared Error: {rmse:.2f}")
#Fourth Try: Root Mean Squared Error: 14.43
#Best Parameters: {'eta': 0.22, 'max_depth': 4, 'n_estimators': 80, 'subsample': 0.75}
#FINAL PREDICTION
#Evaluation
pred = xgb_mlb.predict(Hitters_pred)
#Best Parameters: {'eta': 0.22, 'max_depth': 4, 'n_estimators': 80, 'subsample': 0.75}
Hitters_pred['batter'] = BattingFinalStats.loc[Hitters_pred.index, 'batter']
pred_df = pd.DataFrame({'batter': Hitters_pred['batter'], 'pred': pred})
final_df = BattingFinalStats[['batter']].merge(pred_df, on='batter', how='left')
final_df['pred'] = final_df['pred'].fillna(0)
print(final_df.head())
final_df.to_csv('Batting_Stats_Final.csv', index=False)

#Random Forest HittersModel
#Build Initial Random Forest Model
rf_batters = RandomForestRegressor(n_estimators = 100,
                                   random_state = 12345,
                                   oob_score = True)
rf_batters.fit(X_train, y_train)
#Check Out of Bag Score
rf_batters.oob_score_ #0.9927272012401183
#Plot Variable Importance
forest_importances = pd.Series(rf_batters.feature_importances_, index = rf_batters.feature_names_in_)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax = ax)
ax.set_title("Feature importances")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show() #Lot of Importance for Total Bases
#Tune the Initial Random Forest Model
param_grid = {
    'bootstrap': [True],
    'max_features': [3, 4, 5, 6, 7, 8 ,9 ,10,11,12,13,14,15],
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800]
}
rf = RandomForestRegressor(random_state = 12345)
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 10)
grid_search.fit(X_train, y_train)
grid_search.best_params_
#1st run: {'bootstrap': True, 'max_features': 10, 'n_estimators': 200}
#Build 1st Final Random Forest
rf_batters = RandomForestRegressor(n_estimators = 200,
                                   max_features = 10,
                                   random_state = 12345,
                                   oob_score = True)
rf_batters.fit(X_train, y_train)
# Evaluation
y_predrf1 = rf_batters.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predrf1))
print(f"Root Mean Squared Error: {rmse:.2f}")
#1st Final Random Forest RMSE: 17.75
#2nd run: {'bootstrap': True, 'max_features': 10, 'n_estimators': 200}