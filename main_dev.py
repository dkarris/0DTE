# https://grok.com/share/bGVnYWN5_654f9fe9-1e40-445e-ba48-6e86d68645d6

import pandas as pd
import numpy as np
import datetime
from pandasql import sqldf
import sqlite3
import plotly.express as px


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Or another classifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder  # Use OneHotEncoder


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def create_daily_summary(df):
    # Convert time_converted to datetime
    df['time_converted'] = pd.to_datetime(df['time_converted'])
    
    # Step 1: Process options data
    filtered_options = df[(df['ticker'] == 'SPY') & 
                         (df['DTE_adjusted'] == 0) & 
                         (df['options_earliest_open'] > 0.2)]
    
    options_summary = (filtered_options.groupby(filtered_options['time_converted'].dt.date)
                      .agg({
                          'options_pct_change': [
                              ('Call_Max_Daily', lambda x: x[filtered_options.loc[x.index, 'type'] == 'call'].max()),
                              ('Put_Max_Daily', lambda x: x[filtered_options.loc[x.index, 'type'] == 'put'].max())
                          ]
                      })
                      .reset_index()
                      .rename(columns={'time_converted': 'Date'}))
    
    options_summary.columns = ['Date', 'Call_Max_Daily', 'Put_Max_Daily']
    options_summary['Date'] = pd.to_datetime(options_summary['Date']).dt.date  # Keep as date only
    
    # Step 2: Process equity data
    filtered_equity = df[df['ticker'] == 'SPY'].copy()
    
    # Daily aggregates
    daily_agg = (filtered_equity.groupby(filtered_equity['time_converted'].dt.date)
                 .agg({
                     'equity_start_price': 'max',  # equity_open
                     'high_equity': 'max',
                     'low_equity': 'min'
                 })
                 .reset_index()
                 .rename(columns={'time_converted': 'Date', 'equity_start_price': 'equity_open'}))
    
    daily_agg['Date'] = pd.to_datetime(daily_agg['Date']).dt.date  # Keep as date only
    
    # Last close_equity before 3:45 PM with fallback to last non-null
    filtered_equity['time_only'] = filtered_equity['time_converted'].dt.time
    # Get all SPY data first, then filter and sort
    equity_by_day = filtered_equity[filtered_equity['ticker'] == 'SPY'].copy()
    last_candle_candidates = equity_by_day[equity_by_day['close_equity'].notna()]  # Only rows with valid close_equity
    last_candle_candidates['time_only'] = last_candle_candidates['time_converted'].dt.time  # Add time_only here
    if last_candle_candidates.empty:
        print("Warning: No valid close_equity values found. Check data.")
    else:
        # Sort globally by date ascending and time descending, then filter and sort within groups
        last_candle = (last_candle_candidates
                      .sort_values(by=['time_converted'], key=lambda x: x.map(lambda t: (t.date(), -t.hour * 3600 - t.minute * 60 - t.second)), ascending=True)
                      .groupby(last_candle_candidates['time_converted'].dt.date)
                      .apply(lambda x: x[x['time_only'] <= pd.to_datetime('15:45:00').time()].sort_values(by='time_converted', ascending=False).head(1))
                      .reset_index(drop=True)
                      [['time_converted', 'close_equity']]
                      .rename(columns={'time_converted': 'Date'}))
        last_candle['Date'] = pd.to_datetime(last_candle['Date']).dt.date  # Keep as date only
        print(f"Last candle candidates rows: {len(last_candle_candidates)}")
        print(f"Unique last candle rows: {len(last_candle)}")
        print(last_candle.head())  # Debug to check content
    
    # Join daily_agg with last_candle
    equity_summary = daily_agg.merge(last_candle, on='Date', how='left')
    
    # Step 3: Left join options and equity data
    daily_summary = options_summary.merge(equity_summary, on='Date', how='left')
    
    # Ensure Date is in datetime format for final output
    daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
    
    # Debug: Print intermediate and final results
    print(f"Options rows: {len(options_summary)}")
    print(f"Equity rows: {len(equity_summary)}")
    print(f"Final daily_summary rows: {len(daily_summary)}")
    print(daily_summary.head())
    print(daily_summary.dtypes)
    print(f"NaN count in close_equity: {daily_summary['close_equity'].isna().sum()}")
    
    return daily_summary


def day_classification(row):
    very_high_range = 4
    high_range = 2
    average_range = 1.5
    if row['Max'] > very_high_range:
        return 'very_high'
    elif row['Max'] > high_range:
        return 'high'
    elif row['Max'] > average_range:
        return 'average'
    else:
        return 'low'

df = pd.read_parquet('optionsDB2023-2025_deduped.parquet')
df['time_converted'] = pd.to_datetime(df['time_converted'])
daily_summary = create_daily_summary(df)
daily_summary['Max'] = daily_summary[['Call_Max_Daily', 'Put_Max_Daily']].max(axis=1)
daily_summary['Date'] = pd.to_datetime(daily_summary['Date'])
daily_summary['weekday'] = daily_summary['Date'].dt.day_name()
# Assume daily_summary has 'Max' from SQL
daily_summary['day_classification'] = daily_summary.apply(day_classification, axis=1)
def create_lags(df):
    df['d1'] = df.loc[:,'day_classification'].shift(1)
    df['d2'] = df.loc[:,'day_classification'].shift(2)
    df['d3'] = df.loc[:,'day_classification'].shift(3)
    df['weekday_encoded'] = LabelEncoder().fit_transform(df['weekday'])  # Encode weekday
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    weekday_encoded = ohe.fit_transform(df[['weekday']])
    df = pd.concat([df, pd.DataFrame(weekday_encoded, columns=ohe.get_feature_names_out(['weekday']))], axis=1)
    df['week_number_in_month'] = df['Date'].apply(lambda x: (x.day-1) // 7 +1 )  # Week number in month
    df['week_sin'] = np.sin(2 * np.pi * df['week_number_in_month'] / 5)  # Assuming max 5 weeks
    df['week_cos'] = np.cos(2 * np.pi * df['week_number_in_month'] / 5)
    df['month_half'] = df['Date'].apply(lambda x: 1 if x.day <= 15 else 2)  # First or second half of the month
    df.dropna(inplace=True)
    return df
def split_feature(df):
    # X = df.loc[:,['d1', 'd2', 'd3', 'weekday_encoded', 'week_sin', 'week_cos']]  # Features
    y = df.loc[:,'day_classification']
    # features_list
    features_list = ['d1', 'week_sin']
    weekdays = [col for col in df.columns if 'weekday_' in col]
    # features_list.extend(weekdays)
    X = df.loc[:,features_list]
    #
    # agggregate days in 2 groups: <2 and >=2

    y = y.apply(lambda x: 1 if 'high' in x else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14, stratify=y) # Stratify for class balance
    return X_train, X_test, y_train, y_test
df = create_lags(daily_summary)
X_train, X_test, y_train, y_test = split_feature(df)

# le = LabelEncoder()
# y_train_encoded = le.fit_transform(y_train)
# y_test_encoded = le.transform(y_test)

# Encode the features (d1, d2, d3)
le_features = LabelEncoder()
# for column in ['d1', 'd2', 'd3']:
for column in ['d1']:
    X_train[column] = le_features.fit_transform(X_train[column])
    X_test[column] = le_features.transform(X_test[column])
# Train a classifier (RandomForest is a good starting point)
rf = RandomForestClassifier(random_state=12, class_weight='balanced') # You can try other models (e.g., Gradient Boosting)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy', verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model (convert predictions back to original labels for reporting)
#y_pred_labels = le.inverse_transform(y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate the model
#print(classification_report(y_test, y_pred))

# Feature Importance (to see which lags are most influential)
feature_importances = model.feature_importances_
print("Feature Importances:", feature_importances)


#features = X_train.columns.tolist()

print(X_train.head())
