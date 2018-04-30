# # Predicting Financial Distress Notebook
# - Dataset can be found at: https://www.kaggle.com/c/GiveMeSomeCredit
#     - Contains historical data on 150,000 borrowers
#         - Labels are binary: defaulted or didn't default (SeriousDlqin2yrs)
#     - Goal is to predict default risk on holdout set

# Import dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.externals import joblib

from collections import Counter

import seaborn as sns

# Read CSV
#df = pd.read_csv('../data/cs-test.csv')
df = pd.read_csv('../data/cs-training.csv')
df = df.drop('Unnamed: 0', axis=1)

# Clean column names
cleanCol = []
for i in range(len(df.columns)):
    cleanCol.append(df.columns[i].replace('-', 'to'))

df.columns = cleanCol

# Calculate std dev values
def findStd(series,num):
    mean = series.mean()
    stdDev = series.std()
    std_pos = mean + num*stdDev
    std_neg = mean - num*stdDev

    stdValues = {
        'std+': std_pos,
        'std-': std_neg
    }

    return stdValues

# Remmove Outliers and replace
def standardizeValues(dfColumn,stdRet,columnName,classifier='mean',compareColumn='N/A',lookupTable='N/A'):
    if classifier == 'lookup':
        df.loc[dfColumn.isnull(),columnName] = compareColumn.map(lookupTable)
        df.loc[dfColumn>stdRet['std+'],columnName] = compareColumn.map(lookupTable)
        df.loc[dfColumn<stdRet['std-'],columnName] = compareColumn.map(lookupTable)
    elif classifier == 'median':
        df.loc[dfColumn.isnull(),columnName] = dfColumn.median()
        df.loc[dfColumn>stdRet['std+'],columnName] = dfColumn.median()
        df.loc[dfColumn<stdRet['std-'],columnName] = dfColumn.median()
    else:
        df.loc[dfColumn.isnull(),columnName] = dfColumn.mean()
        df.loc[dfColumn>stdRet['std+'],columnName] = dfColumn.mean()
        df.loc[dfColumn<stdRet['std-'],columnName] = dfColumn.mean()


# Remove and Replace default entries
def standardizePlaceholders(dfColumn,columnName,classifier='mean'):
    if classifier == 'median':
        df.loc[dfColumn==98,columnName] = dfColumn.median()
        df.loc[dfColumn==96,columnName] = dfColumn.median()

    else:
        df.loc[dfColumn==98,columnName] = dfColumn.mean()
        df.loc[dfColumn==98,columnName] = dfColumn.mean()


# Get std deviation and replace outliers
ageStd = findStd(df.age,3)
standardizeValues(df.age,ageStd,'age')
df.age = df.age.astype(int)


# Drop NAN from OG DF
df_MI = df[['age','MonthlyIncome']]


df_MIsansNAN = df_MI.dropna(axis=0, how='any')

# Remove Income Outliers outside of 3 STD
df_MIsansNAN_std = df_MIsansNAN[np.abs(df_MIsansNAN.MonthlyIncome-df_MIsansNAN.MonthlyIncome.mean())<=(3*df_MIsansNAN.MonthlyIncome.std())]


# Create lookup table for mean per age
ageSalaryLookup = pd.DataFrame(df_MIsansNAN_std.groupby(['age']).MonthlyIncome.mean())


# Find all values 3x std
incomeStd = findStd(df_MIsansNAN_std.MonthlyIncome,3)
standardizeValues(df.MonthlyIncome,incomeStd,'MonthlyIncome','lookup',df.age,ageSalaryLookup.MonthlyIncome)


# Define deviation and replace outliers
revLnStd = {'std+': 2, 'std-': 0}
standardizeValues(df.RevolvingUtilizationOfUnsecuredLines,revLnStd,'RevolvingUtilizationOfUnsecuredLines','median')


# Replace placeholders with the median
standardizePlaceholders(df.NumberOfTime30to59DaysPastDueNotWorse,'NumberOfTime30to59DaysPastDueNotWorse','median')

debtRatioStd = {'std+': 2, 'std-': 0}
standardizeValues(df.DebtRatio,debtRatioStd,'DebtRatio','median')


# Replace placeholders with the median
standardizePlaceholders(df.NumberOfTimes90DaysLate,'NumberOfTimes90DaysLate','median')
df.NumberOfTimes90DaysLate = df.NumberOfTimes90DaysLate.astype(int)


# Replace placeholders with the median
standardizePlaceholders(df.NumberOfTime60to89DaysPastDueNotWorse,'NumberOfTime60to89DaysPastDueNotWorse','median')
df.NumberOfTime60to89DaysPastDueNotWorse = df.NumberOfTime60to89DaysPastDueNotWorse.astype(int)


# Drop NAN from OG DF
df_D = df[['NumberOfDependents']]
df_DsansNAN = df_D.dropna(axis=0, how='any')


# Define deviation and replace outliers
depStd = {'std+': df.NumberOfDependents.max(), 'std-': df.NumberOfDependents.min()}
standardizeValues(df.NumberOfDependents,depStd,'NumberOfDependents','median')


# ### Add MonthlyCosts column
# write function to multiply DebtRatio by MonthlyIncome and put the result in a new column
def add_monthlycosts_column(dataframe):
    dataframe_copy = dataframe.copy()
    dataframe_copy['MonthlyCosts'] = dataframe_copy['DebtRatio'] * dataframe_copy['MonthlyIncome']

    return dataframe_copy


# ### Add IncomePerDependent column
# write function to divide MonthlyIncome by (NumberOfDependents + 1) and put the result in a new column
def add_incomeperdependent_column(dataframe):
    dataframe_copy = dataframe.copy()
    dataframe_copy['IncomePerDependent'] = dataframe_copy['MonthlyIncome'] / (dataframe_copy['NumberOfDependents'] + 1)

    return dataframe_copy

# ### Add NumTimesPastDue column
# write function to divide MonthlyIncome by (NumberOfDependents + 1) and put the result in a new column
# the function also drops the original columns
def add_numtimespastdue_column(dataframe):
    dataframe_copy = dataframe.copy()
    dataframe_copy['NumTimesPastDue'] = (dataframe_copy['NumberOfTime30to59DaysPastDueNotWorse'] +
                                         dataframe_copy['NumberOfTime60to89DaysPastDueNotWorse'] +
                                         dataframe_copy['NumberOfTimes90DaysLate'])

    dataframe_copy = dataframe_copy.drop('NumberOfTime30to59DaysPastDueNotWorse', axis=1)
    dataframe_copy = dataframe_copy.drop('NumberOfTime60to89DaysPastDueNotWorse', axis=1)
    dataframe_copy = dataframe_copy.drop('NumberOfTimes90DaysLate', axis=1)

    return dataframe_copy


# ### Function to write indicator variable columns:
# define function to label rows with high monthly income with a 1 (in a new column)
def add_indicator_column(dataframe, threshold, column_name, direction='above'):
    dataframe_copy = dataframe.copy()
    labels = []
    if direction == 'above':
        for index, row in dataframe_copy.iterrows():
            value = row[column_name]
            if value >= threshold:
                labels.append(float(1))
            elif value < threshold:
                labels.append(float(0))
            else:
                print('Error in add_indicator_column(): Base case reached')
    elif direction == 'below':
        for index, row in dataframe_copy.iterrows():
            value = row[column_name]
            if value <= threshold:
                labels.append(float(1))
            elif value > threshold:
                labels.append(float(0))
            else:
                print('Error in add_indicator_column(): Base case reached')
    if len(dataframe_copy) == len(labels):
        dataframe_copy[(str(column_name) + '_' + str(direction) + str(threshold))] = pd.Series(labels)
    else:
        print('Error in add_indicator_column(): Missing labels')
    return dataframe_copy

# #FEATURE ENGINEERING
# Run add / remove all columns
def shapeItUp(df,addMonthlyCosts=False):
    if (monthlyCosts):
        df = add_monthlycosts_column(df)

    df = add_incomeperdependent_column(df)

    df = add_indicator_column(df, 10000, 'MonthlyIncome', direction='above')
    df = add_indicator_column(df, 5, 'NumTimesPastDue', direction='below')
    df = add_indicator_column(df, 21, 'age', direction='below')
    df = add_indicator_column(df, 65, 'age', direction='above')

    # dropping multicollinear features
    df = df.drop('DebtRatio', axis=1)

    df = df[['RevolvingUtilizationOfUnsecuredLines', 'age',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'MonthlyCosts',
       'IncomePerDependent', 'NumTimesPastDue', 'MonthlyIncome_above10000',
       'NumTimesPastDue_below5', 'age_below21', 'age_above65']]

    return df

# Execute feature engineering steps
shapeItUp(df)


# df = add_monthlycosts_column(df)
#
# df = add_incomeperdependent_column(df)
# df = add_numtimespastdue_column(df)
# df = add_indicator_column(df, 10000, 'MonthlyIncome', direction='above')
# df = add_indicator_column(df, 5, 'NumTimesPastDue', direction='below')
# df = add_indicator_column(df, 21, 'age', direction='below')
# df = add_indicator_column(df, 65, 'age', direction='above')
#
# # dropping multicollinear features
# df = df.drop('DebtRatio', axis=1)

# ### Check for multicollinearity
# - new features, new possible multicollinearities

# ## Step 11: Fit Scaler and Transform Data

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df.SeriousDlqin2yrs

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# fit StandardScaler and use it to transform both training and testing data
X_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Step 12: Fit Model and Make Predictions
# instantiate logistic regression model


global model = LogisticRegression()

# fit the model on the cleaned training data
model.fit(X_train_scaled, y_train)

# make predictions on training set (want to evaluate model performance on training set first, then testing set)
class_predictions_training = model.predict(X_train_scaled)
probability_predictions_training = model.predict_proba(X_train_scaled)

# ## Fit Model and Make Predictions
# - will try:
#     - Logistic Regression
#     - Random Forest
#     - Gradient Boosted Classifier
#
# #### Note: What is returned from ```model.predict_proba()``` is an array with the probability of each class for a given row
# - negative class (no default in our case) is in the first column of the array, positive in the second
# - ```roc_curve()``` function takes at least two arguments:
#     - first, the true labels
#     - second, the probability of the positive class (defaulted, in our case)

# ### Logistic Regression
# instantiate logistic regression model
lr_model = LogisticRegression(penalty='l2',
                           dual=False,
                           tol=0.0001,
                           C=1.0,
                           fit_intercept=True,
                           intercept_scaling=1,
                           class_weight='balanced',
                           random_state=None,
                           solver='liblinear',
                           max_iter=100,
                           multi_class='ovr',
                           verbose=0,
                           warm_start=False,
                           n_jobs=1)

lr_model.fit(X_train_scaled, y_train)
lr_probability_predictions_training = lr_model.predict_proba(X_train_scaled)[:,1]
lr_probability_predictions_testing = lr_model.predict_proba(X_test_scaled)[:,1]

lr_fpr_testing, lr_tpr_testing, lr_thresholds_testing = roc_curve(y_test, lr_probability_predictions_testing)

#### Check model coefficients to gauge feature importance (LogReg only)
coefficients = list(lr_model.coef_)
X_cols = list(X_train.columns)

for i in range(0, len(coefficients[0])):
    print(X_cols[i], 'feature strength: ', coefficients[0][i])
    print('---------')

# ### Random Forest
# instantiate random forest classifier
rf_model = RandomForestClassifier(n_estimators=1000,
                               criterion='gini',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=80,
                               min_weight_fraction_leaf=0.0,
                               max_features='auto',
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_impurity_split=None,
                               bootstrap=True,
                               oob_score=False,
                               n_jobs=-1,
                               random_state=None,
                               verbose=0,
                               warm_start=False,
                               class_weight=None)

rf_model.fit(X_train_scaled, y_train)
rf_probability_predictions_training = rf_model.predict_proba(X_train_scaled)[:,1]
rf_probability_predictions_testing = rf_model.predict_proba(X_test_scaled)[:,1]

rf_fpr_testing, rf_tpr_testing, rf_thresholds_testing = roc_curve(y_test, rf_probability_predictions_testing)

# ### Random Forest
# instantiate gradient boosting classifier
gbm_model = GradientBoostingClassifier(loss='exponential',
                           learning_rate=0.11,
                           n_estimators=253,
                           subsample=1.0,
                           criterion='friedman_mse',
                           min_samples_split=14,
                           min_samples_leaf=2,
                           min_weight_fraction_leaf=0.0,
                           max_depth=3,
                           min_impurity_decrease=0.0,
                           min_impurity_split=None,
                           init=None,
                           random_state=None,
                           max_features=None,
                           verbose=0,
                           max_leaf_nodes=None,
                           warm_start=False,
                           presort='auto')

gbm_model.fit(X_train_scaled, y_train)
gbm_probability_predictions_training = gbm_model.predict_proba(X_train_scaled)[:,1]
gbm_probability_predictions_testing = gbm_model.predict_proba(X_test_scaled)[:,1]

gbm_fpr_testing, gbm_tpr_testing, gbm_thresholds_testing = roc_curve(y_test, gbm_probability_predictions_testing)



# Save out models
joblib.dump(rf_model, '../model/rf_model_v1.pkl')
joblib.dump(gbm_model, '../model/gbm_model_v1.pkl')


# Make predictions
def makePredictions(valueDict):

    model = joblib.load('trained_default_model_v1.pkl')

    df = pd.DataFrame(valueDict,index=[0])
    df = shapeItUp(df)

    X = df
    X_scaler = StandardScaler().fit(X)
    X_train_scaled = X_scaler.transform(X)

    class_prediction = model.predict(X_train_scaled)
    probability_prediction = model.predict_proba(X_train_scaled)

    predictions = {
        'class': class_prediction[0].item(),
        'probability': probability_prediction[0][1].item()
    }

    return predictions
