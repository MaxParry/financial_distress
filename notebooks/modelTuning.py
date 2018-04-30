
# coding: utf-8

# # Predicting Financial Distress Notebook
# - Dataset can be found at: https://www.kaggle.com/c/GiveMeSomeCredit
#     - Contains historical data on 150,000 borrowers
#         - Labels are binary: defaulted or didn't default (SeriousDlqin2yrs)
#     - Goal is to predict default risk on holdout set

# In[5]:


# Import dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import Counter

import seaborn as sns


# Read CSV
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

# Run add / remove all columns

def shapeItUp(df):
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

# outside of function

# feature engineering steps


df = add_monthlycosts_column(df)

df = add_incomeperdependent_column(df)
df = add_numtimespastdue_column(df)
df = add_indicator_column(df, 10000, 'MonthlyIncome', direction='above')
df = add_indicator_column(df, 5, 'NumTimesPastDue', direction='below')
df = add_indicator_column(df, 21, 'age', direction='below')
df = add_indicator_column(df, 65, 'age', direction='above')

# dropping multicollinear features
df = df.drop('DebtRatio', axis=1)

# ### Check for multicollinearity
# - new features, new possible multicollinearities

# ## Step 11: Fit Scaler and Transform Data

X = df.drop('SeriousDlqin2yrs', axis=1)
y = df.SeriousDlqin2yrs


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# fit StandardScaler and use it to transform both training and testing data
from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler().fit(X_train)

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# ## Step 12: Fit Model and Make Predictions


# instantiate logistic regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# fit the model on the cleaned training data
model.fit(X_train_scaled, y_train)

# make predictions on training set (want to evaluate model performance on training set first, then testing set)
class_predictions_training = model.predict(X_train_scaled)
probability_predictions_training = model.predict_proba(X_train_scaled)

# #### What is returned from ```model.predict_proba()``` is an array with the probability of each class for a given row
# - negative class (no default in our case) is in the first column of the array, positive in the second
# - ```roc_curve()``` function takes at least two arguments:
#     - first, the true labels
#     - second, the probability of the positive class (defaulted, in our case)

# run roc_curve
from sklearn.metrics import roc_curve, roc_auc_score

# fpr_training, tpr_training, thresholds_training = roc_curve(y_train, positive_probability_predictions_training)
#
# # print AUC for training data
# AUC_score_training = roc_auc_score(y_train, positive_probability_predictions_training)
# print('AUC: ', AUC_score_training)
#
#
# # ## Step 13: Evaluate Model on Testing Set
#
# # same as above, but on testing set
# probability_predictions_testing = model.predict_proba(X_test_scaled)
# positive_probability_predictions_testing = probability_predictions_testing[:,1]
# fpr_testing, tpr_testing, thresholds_testing = roc_curve(y_test, positive_probability_predictions_testing)
#
# AUC_score_testing = roc_auc_score(y_test, positive_probability_predictions_testing)
#
# coefficients = list(model.coef_)
# X_cols = list(X_train.columns)
#
# for i in range(0, len(coefficients[0])):
#     print(X_cols[i], 'feature strength: ', coefficients[0][i])
#     print('---------\n')



# ## Step 14: Hypertune Model Parameters

# ## Step 15: Try Other Models

# ## Step 16: Export Best Model

# In[ ]:


from sklearn.externals import joblib

joblib.dump(model, 'trained_default_model_v1.pkl')





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
