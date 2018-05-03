import numpy as np
import pandas as pd

from sklearn.externals import joblib


# ### Add MonthlyCosts column
# write function to multiply DebtRatio by MonthlyIncome and put the result in a new column
def add_monthlycosts_column(df):
    df['MonthlyCosts'] = df['DebtRatio'] * df['MonthlyIncome']
    return df


# ### Add IncomePerDependent column
# write function to divide MonthlyIncome by (NumberOfDependents + 1) and put the result in a new column
def add_incomeperdependent_column(df):
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    return df

# ### Add NumTimesPastDue column
# write function to divide MonthlyIncome by (NumberOfDependents + 1) and put the result in a new column
# the function also drops the original columns
def add_numtimespastdue_column(df):
    df['NumTimesPastDue'] = df['NumberOfTime30to59DaysPastDueNotWorse'] + df['NumberOfTime60to89DaysPastDueNotWorse'] + df['NumberOfTimes90DaysLate']

    df = df.drop(['NumberOfTime30to59DaysPastDueNotWorse','NumberOfTime60to89DaysPastDueNotWorse','NumberOfTimes90DaysLate'], axis=1)

    return df


# ### Function to write indicator variable columns:
# define function to label rows with high monthly income with a 1 (in a new column)
def add_indicator_column(df, threshold, column_name, direction='above'):
    newColName = (str(column_name) + '_' + str(direction) + str(threshold))

    if direction == 'below':
        try:
            df[newColName] = np.where(df[column_name]<=threshold,float(1),float(0))
        except:
            print('Error in add_indicator_column(): Base case reached')
    else:
        try:
            df[newColName] = np.where(df[column_name]>=threshold,float(1),float(0))
        except:
                print('Error in add_indicator_column(): Base case reached')

    return df

# #FEATURE ENGINEERING
# Run add / remove all columns
def shapeItUp(df,training=False):
    if (training):
        df = add_monthlycosts_column(df)
        df = add_numtimespastdue_column(df)

    df = add_incomeperdependent_column(df)
    df = add_indicator_column(df, 10000, 'MonthlyIncome', direction='above')
    df = add_indicator_column(df, 5, 'NumTimesPastDue', direction='below')
    df = add_indicator_column(df, 21, 'age', direction='below')
    df = add_indicator_column(df, 65, 'age', direction='above')

    # dropping multicollinear features
    df = df.drop('DebtRatio', axis=1)

    if(training):
        df = df[['SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines', 'age',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'MonthlyCosts',
       'IncomePerDependent', 'NumTimesPastDue', 'MonthlyIncome_above10000',
       'NumTimesPastDue_below5', 'age_below21', 'age_above65']]

    else:
        df = df[['RevolvingUtilizationOfUnsecuredLines', 'age',
       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
       'NumberRealEstateLoansOrLines', 'NumberOfDependents', 'MonthlyCosts',
       'IncomePerDependent', 'NumTimesPastDue', 'MonthlyIncome_above10000',
       'NumTimesPastDue_below5', 'age_below21', 'age_above65']]

    return df


def makePredictions(valueDict):

    # load models
    rf_model = joblib.load('model/rf_model_v1.pkl')
    gbm_model = joblib.load('model/gbm_model_v1.pkl')
    X_scaler = joblib.load('model/fitted_X_scaler_v1.pkl')

    # Create
    df = pd.DataFrame(valueDict,index=[0])
    df = shapeItUp(df)

    X = df
    X_scaled = X_scaler.transform(X)

    rf_class_prediction = rf_model.predict(X_scaled)
    rf_probability_prediction = rf_model.predict_proba(X_scaled)

    gbm_class_prediction = gbm_model.predict(X_scaled)
    gbm_probability_prediction = gbm_model.predict_proba(X_scaled)

    def findAvg(value1,value2):
        return (value1+value2)/2

    predictions = {
        'rf_class': rf_class_prediction[0].item(),
        'gbm_class': gbm_class_prediction[0].item(),
        'probability': findAvg(rf_probability_prediction[0][0].item(),gbm_probability_prediction[0][0].item())
    }

    return predictions
