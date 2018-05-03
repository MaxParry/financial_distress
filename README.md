# Project 3: Predicting Financial Distress

### Dataset
Give Me Some Credit, a dataset of 150,000 borrowers’ histories, each labelled with defaulted or didn’t default on a loan.

### Story
Banks look at credit scores to assess a potential borrower’s default risk, and thus the interest rate to charge for a loan. Machine learning has the potential to allow creditors to make more fair and accurate decisions by using historical information on whether a borrower defaulted or not.
We propose to use the Give Me Some Credit dataset to train a supervised classification model to determine whether a borrower will default based on their current financial information. We feel that if we can provide a more accurate means of credit scoring, we can help build a more just credit system.

### Roadmap
- Data exploration
    - Split data and continue with training data
    - Determine cleaning requirements through exploratory analysis .describe() and visualize distributions of values in columns (training and testing set) to inform
- Data cleaning
    - imputation/dropping of missing values, null values, entry errors
    - Produce correlation matrix to test for multicollinearity
    - Drop or combine multicollinearity-risk features
    - Assess testing data for out-of-scale values
- Prepare data
    - Get dummies/ Label encode/ one-hot encode categorical features
    - Drop multicollinearity-risk features
    - Encapsulate cleaning and feature engineering steps in a function and run it on testing data
- Fit model
    - Split predictors and target variable
    - Normalize/scale features (take testing data feature ranges into consideration)
    - Address likely unbalanced target class
    - Fit appropriate model (may try a few)
- Evaluate model
    - Determine proper evaluation metric (ROC/AUC)
    - Tune
    - Hypertune model parameters and re-evaluate
- Model delivery
    - UI design
    - Flask app design
    - Model integration
- Challenges
    - Unbalanced classes
    - Utilizing new models
    - Ensembling and evaluation of models
