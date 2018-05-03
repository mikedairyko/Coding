# Example of how to use scikit-learn
# Followed along from from the Analytics Vidhya website:
# https://www.analyticsvidhya.com/blog/2015/01/scikit-learn-python-machine-learning-tool/

# Description of Variables
# The dataset contains 6366 observations of 9 variables:

# rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
# age: woman's age
# yrs_married: number of years married
# children: number of children
# religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious)
# educ: level of education (9 = grade school, 12 = high school, 14 = some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)
# occupation: woman's occupation (1 = student, 2 = farming/semi-skilled/unskilled, 3 = "white collar", 4 = teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 = professional with advanced degree)
# occupation_husb: husband's occupation (same coding as above)
# affairs: time spent in extra-marital affairs

# ----------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# load dataset
dta =  sm.datasets.fair.load_pandas().data

########## DATA PRE-PROCESSING ##########

# Add column 'affair' where 1 = Yes and 0 = No
dta['affair'] = (dta['affairs'] > 0).astype(int)


########## DATA EXPLORATION ##########

dta.groupby('affair').mean()
# On average women who have affairs rate their marriage lower and have been married longer

dta.groupby('rate_marriage').mean()

# On average an increase in age, years married, and children correlate with 
# a declining marriage rating

########## DATA VISUALIZATION ##########

# Histogram of Education Level
dta.educ.hist()
#dta['educ'].hist()

# Labels
plt.title('Histogram of Education')
plt.xlabel('Education Level')
plt.ylabel('Frequency')

# Grid Lines
plt.grid(False)

# Histogram of Education Level
dta.rate_marriage.hist()
#dta['educ'].hist()

# Labels
plt.title('Histogram of Marriage Rating')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

# Grid Lines
plt.grid(False)

# Crosstab 
# Returns a sub matrix with selected rows and columns
pd.crosstab(dta.rate_marriage, dta.affair.astype(bool))

# Barplot of marriage rating grouped by affair

pd.crosstab(dta.rate_marriage, dta.affair.astype(bool)).plot(kind = 'bar')

plt.title('Marriage Rating Distribution by Affair Status')
plt.xlabel('Marriage Rating')
plt.ylabel('Frequency')

affair_years_married = pd.crosstab(dta.yrs_married, dta.affair.astype(bool))
affair_years_married.div( affair_years_married.sum(1).astype(float), axis = 0)
affair_years_married.div( affair_years_married.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)

########## PREPARE DATA FOR LOGISTICAL REGRESSION ##########

# We need to prepare the data. Add an intercept column and dummy variables for 
# occupation and occupation_husb since these two are categorial variables

y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children +\
                    religious + educ + C(occupation) + C(occupation_husb)',
                    dta, return_type ='dataframe')

# Rename is the columns names of the dummy variables

X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
                        'C(occupation)[T.3.0]' : 'occ_3',
                        'C(occupation)[T.4.0]' : 'occ_4',
                        'C(occupation)[T.5.0]' : 'occ_5',
                        'C(occupation)[T.6.0]' : 'occ_6',
                        'C(occupation_husb)[T.2.0]' : 'occupation_husb_2',
                        'C(occupation_husb)[T.3.0]' : 'occupation_husb_3',
                        'C(occupation_husb)[T.4.0]' : 'occupation_husb_4',
                        'C(occupation_husb)[T.5.0]' : 'occupation_husb_5',
                        'C(occupation_husb)[T.6.0]' : 'occupation_husb_6'})

# Flatten y into 1-D array (so scikit will properly interpret it as the response variable)
y = np.ravel(y)

########## LOGISTIC REGRESSION ##########

model = LogisticRegression()
model = model.fit(X,y)

# check the accuracy on the training set
model.score(X,y)

# So there is 73% accuracy

# Null Error Rate : what percentage had affairs
y.mean()

# only 32% of women had affairs
# This means that there is 68% accuracy for just always predicting NO
# So our model does better than the null error rate, but not by much

# examine the cofficients to see what we can learn

df1 = pd.DataFrame( X.columns)
df2 = pd.DataFrame(np.transpose(model.coef_))
pd.concat([df1, df2], axis = 1)

# an increase in marriage rating and religiousness correspond to a decrease in the likelihood of having an affair.
# for both occupations, the lowest likelihood of having an affair corresponds to the baseline occupation (student)

########## MODEL EVALUATION USING A VALIDATION SET ##########

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# Predict the class labels for the test set
predicted = model2.predict(X_test)

# Generate class probabilities 
# predicts 1 whenever second column is greater than 0.5
probs = model2.predict_proba(X_test)

# generate evaluation metrics
print(metrics.accuracy_score(y_test, predicted))
# This has an accuracy of 73%

print(metrics.roc_auc_score(y_test, probs[:, 1]))
print(metrics.confusion_matrix(y_test, predicted))
print(metrics.classification_report(y_test, predicted))

########## MODEL EVALUATION USING CROSS-VALIDATION ##########

# evaluate the model using 10-fold cross-validation

scores = cross_val_score(LogisticRegression(), X, y, scoring = 'accuracy', cv = 10)
print(scores)
print(scores.mean())

# Performs at 73% accuracy 


