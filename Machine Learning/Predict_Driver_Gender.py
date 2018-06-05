############# OVERVIEW OF THE DATASET #########################
# The Stanford Open Policing Project data
# This dataset comes from: https://openpolicing.stanford.edu/data/
# and more information on data: https://github.com/5harad/openpolicing/blob/master/DATA-README.md

# Data represents the traffic stops in the state of Colorado from January 2010 to February 2017.
# 544 MB
# There are 25 columns:

# 'state',
# 'stop_date',
# 'stop_time',
# 'location_raw',
# 'county_name',
# 'county_fips',
# 'fine_grained_location',
# 'police_department',
# 'driver_gender',
# 'driver_age_raw',
# 'driver_age',
# 'driver_race_raw',
# 'driver_race',
# 'violation_raw',
# 'violation',
# 'search_conducted',
# 'search_type_raw',
# 'search_type',
# 'contraband_found',
# 'stop_outcome',
# 'is_arrested',
# 'officer_id',
# 'officer_gender',
# 'vehicle_type',
# 'out_of_state'

# Goal: to predict the gender of a driver involved in a traffic stop in Colorado

#############################################################################################################################

import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss, roc_auc_score
from sklearn.model_selection import cross_validate

##### Import and clean the data #####
df = pd.read_csv('CO_cleaned.csv', index_col = 'id')

# Finds number of null values per column
df.isnull().sum()

# list of columns with too many null values. 
null_cols = ['fine_grained_location', 'search_type_raw', 'search_type','officer_gender']

# list of columns with repeative information or not related to driver_gender
rep_cols = ['state','county_fips', 'driver_race_raw', 'driver_age_raw', 'stop_outcome',
             'violation_raw', 'vehicle_type','county_name','police_department', 'officer_id' ]

drop_cols = null_cols + rep_cols
df = df.drop(drop_cols, axis = 1) 

df.isnull().sum()

df.info()

########### Exploring and visualizing the data ##########

df.boxplot( column = 'driver_age', by = 'contraband_found', showfliers=False, figsize=(6,6))

df[['driver_age', 'contraband_found']].groupby('contraband_found').mean()

# loosely speaking, the younger a driver is, the more likely it is for them to have 
# contraband found in the car. The following is a function that approximates a drivers 
# age based on whether they had contraband 

def age_approx(cols):
    driver_age = cols[0]
    contraband_found = cols[1]
     
    if pd.isnull(driver_age):
        if contraband_found == 1:
            return 26
        else:
            return 36
    else:
        return driver_age

df['driver_age'] = df[['driver_age', 'contraband_found']].apply(age_approx, axis=1)

# Convert the race of the driver: White = 1, Hispanic = 2, Black = 3, Asian = 4
race = {'White': 1, 'Hispanic': 2, 'Black': 3, 'Asian':4, 'Other': 5}
df['driver_race']= df['driver_race'].map(race)

# Convert driver gender to binary: Male = 1, Female = 0
gender = {'M': 1, 'F': 0}
df['driver_gender']= df['driver_gender'].map(gender)

# Convert strings to integers
d = {True: 1, False: 0}
df['search_conducted'] = df['is_arrested'].map(d)
df['out_of_state'] = df['out_of_state'].map(d)
df['contraband_found'] = df['contraband_found'].map(d)
df['is_arrested'] = df['is_arrested'].map(d)

# There were no more loose relationships found, so we must drop remaining NaN rows
df = df.dropna()

df['driver_race'].value_counts()

# Noticed that there is now only one row with driver_race as 'other'. Drop this row
df = df[ df['driver_race'] != 5]

# Create month and violation bins for further data exploration

def assign_month(M):
    if M.strftime('%m') == '01':
        return 1
    if M.strftime('%m') == '02':
        return 2
    if M.strftime('%m') == '03':
        return 3
    if M.strftime('%m') == '04':
        return 4
    if M.strftime('%m') == '05':
        return 5
    if M.strftime('%m') == '06':
        return 6
    if M.strftime('%m') == '07':
        return 7
    if M.strftime('%m') == '08':
        return 8
    if M.strftime('%m') == '09':
        return 9
    if M.strftime('%m') == '10':
        return 10
    if M.strftime('%m') == '11':
        return 11
    if M.strftime('%m') == '12':
        return 12


# create bins for month column for data exploration 
df['stop_date'] = pd.to_datetime(df['stop_date'])
df['month_label'] = df['stop_date'].apply(assign_month)

#dummy_df = pd.get_dummies(df['month_label'], prefix = 'month')
# Combine the two datasets
#df = pd.concat([df, dummy_df], axis =1)

# I noticed these 10 categorical violations using df['violation'].unique().tolist()

def assign_vio_bin(W):
    if W.lower()[0:5] == 'light':
        return 1
    if W.lower()[0:3] == 'dui':   
        return 2
    if W.lower()[0:7] == 'license':
        return 3
    if W.lower()[0:9] == 'paperwork':
        return 4
    if W.lower()[0:6] == 'moving':
        return 5
    if W.lower()[0:9] == 'equipment':
        return 6
    if W.lower()[0:4] == 'seat':
        return 7
    if W.lower()[0:12] == 'registration':
        return 8
    if W.lower()[0:5] == 'truck':
         return 9
    else:
        return 10   
    

# create bins for violation column for data exploration
first_word_violation = df['violation'].apply(lambda x: x.split()[0])
df['violation_bins'] = first_word_violation.apply(assign_vio_bin)

# create dummy variables for each 'violation'
#dummy_df = pd.get_dummies(df['violation_bins'], prefix = 'violation')
# Combine the two datasets
#df = pd.concat([df, dummy_df], axis =1)

df.boxplot( column = 'driver_age', by = 'driver_race', showfliers=False, figsize=(6,6))

import matplotlib.pyplot as plt

plt.hist(df['driver_age'], bins = 30, histtype = 'stepfilled')
plt.xlabel('Age of Driver')
plt.title('Distribution of Age')
plt.show()

pd.crosstab(df['month_label'], df['is_arrested']).plot(kind = 'bar')
plt.title('Arrest Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Arrest')

arrested = df[df['is_arrested'] == 1]
pd.crosstab(df['month_label'],arrested['is_arrested'] ).plot(kind = 'bar')

df.groupby('driver_gender').boxplot( column = 'driver_age', by = 'month_label', showfliers=False, figsize=(14,6))

import seaborn as sns

sns.countplot( x = 'driver_gender', data = df, palette = 'hls')
plt.xlabel('Driver Gender')
plt.title('Frequency of Driver Gender')
plt.show()

df['driver_gender'].value_counts()

numeric_cols = ['driver_gender', 'driver_age', 'driver_race',
 'search_conducted', 'out_of_state','month_label', 'violation_bins', 'is_arrested']

data = df[numeric_cols]

######## Logistic Regression #############3
# I decided to use logisitic regression as a classifier becuase the target column 'driver_gender' is binary
# I am interested studying how various factors influence the gender of the person being pulled over

y=data['driver_gender']
X=data.drop('driver_gender', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=47)

# check classification scores of logistic regression
logreg = LogisticRegression(class_weight = 'balanced')
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

scoring = {'accuracy': 'accuracy', 'auc': 'roc_auc'}

modelCV = LogisticRegression(class_weight = 'balanced')

results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))
                               
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# DriverA is 27, Black, search was not conducted, no contraband found, is from out of state, month is June,
# was pulled over for a broken taillight, and was not arrested. What is the gender of DriverA?

DriverA = [[27,3,0,1,6,1,0]]
DriverA_gender = logreg.predict(DriverA)
print("driver info=%s, Predicted=%s" % (DriverA, DriverA_gender))
