# Scikit-Learn Tutorial: Baseball Analytics Pt 1
# Source: DataCamp (https://www.datacamp.com/community/tutorials/scikit-learn-tutorial-baseball-1)
# SQLite database: (https://github.com/jknecht/baseball-archive-sqlite)
# Additonal information on the database: (http://www.seanlahman.com/files/database/readme2016.txt)

# Here I followed along the with the tutorial and learned how to test out several machine learning
# model from sklearn to predict the number of games an MLB team won in a season based on stats from the season. 
# In particular, how to use scikit-Learn to analyze sports data:
#    - import, clean, and visualize data from an SQLite database
#    - engineer several new features 
#    - create a K-means clustering model
#    - create a couple different Linear Regression models
#    - test prediction with the mean absolute error metric.

#################################################################################################################

####### IMPORT THE DATA #######
# Read in the data using sqlite3 and convert to DataFrame using pandas

import pandas as pd
import sqlite3

# Connect to the SQLite Database
con = sqlite3.connect('lahman2016.sqlite')

# Query the database for all seasons where a team played 150+ games and is still active today
query =  ''' SELECT * from Teams inner join TeamsFranchises on Teams.franchID == TeamsFranchises.franchID
             WHERE Teams.G > 150 and TeamsFranchises.active == 'Y'; '''

# Create dataframe from query
Teams = con.execute(query).fetchall()
teams_df = pd.DataFrame(Teams)

# print out first 5 rows and length of dataframe
print( teams_df.head())
print( len(teams_df))

####### CLEAN AND PREPARING THE DATA #######
# Adding the column names to the dataframe

cols = ['yearID','lgID','teamID','franchID','divID','Rank','G',
        'Ghome','W','L','DivWin','WCWin','LgWin','WSWin','R','AB',
        'H','2B','3B','HR','BB','SO','SB','CS','HBP','SF','RA','ER',
        'ERA','CG','SHO','SV','IPouts','HA','HRA','BBA','SOA','E','DP',
        'FP','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45',
        'teamIDretro','franchID','franchName','active','NAassoc']

teams_df.columns = cols

# Drop columns that are not related to Wins

drop_cols = ['lgID','franchID','divID','Rank','Ghome','L','DivWin','WCWin','LgWin',
             'WSWin','SF','name','park','attendance','BPF','PPF','teamIDBR','teamIDlahman45',
             'teamIDretro','franchID','franchName','active','NAassoc']

df = teams_df.drop(drop_cols, axis =1)
print(df.head())

# Check the count of null values in each column. Note that axis = 0: cols and axis = 1: rows
print(df.isnull().sum(axis=0).tolist())

# Finds the columns that contain null values
count = df.isnull().sum(axis=0).tolist()
nonNull = [index for index in range(len(count)) if count[index] >0]
nullCols = list(df.iloc[:,nonNull].columns)
df[nullCols].isnull().sum(axis=0)

# 'CS' and 'HBP' columns have too many null values, so we drop these columns
df = df.drop( ['CS', 'HBP'], axis =1)

# 'SO' and 'DP' null values will be filled with the median
df['SO'] = df['SO'].fillna(df['SO'] .median())
df['DP'] = df['DP'].fillna(df['DP'] .median())

# Check the count of null values in each column. Note that axis = 0: cols and axis = 1: rows
print(df.isnull().sum(axis=0).tolist())

############## EXPLORING AND VISUALIZING THE DATA ########################
import matplotlib.pyplot as plt

# matplotlib plots inline (for Jupyter notebook)
%matplotlib inline

# Plot the distribution of wins

# df['W'].plot(kind = 'hist') Alternate plot method
plt.hist(df['W'])
plt.xlabel('Wins')
plt.title('Distribution of Wins')
plt.show()

# Determine the average wins per year
print( df['W'].mean())

# Create bins for Win column for data exploration

def assign_win_bin(W):
    if W < 50:
        return 1
    if W >= 50 and W <= 69:
        return 2
    if W >= 70 and W <=89:
        return 3
    if W >=90 and W <= 109:
        return 4
    if W >= 110:
        return 5
# Apply function to Win column in df
​
df['win_bins'] = df['W'].apply(assign_win_bin)
​
# Create a scatter graph of Year vs Wins
​
plt.scatter(df['yearID'], df['W'], c = df['win_bins'])
plt.xlabel('Year')
plt.ylabel('Wins')
plt.title('Wins Scatter Plot')
plt.show()

# There are only a few games before 1900 and the game was different back then,
# so we can drop these rows from the data set

df = df[df['yearID'] > 1900]

# We now create a graph to indicate how much scoring there was for each year to account
# for the different runs per game eras in MLB history

# Create runs per year and games per year dictonaries

runs_per_year = {}
games_per_year = {}

for i, row in df.iterrows():
    year = row['yearID']
    runs = row['R']
    games = row['G']
    if year in runs_per_year:
        runs_per_year[year] = runs_per_year[year] + runs
        games_per_year[year] = games_per_year[year] + games
    else:
        runs_per_year[year] = runs
        games_per_year[year] = games
        
print(runs_per_year)
print(games_per_year)

# Create MLB runs per game (per year) dictonary 

mlb_runs_per_game = {}
for k,v in games_per_year.items():
    year = k
    games = v
    runs = runs_per_year[year]
    mlb_runs_per_game[year] = runs / games
    
print(mlb_runs_per_game) 

# Add mlb_runs_per_game (per year) column to dataset
def assign_mlb_rpg(year):
    return mlb_runs_per_game[year]

df['mlb_rpg'] = df['yearID'].apply(assign_mlb_rpg)

# Create a line plot from mlb_runs_per_game dictonary

lists = sorted(mlb_runs_per_game.items() )
years, runs = zip(*lists)

plt.plot(years , runs)
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')
plt.title('MLB Yearly Runs per Game')
plt.show()

# Create 'year_label' column, which gives algorithm information about how certain years are related
# (Dead ball eras, Live ball/ Steroid Eras)

def assign_label(year):
    if year < 1920:
        return 1
    elif year >= 1920 and year <= 1941:
        return 2
    elif year >= 1942 and year <= 1945:
        return 3
    elif year >= 1946 and year <= 1962:
        return 4
    elif year >= 1963 and year <= 1976:
        return 5
    elif year >= 1977 and year <= 1992:
        return 6
    elif year >= 1993 and year <= 2009:
        return 7
    elif year >= 2010:
        return 8
    
# Add 'year_label' column to dataframe
df['year_label'] = df['yearID'].apply(assign_label)

# create dummy variables for each 'era'
dummy_df = pd.get_dummies(df['year_label'], prefix = 'era')

# Combine the two datasets
df = pd.concat([df, dummy_df], axis =1)

# Convert the years into decades 

def assign_decade(year):
    if year < 1920:
        return 1910
    elif year >= 1920 and year < 1929:
        return 1920
    elif year >= 1930 and year < 1939:
        return 1930
    elif year >= 1940 and year < 1949:
        return 1940
    elif year >= 1950 and year < 1959:
        return 1950
    elif year >= 1960 and year < 1969:
        return 1960
    elif year >= 1970 and year < 1979:
        return 1970  
    elif year >= 1980 and year < 1989:
        return 1980
    elif year >= 1990 and year < 1999:
        return 1990
    elif year >= 2000 and year < 2009:
        return 2000    
    elif year > 2010:
        return 2010 
    
df['decade_label'] = df['yearID'].apply(assign_decade)
decade_dummydf = pd.get_dummies(df['decade_label'], prefix = 'decade')
df = pd.concat([df, decade_dummydf], axis = 1)

# Drop unnecessary columns
df = df.drop(['yearID', 'year_label', 'decade_label'], axis = 1)

# Create new features for Runs per Game and Runs Allowed per Game

df['R_per_game'] = df['R']/df['G']
df['RA_per_game'] = df['RA'] / df['G']

# Create scatter plots for runs per game vs wins and runs allowed per game vs wins

fig = plt.figure(figsize=(12,6))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.scatter( df['R_per_game'], df['W'], c= 'blue')
ax1.set_title('Runs per Game vs. Wins')
ax1.set_ylabel('Wins')
ax1.set_xlabel('Runs per Game')

ax2.scatter( df['RA_per_game'], df['W'], c ='red')
ax2.set_title('Runs Allowed per Game vs. Wins')
ax2.set_xlabel('Runs Allowed per Game')

plt.show()

# Check to see how the variables are correlated with the target variable
df.corr()['W']

########## Machine Learning on Data ###############

# Run K-means algorithm: Each data pt is assigned to a cluster based on which centroid has the lowest 
# Euclidean distance from the data point

attributes = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
       'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA',
       'SOA', 'E', 'DP', 'FP', 'mlb_rpg', 'era_1', 'era_2',
       'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8', 'decade_1910.0',
       'decade_1920.0', 'decade_1930.0', 'decade_1940.0', 'decade_1950.0',
       'decade_1960.0', 'decade_1970.0', 'decade_1980.0', 'decade_1990.0',
       'decade_2000.0', 'decade_2010.0', 'R_per_game', 'RA_per_game']

data_attributes = df[attributes]
print(df.head())

# Import necessary modules from `sklearn` 
from sklearn.cluster import KMeans
from sklearn import metrics

# Create silhouette score dictionary
s_score_dict = {}
for i in range(2,11):
    km = KMeans(n_clusters=i, random_state=1)
    l = km.fit_predict(data_attributes) 
    s_s = metrics.silhouette_score(data_attributes, l)
    s_score_dict[i] = [s_s]
print(s_score_dict)    

# Create K-means model and determine eculidean distances for each data point

kmeans_model = KMeans(n_clusters = 6, random_state = 1)
distances = kmeans_model.fit_transform(data_attributes)

# Create scatter plot using labels from K-means model as color
labels = kmeans_model.labels_

plt.scatter(distances[:,0], distances[:,1], c=labels)
plt.title('Kmeans Clusters')
plt.show()

# Add labels from K-means model to dataframe and attributes list

df['labels'] = labels
attributes.append('labels')

# create a new dataframe using only variables to be included in the model

numeric_cols = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB',
       'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts', 'HA', 'HRA', 'BBA',
       'SOA', 'E', 'DP', 'FP', 'mlb_rpg', 'era_1', 'era_2',
       'era_3', 'era_4', 'era_5', 'era_6', 'era_7', 'era_8', 'decade_1910.0',
       'decade_1920.0', 'decade_1930.0', 'decade_1940.0', 'decade_1950.0',
       'decade_1960.0', 'decade_1970.0', 'decade_1980.0', 'decade_1990.0',
       'decade_2000.0', 'decade_2010.0', 'R_per_game', 'RA_per_game', 'labels', 'W']

data = df[numeric_cols]
print(data.head())

# spilt dataframe into train and test sets

train = data.sample(frac = 0.75, random_state = 1)
test = data.loc[~ data.index.isin(train.index)]

x_train = train[numeric_cols].drop(columns = 'W')
y_train = train['W']
x_test = test[numeric_cols].drop(columns = 'W')
y_test = test['W']

# Select error metric and model
# Mean Absolute Error (MAE) is the metric to determine how accurate the model is
# We we train a linear regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Create Linear Regression model, fit model, and make predictions

lr = LinearRegression(normalize = True)
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

# Determine mean absolute error
mae = mean_absolute_error(y_test, predictions)
print(mae)

# on average, the prediction miss the target amount by an average of 2.670 wins

# We now try a Ridge regression model

from sklearn.linear_model import RidgeCV

# Create Ridge Linear Regressions model, fit model, and make predictions
rrm = RidgeCV(alphas = (0.01, 0.1, 1.0, 10.0), normalize = True)
rrm.fit(x_train, y_train)
predictions_rrm = rrm.predict(x_test)

# Determine mean absolute error
mae_rrm = mean_absolute_error(y_test, predictions_rrm)
print(mae_rrm)

# on average, the prediction miss the target amount by an average of 2.653 wins
