import pandas as pd

cities = [[   1,  2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,   1   ], 4722]

print cities


sonar = pd.read_csv('sonar.all-data.csv', header=None)
print sonar.head()

labels = sonar.iloc[:,-1]
data = sonar.iloc[:,:-1]

print labels