#import necessary data analysis/machine learning/plotting libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

#importing the data
easyBlind = pd.read_csv("EASY_BLINDED.csv", index_col=0, header=None)
easyTest = pd.read_csv("EASY_TEST.csv", index_col=None, header=None)
easyTrain = pd.read_csv("EASY_TRAIN.csv", index_col=None, header=None)
modBlind = pd.read_csv("MODERATE_BLINDED.csv", index_col=0, header=None)
modTest = pd.read_csv("MODERATE_TEST.csv", index_col=None, header=None)
modTrain = pd.read_csv("MODERATE_TRAIN.csv", index_col=None, header=None)
difBlind = pd.read_csv("DIFFICULT_BLINDED.csv", index_col=0, header=None)
difTest = pd.read_csv("DIFFICULT_TEST.csv", index_col=None, header=None)
difTrain = pd.read_csv("DIFFICULT_TRAIN.csv", index_col=None, header=None)
data = []
#converting the string value categories into numeric categories
data.append(easyBlind); data.append(easyTest); data.append(easyTrain)
data.append(modBlind); data.append(modTest); data.append(modTrain)
data.append(difBlind); data.append(difTest); data.append(difTrain)
labels = ['Endosomes',
          'Lysosome',
          'Mitochondria',
          'Peroxisomes',
          'Actin',
          'Plasma_Membrane',
          'Microtubules',
          'Endoplasmic_Reticulum']
for i in data:
    for j in range(len(labels)):
        i.replace(labels[j], j, inplace=True)
    i[i.columns[-1]] = i[i.columns[-1]].astype('category')

#input:
#1. mdl, the model to find the best query for
#2. df, the dataframe containing the data at hand
#3. queriedLabels, the list of indicies that have aready been queried
#   so as not to query the same label twice
#
#output:
#1. the index of the best query in the dataframe: Least Confidence Uncertainty Sampling
def getBestQuery(mdl, df, queriedLabels):
    #produces an array of probabilities that each data point lies in each class
    predictions = mdl.predict_proba(X=df.drop(df.columns[-1], axis=1))
    maximum = []
    for i in predictions:
        maximum.append(max(i))
    #gets the index of the lowest maximum likelihood in the predictions
    index = maximum.index(min(maximum))
    while index in queriedLabels:
        maximum[index] = 1
        index = maximum.index(min(maximum))
    return index

#input:
#1. df, a dataframe to build the model on
#2. testdf, the corresponding test dataframe for df
#3. modeltype, sklearn model type to learn on for the active learner
#4. randommodeltype, sklearn model type to learn on for the random learner
#4. getBestQuery, the query function to use
#
#output:
#1. mdl, a random forest classifier which uses active learning to query points to the oracle
#2. randmdl, a random forest classifier which is a random learner
#3. testError, the test error of the active learning model
#4. randError, the test error of the random learner
def getModel(df, testdf, modeltype, randommodeltype):
    #saving the indicies of the data points that have already been queried
    #so as not to query the same data point twice for the active and
    #random models
    queriedLabels = []
    randQueriedLabels = []

    #number of datapoints in the dataframes
    datacount = df.shape[0]
    testcount = testdf.shape[0]

    #test error of the active model and random model
    testError = []
    randError = []

    classes = df.iloc[:,-1].unique

    #the model that will be built on the active learning algorithm
    mdl = modeltype
    #the random learn
    randmdl = randommodeltype

    #begins by querying 100 random data points to build a base learner
    print('BUILDING BASE LEARNER...')
    for i in range(100):
        #getting a random index that has not been queried
        index = random.randint(0, datacount)
        while index in queriedLabels:
            index = random.randint(0, datacount)
        queriedLabels.append(index)
        randQueriedLabels.append(index)
        #learning the random data point and computing the errors for both models
        mdl.fit(X=df.drop(df.columns[-1], axis=1).iloc[queriedLabels], y=df[df.columns[-1]].iloc[queriedLabels])
        randmdl.fit(X=df.drop(df.columns[-1], axis=1).iloc[randQueriedLabels], y=df[df.columns[-1]].iloc[randQueriedLabels])
        mdlPredict = mdl.predict(X=testdf.drop(testdf.columns[-1], axis=1))
        randmdlPredict = randmdl.predict(X=testdf.drop(testdf.columns[-1], axis=1))
        #error is calculated by 1 - (the percentage of correctly labeled points)
        testError.append(1 - sum(mdlPredict == testdf.iloc[:, -1])/testcount)
        randError.append(1 - sum(randmdlPredict == testdf.iloc[:, -1])/testcount)

    #iterating through the remaining budget
    print('PERFORMING ACTIVE LEARNING...')
    for i in range(2400):
        if i % 240 == 0 and i != 0:
            print(str((i/240)*10) + '% COMPLETE')
        #the active learner learns the best query
        index = getBestQuery(mdl, df, queriedLabels)
        queriedLabels.append(index)
        mdl.fit(X=df.drop(df.columns[-1], axis=1).iloc[queriedLabels], y=df[df.columns[-1]].iloc[queriedLabels])
        mdlPredict = mdl.predict(X=testdf.drop(testdf.columns[-1], axis=1))
        #updating test error
        testError.append(1 - sum(mdlPredict == testdf.iloc[:, -1])/testcount)
        #getting a random index for the random learner
        index = random.randint(0, datacount - 1)
        while index in randQueriedLabels:
            index = random.randint(0, datacount - 1)
        randQueriedLabels.append(index)
        randmdl.fit(X=df.drop(df.columns[-1], axis=1).iloc[randQueriedLabels], y=df[df.columns[-1]].iloc[randQueriedLabels])
        randmdlPredict = randmdl.predict(X=testdf.drop(testdf.columns[-1], axis=1))
        #updating random error
        randError.append(1 - sum(randmdlPredict == testdf.iloc[:, -1])/testcount)

    return mdl, randmdl, testError, randError

#running the models
#training the easy difficulty model
[model, randmodel, easytestError, easyrandError] = getModel(easyTrain,
                                                    easyTest,
                                                    RandomForestClassifier(n_estimators=10),
                                                    RandomForestClassifier(n_estimators=10)
                                                    )
easyscore = model.score(X=easyTest.drop(easyTest.columns[-1], axis=1), y=easyTest[easyTest.columns[-1]])
easyrandscore = randmodel.score(X=easyTest.drop(easyTest.columns[-1], axis=1), y=easyTest[easyTest.columns[-1]])
#saving blinded predictions
easypredictions = model.predict(X=easyBlind)
easyblinded = pd.DataFrame(easypredictions, index=easyBlind.index.values)

#training the moderate difficulty model
[model, randmodel, modtestError, modrandError] = getModel(modTrain,
                                                    modTest,
                                                    RandomForestClassifier(n_estimators=10),
                                                    RandomForestClassifier(n_estimators=10)
                                                    )
modscore = model.score(X=modTest.drop(modTest.columns[-1], axis=1), y=modTest[modTest.columns[-1]])
modrandscore = randmodel.score(X=modTest.drop(modTest.columns[-1], axis=1), y=modTest[modTest.columns[-1]])
#saving blinded predictions
modpredictions = model.predict(X=modBlind)
modblinded = pd.DataFrame(modpredictions, index=modBlind.index.values)

#training the difficult model
[model, randmodel, diftestError, difrandError] = getModel(difTrain,
                                                    difTest,
                                                    RandomForestClassifier(n_estimators=10),
                                                    RandomForestClassifier(n_estimators=10)
                                                    )
difscore = model.score(X=difTest.drop(difTest.columns[-1], axis=1), y=difTest[difTest.columns[-1]])
difrandscore = randmodel.score(X=difTest.drop(difTest.columns[-1], axis=1), y=difTest[difTest.columns[-1]])
#saving blinded predictions
difpredictions = model.predict(X=difBlind)
difblinded = pd.DataFrame(difpredictions, index=difBlind.index.values)

#converting integer predictions to classification strings
for j in range(len(labels)):
    easyblinded.replace(j, labels[j], inplace=True)
    modblinded.replace(j, labels[j], inplace=True)
    difblinded.replace(j, labels[j], inplace=True)

#saving blinded predictions to text files
easyblinded.to_csv('EASY_BLINDED.txt')
modblinded.to_csv('MODERATE_BLINDED.txt')
difblinded.to_csv('DIFFICULT_BLINDED.txt')

#printing accuracy metrics
print('easy: ' + str(easyscore))
print('easyrand: ' + str(easyrandscore))
print('mod: ' + str(modscore))
print('modrand: ' + str(modrandscore))
print('dif: ' + str(difscore))
print('difrand: ' + str(difrandscore))

#plotting errors
plt.plot(easytestError)
plt.plot(easyrandError)
plt.legend(['Active Learner', 'Random Learner'])
plt.xlabel('Queries')
plt.ylabel('Error')
plt.title('Easy/Random Forest')
plt.show()
plt.plot(modtestError)
plt.plot(modrandError)
plt.legend(['Active Learner', 'Random Learner'])
plt.xlabel('Queries')
plt.ylabel('Error')
plt.title('Moderate/Random Forest')
plt.show()
plt.plot(diftestError)
plt.plot(difrandError)
plt.legend(['Active Learner', 'Random Learner'])
plt.xlabel('Queries')
plt.ylabel('Error')
plt.title('Difficult/Random Forest')
plt.show()