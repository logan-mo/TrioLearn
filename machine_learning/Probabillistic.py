import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from ..data_preprocessing import DataPrepocessing

class NumericNB:
  def fit(self, train_data):
    self.columns = train_data.columns
    self.means = train_data.groupby(by='class').mean().to_dict(orient='index')
    self.standard_deviation = train_data.groupby(by='class').std().to_dict(orient='index')
    self.prior = (train_data['class'].value_counts() / len(train_data['class'])).to_dict()
    self.labels = list(train_data['class'].unique())

  def probability(self, x, mean, std):
    denominator = np.sqrt(2*np.pi) * std
    numerator = np.exp(-((x-mean)**2 / (2*(std**2))))
    return numerator / denominator

  def predict(self, X_test):
    y_pred = list()

    inputCols = self.columns[:-1]
    for i in range(len(X_test)):
        outputDict = dict()
        for output in self.labels:
            outputProbTemp = 1
            for feature in inputCols:
                outputProbTemp *= self.probability((X_test.iloc[i])[feature], (self.means[output])[feature], (self.standard_deviation[output])[feature])
            outputDict[output] = outputProbTemp*self.prior[output]
        y_pred.append(max(outputDict, key=outputDict.get))

    return y_pred

  def predict_prob(self, X_test):
    y_pred = list()

    inputCols = self.columns[:-1]
    for i in range(len(X_test)):
        outputDict = dict()
        for output in self.labels:
            outputProbTemp = 1
            for feature in inputCols:
                outputProbTemp *= self.probability((X_test.iloc[i])[feature], (self.means[output])[feature], (self.standard_deviation[output])[feature])
            outputDict[output] = outputProbTemp*self.prior[output]
        y_pred.append(list(outputDict.values()))

    return np.array(y_pred)


class CategoricNB:

    def __init__(self):
        self.classFrequency = {}  
        self.categoryFrequency = {} 
        self.priorProbabilities = {}
        self.posteriorProbabilities = {}
        self.uniqueValuesPerFeature = {}
        self.unique_labels = {}
        self.alpha = 0.5

    def fit(self, TrainData):
        self.unique_label, counts = np.unique(TrainData.iloc[:,-1], return_counts=True)
        d = dict(zip(self.unique_label, counts))
        for self.unique_label in d:
            counts = d[self.unique_label]
            self.classFrequency[self.unique_label] = counts 
            self.priorProbabilities[self.unique_label] = (counts) / (len(TrainData)) 
        
        for i in range(len(TrainData)):
          for j in range(TrainData.shape[1]-1):
            feature = j 
            category = TrainData.loc[i,j]
            label = TrainData.iat[i,TrainData.shape[1]-1]
            key = (feature,category,label)
            if key in self.categoryFrequency:
                self.categoryFrequency[key] += 1
            else:
                self.categoryFrequency[key] = 1
                    

        for col in range(0,TrainData.shape[1]-1):
            self.uniqueValuesPerFeature[col] = len(set(TrainData.loc[:,col]))
        
        self.unique_labels = np.unique(TrainData.iloc[:,-1])
        for key in self.categoryFrequency.keys():
            for label in self.unique_labels:
                if label == key[2]: 
                    self.posteriorProbabilities[key] = (self.categoryFrequency[key] + self.alpha)  / (self.classFrequency[key[2]]+ self.uniqueValuesPerFeature[key[0]])                 
                elif (key[0],key[1],label) not in self.posteriorProbabilities.keys(): 
                    self.posteriorProbabilities[(key[0],key[1],label)] = (0 + self.alpha)  / (self.classFrequency[label]+ self.uniqueValuesPerFeature[key[0]]) 
                   
                    
    def predict(self, TestData):
        predictedProbabilities = [] 
        for i in range(len(TestData)):
            rowProbability = np.zeros(len(self.unique_labels))
            for j in range(TestData.shape[1]):
                feature = j
                category = TestData.iat[i,j]
                for k in range(0,len(self.unique_labels)):
                    key = (feature,category,self.unique_labels[k])
                    posterior_prob = self.posteriorProbabilities[key]
                    if j == 0:
                        rowProbability[k] = posterior_prob * self.priorProbabilities[self.unique_labels[k]]
                    else:
                        rowProbability[k] *= posterior_prob
            predictedProbabilities.append(rowProbability)   
        predictedProbabilities = np.array(predictedProbabilities)
        return self.labelPrediction(predictedProbabilities)
            
    def labelPrediction(self,predictedProbabilities):
        maximumIndex = np.argmax(predictedProbabilities,axis=1) 
        predicted_labels = np.array(self.unique_labels[maximumIndex])
        return predicted_labels

    def predict_prob(self, TestData):
        predictedProbabilities = [] 
        for i in range(len(TestData)):
            rowProbability = np.zeros(len(self.unique_labels))
            for j in range(TestData.shape[1]):
                feature = j
                category = TestData.iat[i,j]
                for k in range(0,len(self.unique_labels)):
                    key = (feature,category,self.unique_labels[k])
                    posterior_prob = self.posteriorProbabilities[key]
                    if j == 0:
                        rowProbability[k] = posterior_prob * self.priorProbabilities[self.unique_labels[k]]
                    else:
                        rowProbability[k] *= posterior_prob
            
            predictedProbabilities.append(rowProbability)
        predictedProbabilities = np.array(predictedProbabilities)
        return predictedProbabilities



class NaiveBayes:
    def __init__(self):
        self.CatNB = CategoricNB()
        self.NumNB = NumericNB()

    def DataPreProcessing(self, data):
        y_col = data.columns[-1]
        y = data[y_col]
        X = data.drop(columns=[y_col])

        self.numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_ix = X.select_dtypes(include=['object', 'bool']).columns
        
        num_data = X[self.numerical_ix]
        cat_data = X[self.categorical_ix]

        preprocessor = DataPrepocessing()
        num_X, num_y = preprocessor.dataCleaning(pd.concat([num_data, data.iloc[:,-1]], ignore_index=True, axis=1))
        cat_X, cat_y = preprocessor.dataCleaning(pd.concat([cat_data, data.iloc[:,-1]], ignore_index=True, axis=1))

        n_cols = num_X.shape[1]
        columns = [x for x in list(np.arange(n_cols))] +['class']
        num_train_data = pd.DataFrame(data=np.concatenate((num_X, num_y.reshape(-1,1)), axis=1), columns=columns)

        n_cols = cat_X.shape[1]
        columns = [x for x in list(np.arange(n_cols))] +['class']
        cat_train_data = pd.DataFrame(data=np.concatenate((cat_X, cat_y.reshape(-1,1)), axis=1), columns=columns)
        
        return num_train_data, cat_train_data
        

    def fit(self, training_data):
        num_train_data, cat_train_data = self.DataPreProcessing(training_data)
        self.CatNB.fit(cat_train_data)
        self.NumNB.fit(num_train_data)

    def predict(self, test_data):
        num_test_data, cat_test_data = self.DataPreProcessing(test_data)
        y_pred_num = self.NumNB.predict_prob(num_test_data.iloc[:,:-1])
        y_pred_cat = self.CatNB.predict_prob(cat_test_data.iloc[:,:-1])
        y_pred = self.CatNB.labelPrediction(y_pred_num * y_pred_cat)
        return  y_pred
    
    def predict_prob(self, test_data):
        num_test_data, cat_test_data = self.DataPreProcessing(test_data)
        y_pred_num = self.NumNB.predict_prob(num_test_data.iloc[:,:-1])
        y_pred_cat = self.CatNB.predict_prob(cat_test_data.iloc[:,:-1])
        
        y_pred = y_pred_num * y_pred_cat
        return  y_pred

    def score(self, actual, X_test):
        y_pred = self.predict(X_test)
        if actual.dtype == 'object' or actual.dtype == 'bool':
            preprocessor = DataPrepocessing()
            actual = preprocessor.LabelEncodeOutput(actual)
        return f1_score(actual, y_pred)