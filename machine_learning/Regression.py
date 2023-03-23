from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import os

class Binary_Logistic_Regressor:

  def __init__(
    self, main_gradient_descent='mini_batch', regularizer= 'l2', 
    hyper_parameters_assign='auto-assign', hyper_parameters_tuning_gradient_descent = 'batch',
    max_iter = 1000, early_stopping=True, lamda_value = None, lr_value = None, 
    monitor='val_error', paitiance = 3, error_roundoff = 8, acc_roundoff = 4, 
    acc_change=0.001, error_change=0.0000001, verbose=False):

    self.main_gradient_descent=main_gradient_descent
    self.weights = None
    self. hyper_parameters_assign = hyper_parameters_assign
    self.hyper_paremeters_tuning_gradient_descent = hyper_parameters_tuning_gradient_descent
    self.lamda = lamda_value
    self.learning_rate = lr_value
    self.regularizer = regularizer
    self.verbose = verbose
    self.paitiance = paitiance
    self.max_iter = max_iter
    self.early_stopping = early_stopping
    self.no_of_features = None
    self.error_train = list()
    self.error_validation = list()
    self.val_accuracyList = list()
    self.monitor=monitor
    self.final_error = None
    self.acc_change = acc_change
    self.error_change = error_change
    self.error_roundoff = error_roundoff
    self.acc_roundoff = acc_roundoff
    self.score = None
    self.classification_report = None
    self.confusion_matrix = None


  def lowLevelFunction(self, trainingX, trainingY, testingX, testingY):
    
    def resetInitialVariables():
      self.weights = InitializeWeights()
      self.error_train = list()
      self.error_validation = list()
      self.final_error = None
      self.val_accuracyList = list()
      return self

    def addColumnforBias(trainX):
      LogesticX = pd.DataFrame(trainX)
      LogesticX.insert(loc=0, column='X0', value = 1)
      logesticNumpy = LogesticX.to_numpy()
      return logesticNumpy
    
    def hyperParemTuning(
    trainX, trainY, kfold_value=5,showLogs = False):

      if self.regularizer == 'l2' or self.regularizer == 'l1':
        if showLogs==True:
          print('\n', '*'*80, '\n\tPLEASE WAIT......Preforming Tuning of Hyperparameters....', '\n','*'*80)
        dataset = np.c_[trainX, trainY]
        InstancesIn_Fold = len(dataset)//kfold_value

        Fold1 = dataset[:InstancesIn_Fold, :]
        Fold2 = dataset[InstancesIn_Fold : InstancesIn_Fold*2 , :]
        Fold3 = dataset[InstancesIn_Fold*2 : InstancesIn_Fold*3 , :]
        Fold4 = dataset[InstancesIn_Fold*3 : InstancesIn_Fold*4 , :]
        Fold5 = dataset[InstancesIn_Fold*4 : InstancesIn_Fold*5, :]

        Fold_List = [Fold1, Fold2, Fold3, Fold4, Fold5]
        lamda_AvgFoldscore = list()
        lamda_list = list()
        lr_list = list()
        lamda_weights = np.copy(self.weights)
        for lr in [1e-3, 1e-2, 1e-1]:
          self.learning_rate = lr
          for i in [1e-3, 1e-2, 1e-1, 1, 10]:
            self.lamda = i
            temp_holdoutScore = list()
            for c in range(kfold_value):
              self.weights = np.copy(lamda_weights)
              TrainingData =  np.vstack(tuple([x for x in Fold_List if True in (np.where(x != Fold_List[c], True, False))]))
              TestingData = Fold_List[c]
              trX, trY, teX, teY = TrainingData[:,:-1], TrainingData[:,-1], TestingData[:,:-1], TestingData[:,-1]
              if self.hyper_paremeters_tuning_gradient_descent == 'batch':
                  Batch_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              elif self.hyper_paremeters_tuning_gradient_descent == 'stochastic':
                  Stochastic_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              else:
                  MiniBatch_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              temp_holdoutScore.append(self.final_error)
              resetInitialVariables()
      
            lamda_list.append(i)
            lamda_AvgFoldscore.append(np.mean(temp_holdoutScore))
            lr_list.append(lr)  
        resetInitialVariables()
        smallestIndex = np.argmin(lamda_AvgFoldscore)
        self.lamda = lamda_list[smallestIndex]
        self.learning_rate = lr_list[smallestIndex]
      else:
        if showLogs==True:
          print('\n', '*'*80, '\n\tPLEASE WAIT......Preforming Tuning of Hyperparameters....', '\n','*'*80)
        dataset = np.c_[trainX, trainY]
        InstancesIn_Fold = len(dataset)//kfold_value

        Fold1 = dataset[:InstancesIn_Fold, :]
        Fold2 = dataset[InstancesIn_Fold : InstancesIn_Fold*2 , :]
        Fold3 = dataset[InstancesIn_Fold*2 : InstancesIn_Fold*3 , :]
        Fold4 = dataset[InstancesIn_Fold*3 : InstancesIn_Fold*4 , :]
        Fold5 = dataset[InstancesIn_Fold*4 : InstancesIn_Fold*5, :]

        Fold_List = [Fold1, Fold2, Fold3, Fold4, Fold5]
        lr_AvgFoldscore = list()
        lr_list = list()
        lamda_weights = np.copy(self.weights)
        for i in [1e-3, 1e-2, 1e-1]:
            self.learning_rate = i
            temp_holdoutScore = list()
            for c in range(kfold_value):
              self.weights = np.copy(lamda_weights)
              TrainingData =  np.vstack(tuple([x for x in Fold_List if True in (np.where(x != Fold_List[c], True, False))]))
              TestingData = Fold_List[c]
              trX, trY, teX, teY = TrainingData[:,:-1], TrainingData[:,-1], TestingData[:,:-1], TestingData[:,-1]
              if self.hyper_paremeters_tuning_gradient_descent == 'batch':
                  Batch_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              elif self.hyper_paremeters_tuning_gradient_descent == 'stochastic':
                  Stochastic_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              else:
                  MiniBatch_GD(trX, trY, teX, teY, monitor='val_error', kfoldInProgress=True)
              temp_holdoutScore.append(self.final_error)
              resetInitialVariables()

            lr_AvgFoldscore.append(np.mean(temp_holdoutScore))
            lr_list.append(i)  
        resetInitialVariables()
        smallestIndex = np.argmin(lr_AvgFoldscore)
        self.learning_rate = lr_list[smallestIndex]
      
      return self

    def earlyStopping(
    val, best, b_weights, count , pait, monitor, c, kfoldInProgess):

      if c > 0:
        if monitor == 'val_error':
          if val < best:
            self.final_error = val
            b_weights = np.copy(self.weights)
            best = val
            count = 0
          else:
            count+=1
          if count > 15 or (self.error_validation[-1] - self.error_validation[-2]) < self.error_change:
            pait+=1
            if pait > self.paitiance:
              self.weights = np.copy(b_weights)
              if kfoldInProgess == False and self.verbose==True:
                print('****************BEST WEIGHTS ON WHICH HIGHEST ACCURACY ACHIEVED - RESTORIED******************')
              if self.final_error == None:
                self.final_error = val
              return True, best, b_weights, count, pait
            else:
              return False, best, b_weights, count, pait
          else:
            pait = 0
            return False, best, b_weights, count, pait
        else:
          if val > best:
            b_weights = np.copy(self.weights)
            best = val
            count = 0
          else:
            count+=1
          if count > 15 or (self.val_accuracyList[-1] - self.val_accuracyList[-2]) < self.acc_change:
            pait+=1
            if pait > 3:
              self.weights = np.copy(b_weights)
              if kfoldInProgess == False and self.verbose==True:
                print('****************BEST WEIGHTS ON WHICH HIGHEST ACCURACY ACHIEVED - RESTORIED******************')
              if self.final_error == None:
                self.final_error = val
              return True, best, b_weights, count, pait
            else:
              return False, best, b_weights, count, pait
          else:
            pait = 0
            return False, best, b_weights, count, pait
      else:
        return False, best, self.weights, count, pait
    
    def InitializeWeights():
      temp = list()
      for i in range(self.no_of_features):
        temp.append(random.uniform(0.001, 0.1))
      return np.array(temp)

    def Pridction(x, getLabel = False):
      if getLabel == True:
        sig = 1/(1 + np.exp(- np.matmul(x, self.weights)))
        sig = np.minimum(sig, 0.9999)
        sig = np.maximum(sig, 0.0001)
        return self.convert_predictions_to_labels(sig)
      else:
        sig = 1/(1 + np.exp(- np.matmul(x, self.weights)))
        sig = np.minimum(sig, 0.9999)
        return np.maximum(sig, 0.0001)
        
    def Generic_BinaryCrossEntropy(pridicted, actual, TotalnoOfInstance=None):
      noOfInstance = actual.shape[0]
      if TotalnoOfInstance == None:
        TotalnoOfInstance = noOfInstance
      tempTheta = np.copy(self.weights)
      tempTheta[0] = 0
      if self.regularizer == 'l2':
        return (np.sum(-((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted))))/noOfInstance) + \
                     (self.lamda/(2*TotalnoOfInstance))*np.matmul(tempTheta.transpose(), tempTheta)
      elif self.regularizer == 'l1':
        return (np.sum(-((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted))))/noOfInstance) + \
                     (self.lamda/TotalnoOfInstance)*np.absolute(tempTheta).sum()
      else:
        return np.sum(-((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted))))/noOfInstance
  
    def Generic_DerivateWeights(pridcted, actual, actualX, TotalnoOfInstance=None):
      noOfInstance = actualX.shape[0]
      if TotalnoOfInstance == None:
        TotalnoOfInstance = noOfInstance
      tempTheta = np.copy(self.weights)
      tempTheta[0] = 0
      if self.regularizer == 'l2':
        return (np.matmul(actualX.transpose(), (pridcted-actual))/noOfInstance) + \
                    (self.lamda/TotalnoOfInstance)*tempTheta

      elif self.regularizer == 'l1':
        return (np.matmul(actualX.transpose(), (pridcted-actual))/noOfInstance) + \
                    (self.lamda/TotalnoOfInstance)*np.sign(tempTheta)

      else:
        return (np.matmul(actualX.transpose(), (pridcted-actual))/noOfInstance)

    def Generic_UpdateWeights(derivate):
      new_weights = list()
      for i in range(len(derivate)):
        new_weights.append((self.weights[i]-derivate[i]*self.learning_rate))
      self.weights = np.array(new_weights)
    
    def Batch_GD(
      trainX, trainY, testX, testY, monitor, kfoldInProgress=False):

      if monitor == 'val_error':
        best_error = 0
      else:
        best_accuracy = 0

      Patience = 0
      best_weights = np.copy(self.weights)
      bestUnchangedCount = 0

      for c in range(self.max_iter):
        if kfoldInProgress == False and self.verbose==True:
          print(f"Epoch {c+1}:\n")
        hTrain = Pridction(trainX)
        JTrain = np.round(Generic_BinaryCrossEntropy(hTrain, trainY), self.error_roundoff)
        derivate_weights = Generic_DerivateWeights(hTrain, trainY, trainX)
        Generic_UpdateWeights(derivate_weights)

        self.error_train.append(JTrain)
        validation = Pridction(testX)
        validationError = np.round(Generic_BinaryCrossEntropy(validation, testY),self.error_roundoff)
        validation = self.convert_predictions_to_labels(validation)
        validation_acc = self.accuracy_score(testY, validation)
        self.error_validation.append(validationError)
        self.val_accuracyList.append(validation_acc)

        if kfoldInProgress == False and self.verbose == True:
          print("Traning Error : ", end = " ")
          print(f"{self.error_train[c]}   ||   Validation Error : {validationError}    Validation Accuracy : {validation_acc}  ")
          print()
          print("*"*100)

        if kfoldInProgress == True or self.early_stopping==True:
          if monitor == 'val_error':
            loopBreaker, best_error, best_weights, bestUnchangedCount, Patience = earlyStopping(
                validationError, best_error, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
            if(loopBreaker == True):
              break
          else:
            loopBreaker, best_accuracy, best_weights, bestUnchangedCount, Patience = earlyStopping(
                validation_acc, best_accuracy, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
            if(loopBreaker == True):
              break

      return self  

    def Stochastic_GD(
    trainX, trainY, testX, testY, monitor, kfoldInProgress=False):

      def Stochastic_BinaryCrossEntropy(pridicted, actual, TotalnoOfInstance):
        tempTheta = np.copy(self.weights)
        tempTheta[0] = 0
        if self.regularizer == 'l2':
          return (-((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted)))) + \
                        (self.lamda/(2*TotalnoOfInstance))*np.matmul(tempTheta.transpose(), tempTheta)
        elif self.regularizer == 'l1':
          return (-((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted)))) + \
                        (self.lamda/(2*TotalnoOfInstance))*np.absolute(tempTheta).sum()
        else:
          return -((actual * np.log2(pridicted)) + ((1 - actual) * np.log2(1-pridicted)))

      def Stochastic_DerivateWeights(pridicted, actual, actualX, TotalnoOfInstance):
        tempTheta = np.copy(self.weights)
        tempTheta[0] = 0
        if self.regularizer == 'l2':
          return actualX * (pridicted-actual) + (self.lamda/TotalnoOfInstance)*tempTheta
        elif self.regularizer == 'l1':
          return actualX * (pridicted-actual) + (self.lamda/TotalnoOfInstance)*np.sign(tempTheta)
        else:
          return actualX * (pridicted-actual)

      if monitor == 'val_error':
          best_error = 0
      else:
          best_accuracy = 0

      Patience = 0
      best_weights = np.copy(self.weights)
      bestUnchangedCount = 0

      for c in range(self.max_iter):
        if kfoldInProgress == False and self.verbose==True:
          print(f"Epoch {c+1}:\n")

        tempTrainError = list()
        for r in range(len(trainX)):
          hTrain = Pridction(trainX[r,:])
          JTrain = np.round(Stochastic_BinaryCrossEntropy(hTrain, trainY[r], len(trainX)),self.error_roundoff)
          tempTrainError.append(JTrain)
          derivate_weights = Stochastic_DerivateWeights(hTrain, trainY[r], trainX[r,:], len(trainX))
          Generic_UpdateWeights(derivate_weights)

      
        self.error_train.append(np.mean(tempTrainError))
        validation = Pridction(testX)
        validationError = np.round(Generic_BinaryCrossEntropy(validation, testY),self.error_roundoff)
        validation = self.convert_predictions_to_labels(validation)
        validation_acc = self.accuracy_score(testY, validation)
        self.error_validation.append(validationError)
        self.val_accuracyList.append(validation_acc)
      

        if kfoldInProgress == False and self.verbose ==True:
          print(end=f"{r}/{len(trainX)-1} :   Traning Error : ")
          print(f"{self.error_train[c]}   ||   Validation Error : {validationError}    Validation Accuracy : {validation_acc}  ")
          print()
          print("*"*100)
        
        if kfoldInProgress == True or self.early_stopping==True:
          if monitor == 'val_error':
              loopBreaker, best_error, best_weights, bestUnchangedCount, Patience = earlyStopping(
                  validationError, best_error, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
              if(loopBreaker == True):
                break
          else:
              loopBreaker, best_accuracy, best_weights, bestUnchangedCount, Patience = earlyStopping(
                  validation_acc, best_accuracy, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
              if(loopBreaker == True):
                break
      
      return self

    def MiniBatch_GD(
      trainX, trainY, testX, testY, monitor,batch_size=32, kfoldInProgress= False):
      
      if monitor == 'val_error':
          best_error = 0
      else:
          best_accuracy = 0

      Patience = 0
      best_weights = np.copy(self.weights)
      bestUnchangedCount = 0
      No_ofDataPerEpoch = len(trainX)//batch_size

      for c in range(self.max_iter):
        if kfoldInProgress == False and self.verbose==True:
          print(f"Epoch {c+1}:\n")

        batch_start = 0
        batch_end = 0
        tempTrainError = list()

        for r in range(No_ofDataPerEpoch):
          batch_start = batch_end
          batch_end = batch_start + batch_size
          hTrain = Pridction(trainX[batch_start:batch_end,:])
          JTrain = np.round(Generic_BinaryCrossEntropy(hTrain, trainY[batch_start:batch_end], trainX.shape[0]),self.error_roundoff)
          tempTrainError.append(JTrain)
          derivate_weights = Generic_DerivateWeights(hTrain, trainY[batch_start:batch_end], trainX[batch_start:batch_end,:], trainX.shape[0])
          Generic_UpdateWeights(derivate_weights)

        self.error_train.append(np.mean(tempTrainError))
        validation = Pridction(testX)
        validationError = np.round(Generic_BinaryCrossEntropy(validation, testY),self.error_roundoff)
        validation = self.convert_predictions_to_labels(validation)
        validation_acc = self.accuracy_score(testY, validation)
        self.error_validation.append(validationError)
        self.val_accuracyList.append(validation_acc)

        if kfoldInProgress == False and self.verbose==True:
          print(end="Traning Error : ")
          print(f"{self.error_train[c]}   ||   Validation Error : {validationError}    Validation Accuracy : {validation_acc}  ")
          print()
          print("*"*100)
        
        if kfoldInProgress == True or self.early_stopping==True:
          if monitor == 'val_error':
              loopBreaker, best_error, best_weights, bestUnchangedCount, Patience = earlyStopping(
                  validationError, best_error, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
              if(loopBreaker == True):
                break
          else:
              loopBreaker, best_accuracy, best_weights, bestUnchangedCount, Patience = earlyStopping(
                  validation_acc, best_accuracy, best_weights,bestUnchangedCount, Patience,monitor, c, kfoldInProgress)
              if(loopBreaker == True):
                break
      
      return self
    
    trainingX = addColumnforBias(trainingX)
    testingX = addColumnforBias(testingX)
    self.no_of_features = trainingX.shape[1]
    self.weights = InitializeWeights()
    hyperParemTuning(trainingX, trainingY, showLogs=self.verbose)
    if self.main_gradient_descent == 'batch':
      Batch_GD(trainingX, trainingY, testingX, testingY, monitor=self.monitor)
    elif self.main_gradient_descent == 'stochastic':
      Stochastic_GD(trainingX, trainingY, testingX, testingY, monitor=self.monitor)
    else:
      MiniBatch_GD(trainingX, trainingY, testingX, testingY, batch_size=32, monitor=self.monitor)

    return self

  
  def trainTestSplit(X, Y):
    return train_test_split(X, Y, test_size=0.2, random_state=1234)
  
  def fit(self, trainX, trainY, testX, testY):
    self.lowLevelFunction(trainX, trainY, testX, testY)
    predictions = self.predict(testX)
    self.score = self.accuracy_score(testY, predictions)
    self.classification_report = self.classification_report_dataframe(testY, predictions)
    self.confusion_matrix = self.confusion_matrix_f(testY, predictions)
    return self


  def predict(self, x):
      if x.shape[1] < self.weights.shape[0]:
        LogesticX = pd.DataFrame(x)
        LogesticX.insert(loc=0, column='X0', value = 1)
        x = LogesticX.to_numpy()
      sig = 1/(1 + np.exp(- np.matmul(x, self.weights)))
      sig = np.minimum(sig, 0.9999)
      sig = np.maximum(sig, 0.0001)
      return self.convert_predictions_to_labels(sig)

  def accuracy_score(self, actual, pridicted):
    return np.round(accuracy_score(actual, pridicted), self.acc_roundoff)
  
  def classification_report_dataframe(self, actual, pridicted):
    return classification_report(actual, pridicted)
  
  def confusion_matrix_f(self, actual, pridicted):
    return confusion_matrix(actual, pridicted)

  def convert_predictions_to_labels(self, pridicted):
    return np.where(pridicted>0.5, 1, 0)

  def saveModel(self, path, model=None):
    if model==None:
      model = self
    if not os.path.isdir(path):
      os.mkdir(path)

    with open(path +'modelParem.txt', 'w') as f:
      count = 0
      for w in model.weights:
        if count == 0:
            f.write(str(w))
            count+=1
        else:
            f.write(','+str(w))
    f.close()
    with open(path +'modelParem.txt', 'a') as f:
      f.write('\n'+str(model.lamda))
      f.write('\n'+str(model.learning_rate))
    f.close()
    return 'Succesfully Saved Model!!'
   
  def loadModel(path):
    if os.path.isfile(path + 'modelParem.txt'):
      temp = open(path + 'modelParem.txt', 'r').read().splitlines()
      t = temp[0].split(',')
      w = [float(c) for c in t]
      l = float(temp[1])
      lr = float(temp[2])
      model = Binary_Logistic_Regressor()
      model.weights = np.array(w)
      model.lamda = l
      model. learning_rate = lr
      return model
    else:
      raise ValueError("Invalid path, file doesn't exit at this path")

  
