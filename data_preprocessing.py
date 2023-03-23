import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class DataPrepocessing:
    def _init_(self):
        self.dataset = None
    
    def dataCleaning(self, dataset):
        y_col = dataset.columns[-1]
        y = dataset[y_col]
        X = dataset.drop(columns=[y_col])

        numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

        num_col_index = [list(X.columns).index(x) for x in numerical_ix]
        cat_col_index = [list(X.columns).index(x) for x in categorical_ix]

        imputing_transformer = [('num', SimpleImputer(strategy='median'), num_col_index), ('cat', SimpleImputer(strategy='most_frequent'), cat_col_index)]
        encoding_transformer = [('one_hot_encoder', OneHotEncoder(), cat_col_index)]
        data_transformer = ColumnTransformer(transformers=imputing_transformer+encoding_transformer)

        def MeanNormalization(a):
            return ((a-np.mean(a))/np.std(a))

        X = data_transformer.fit_transform(X)
        ls = list(range(len(numerical_ix), len(numerical_ix) + len(categorical_ix)))
        df = pd.DataFrame(X)
        df.drop(columns=df.columns[ls], inplace= True)
        for i in df.columns:
            if (np.max(df[i]) - np.min(df[i])) > 50:
                df[i] = MeanNormalization(df[i])
        
        X = df.to_numpy()
        y = self.LabelEncodeOutput(y)
    
        return X.astype(float), y
    
    def SeparateCatNum(self, X):
        """ To be used in Naive Bayes """
        cat_x = X
        num_x = X
        return cat_x, num_x

    def LabelEncodeOutput(self, outputColumn):
        return LabelEncoder().fit_transform(outputColumn)