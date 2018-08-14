import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

class MyTranformer:
    
    @staticmethod
    def select_col(X, cols=None):
        return X[cols]
    
    @staticmethod
    def select_single_col(X, col):
        return X[col]
    
    @staticmethod
    def drop_col(X, drop_list):
        return X.drop(drop_list, axis = 1)
    
    @staticmethod
    def fill_na(X, val):
        return X.fillna(val)
    
    @staticmethod
    def rename_col(X, col_map):
        return X.rename(columns=col_map)
    
    @staticmethod
    def split_n_get_first_item(series):
        def check_nan(v):
            if type(v) == float:
                if np.isnan(v):
                    return True
            return False
        return series.apply(lambda v: v.split()[0] if not check_nan(v) else v)
    
    @staticmethod
    def str2float(series):
        return series.astype('float')
    
    @staticmethod
    def NA2na(series, val):
        return series.apply(lambda v: np.nan if v == val else v)
    
    @staticmethod
    def deal_specific_val(series, input_val, output_val):
        return series.apply(lambda v: output_val if v == input_val else v)
    
    @staticmethod
    def trim_if_string(series):
        return series.apply(lambda v: v.strip() if isinstance(v, str) else v)
    
    @staticmethod
    def to_frame(series):
        return series.to_frame()
    
    @staticmethod
    def get_month(series):
        return series.apply(lambda v: int(datetime.strptime(v, '%Y-%m-%d').strftime('%m')))


class MultiColumnLabelEncoder:
    '''
        Transforms columns of X specified in self.columns using LabelEncoder(). 
        If no columns specified, transforms all columns in X.
        Not transform np.nan value
    '''
    
    def __init__(self, multi_trans_map=None, columns=None):
        # array of column names to encode
        self.columns = columns
        self.multi_trans_map = multi_trans_map
    
    def fit(self, X, y=None): 
        if self.columns is None:
            self.columns = X.columns
        if self.multi_trans_map is None:
            self.multi_trans_map = dict(zip(self.columns, [None]*len(self.columns)))
            for column in self.multi_trans_map:
                le = LabelEncoder()
                out = le.fit(X[column].dropna())
                self.multi_trans_map[column] = dict(zip(out.classes_, le.transform(out.classes_)))
        return self

    def transform(self, X):
        def get_val(col, val):
            return self.multi_trans_map[col][val] if val in self.multi_trans_map[col] else val
        
        out = X.copy()
        for column in self.columns:
            out[column] = out[column].apply(lambda rows: get_val(column, rows))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
