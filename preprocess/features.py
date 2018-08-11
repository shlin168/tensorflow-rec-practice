from __future__ import print_function
from sklearn.preprocessing import LabelEncoder


class Feature_engineering:
    
    @staticmethod
    def get_numeric_features(df):
        return df.select_dtypes(exclude=['object']).columns
    
    @staticmethod
    def fillna(df, cols, val):
        df[cols] = df.copy()[cols].fillna(-1)
        return df
    
    @staticmethod
    def float2int(df, cols):
        for col in cols:
            df[col] = df.copy()[col].astype(int)
        return df
    
    @staticmethod
    def label_encoder(df, col, new_col):
        le = LabelEncoder()
        le.fit(df[col])
        df_copy = df.copy()
        df_copy.loc[:, new_col] = le.transform(df[col])
        df_copy = Feature_engineering.float2int(df_copy, [new_col])
        return df_copy