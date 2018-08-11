import pandas as pd

class Data:

    @staticmethod
    def read_data(file_path):
        return pd.read_csv(file_path, low_memory=False)
    
    @staticmethod
    def get_train_test_data(df):
        return df[df['partition'] != '2016-06-28'], df[df['partition'] == '2016-06-28']