import numpy as np
import tensorflow as tf

class FeatureColumnsBuilder(object):
    
    @staticmethod
    def get_numeric_cols(df, cols, normalize=False):
        def zscore(col):
            return (col - mean) / std

        tf_numeric_cols = list()
        if normalize:     
            for col in cols:
                mean, std = np.mean(df[col].values), np.std(df[col].values)
                normalizer_fn = zscore
                tf_numeric_cols.append(tf.feature_column.numeric_column(key=col, normalizer_fn=normalizer_fn))
        else:
            tf_numeric_cols = [tf.feature_column.numeric_column(key=col) for col in cols]
        return tf_numeric_cols
    
    @staticmethod
    def get_bucket_col(col, boundaries):
        return tf.feature_column.indicator_column(
            tf.feature_column.bucketized_column(tf.feature_column.numeric_column(key='age'), boundaries=boundaries)
        )
    
    @staticmethod
    def get_embeding_col(df, col, dim):
        cate_col = tf.feature_column.categorical_column_with_vocabulary_list(col, df[col].unique())
        return tf.feature_column.embedding_column(cate_col, dim)
    
    @staticmethod
    def get_all(df, feature_dict):
        tf_feature_columns = list()
        tf_feature_columns.extend(FeatureColumnsBuilder.get_numeric_cols(df, feature_dict['num']))
        tf_feature_columns.extend(FeatureColumnsBuilder.get_numeric_cols(df, feature_dict['norm_num'], normalize=True))
        for col, bucket in feature_dict['bucket']:
            tf_feature_columns.append(FeatureColumnsBuilder.get_bucket_col(col, bucket))
        for col, dim in feature_dict['embeddings']:
            tf_feature_columns.append(FeatureColumnsBuilder.get_embeding_col(df, col, dim=dim))
        
        return tf_feature_columns