from __future__ import print_function
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
from datetime import timedelta
from sklearn.model_selection import train_test_split

from preprocess.data import Data
from preprocess.features import Feature_engineering
from tf_model.tf_feature_builder import FeatureColumnsBuilder
from tf_model.model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def main():
    start_time = time.time()
    print('read data ...')
    data = Data.read_data('rec_data.csv')

    print('feature engineering ...')
    features = Feature_engineering.get_numeric_features(data)
    label = 'product_code'
    float2int_list = ['employee_idx', 'country_residence', 'sex', 'relation_type', 
                        'residence_idx', 'foreigner_idx', 'spouse_idx', 'channel', 
                        'deceased_idx', 'province_name', 'type', 'new_customer',
                        'primary', 'address_type','province_code','activity_idx', 
                        'segmentation', 'age', 'seniority']
    data = Feature_engineering.fillna(data, features, -1)
    data = Feature_engineering.float2int(data, float2int_list)
    train_data, _ = Data.get_train_test_data(data)
    train_data = Feature_engineering.label_encoder(train_data, 'product', 'product_code')

    feature_dict = {
        'num': ['foreigner_idx',  'spouse_idx', 'deceased_idx', 'new_customer',
                'primary', 'address_type','province_code','activity_idx'],
        'norm_num': ['age', 'seniority', 'gross_income_household'],
        'bucket': [('age', list(range(0,170,10)))],
        'embeddings': [('channel', 10), ('province_code', 10), ('country_residence', 10)]
    }
    # return feature columns in tensorflow format
    tf_cols = FeatureColumnsBuilder.get_all(train_data, feature_dict)

    # seperate train and valid data
    print('split data to train and valid data')
    X_train, X_valid, y_train, y_valid = train_test_split(train_data[features], train_data[label], test_size=0.2, random_state=0)
    train_dataset = X_train.copy()
    train_dataset[label] = y_train
    valid_dataset  = X_valid.copy()
    valid_dataset[label] = y_valid

    # params for input
    BATCH_SIZE = 300000
    SHUFFLE = True
    dup_cols_map = {
        'bucket': ['age']
    }

    # get input
    train_X, train_y = Model.rename_dup_cols(train_dataset, features, label, dup_cols_map)
    valid_X, valid_y = Model.rename_dup_cols(valid_dataset, features, label, dup_cols_map)
    train_input_fn  = Model.get_input_fn(train_X, train_y, None, BATCH_SIZE, SHUFFLE)
    valid_input_fn = Model.get_input_fn(valid_X, valid_y, 1, None, False)

    # params for model
    N_CLASSES = train_data[label].nunique()
    TOP_K = 7

    # create model
    classifier = tf.estimator.Estimator(
        model_fn = Model.create_model,
        model_dir = 'mymodel1',
        config = tf.estimator.RunConfig(
                save_checkpoints_steps=50,
                save_summary_steps=10
            ),
        params={
            'feature_columns': tf_cols,
            'hidden_units': [100, 50, 25],
            'n_classes': N_CLASSES,
            'k': TOP_K
    })

    # hide logging
    tf.logging.set_verbosity(tf.logging.ERROR)

    # params for training
    EPOCHS = 2
    DISPLAY_STEPS = 1

    # start training
    print('start training ...')
    for n in range(EPOCHS):
        classifier.train(
            input_fn = train_input_fn,
            steps = 10
        )
        results = classifier.evaluate(input_fn = valid_input_fn)
        
        if (n+1) % DISPLAY_STEPS == 0:
            print(n + 1, 'rounds')
            # Display evaluation metrics
            print('Results at epoch', (n + 1) * DISPLAY_STEPS)
            print('-' * 30)
            for key in sorted(results):
                print('%s: %s' % (key, results[key]))
    
    print('execution time: {}'.format(timedelta(seconds=time.time() - start_time)))

if __name__ == '__main__':
    main()
