from __future__ import print_function
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.relativedelta

from column_map import column_map
from pipeline_preprocess import MyPipeline

def mulit_prod_cols2set(train_data, target_columns):
    '''Integrate all products to one column (set), need around 10 mins to finish
        np_idx_prod ------------------------------
        |   index   | prod1 | prod2 | prod3 | ... |
        |   12345   |   1   |   0   |   1   | ... |
        ------------------------------------------
        idx_prod_df --------------------
        |   index   | prod_list         |
        |   12345   | [prod1, prod3]    |
        --------------------------------
    '''
    def map_target(v):
        map_list = set([c[0] for c in zip(target_columns, v[1:]) if c[1] == 1])
        return map_list

    # pandas -> numpy
    train_np = train_data.reset_index()[['index'] + target_columns].values
    # multiple columns to one column, need to use set to avoid numpy shape problem
    np_prod = np.apply_along_axis(map_target, 1, train_np).reshape(-1 , 1)
    # append index back to numpy matrix
    np_idx_prod = np.append(train_np[:, 0].reshape(-1, 1), np_prod, axis=1)
    # numpy -> pandas
    idx_prod_df = pd.DataFrame(np_idx_prod, columns=['index', 'prod_list'])
    # turn prod_list from set to list type
    idx_prod_df['prod_list'] = idx_prod_df['prod_list'].apply(list)
    return idx_prod_df.set_index('index')

def get_prev_prod_set(df):
    '''Add one month in partition col, rename prod_list to prev_prod_list,
       around 10 mins to finish
        prev month ----------------------------------------------
        |   customer_id   |   partition   |  prev_prod_list      |
        |   12345         |   2016-3-28   |  [prod1, prod2, ...] |
        ---------------------------------------------------------
    '''
    def add_1m(partition):
        date_obj = datetime.strptime(partition, '%Y-%m-%d')
        date_prev = date_obj + dateutil.relativedelta.relativedelta(months=1)
        return date_prev.strftime('%Y-%m-%d')

    prev_month = df[['customer_id', 'partition', 'prod_list']]
    prev_month['partition'] = prev_month.copy()['partition'].apply(add_1m)
    prev_month.rename(columns={'prod_list':'prev_prod_list'}, inplace=True)
    return prev_month.set_index(['customer_id','partition'])

def prod_new(prod_list, prev_prod_list):
    '''
        Compare prod_list from last month to find new products cusomter bought this month
    '''
    if isinstance(prod_list, float) and not isinstance(prev_prod_list, float):
        return np.nan
    elif isinstance(prev_prod_list, float) and not isinstance(prod_list, float):
        return prod_list
    elif not isinstance(prev_prod_list, float) and not isinstance(prod_list, float):
        diff = list(set(prod_list) - set(prev_prod_list))
        return diff if len(diff) > 0 else np.nan 

def generate_data(train, test):
    # rename columns
    train.rename(columns=column_map, inplace=True)
    test.rename(columns=column_map, inplace=True)

    # get feature and target columns
    feature_columns = [col for col in train.columns if not col.startswith('prod')]
    target_columns  = [col for col in train.columns if col.startswith('prod')]
    print('{} features'.format(len(feature_columns)))
    print('{} kinds of target to recommend'.format(len(target_columns)))

    # handling train data
    # integrate multiple columns to one list and join back to train data
    print('integrate multiple cols to one col')
    idx_prod_df = mulit_prod_cols2set(train, target_columns)
    train_prod = train.reset_index().join(idx_prod_df, 'index')

    # get prod_list from last month for each customer_id and join back to train data
    print('get prev_prod_list')
    prev_month = get_prev_prod_set(train_prod)
    non_target_cols = [c for c in train_prod.columns if c not in target_columns]
    train = train_prod[non_target_cols].join(prev_month, ['customer_id', 'partition'])

    # compare and find new products each customer bought this month
    train['new_product'] = train.apply(
            lambda row: prod_new(row['prod_list'], row['prev_prod_list']), axis = 1)

    # handling test data
    print('process test data')
    prev_month_test = prev_month[prev_month.index.get_level_values('partition') == '2016-06-28']
    test = test.join(prev_month_test, ['customer_id', 'partition'])

    # merget train and test data
    print('merge train and test data')
    test['prod_list'] = np.nan
    test['new_product'] = np.nan
    train = train[train['new_product'].notnull()]
    data = pd.concat([train, test], sort=False)

    # flapmap, unstack() need around 20 mins to finish
    print('data flatmap')
    data_flat_np = data['new_product'].apply(pd.Series) \
                                        .unstack() \
                                        .dropna()
    data_flat_df = data_flat_np.to_frame() \
                                .reset_index() \
                                .drop('level_0', axis=1) \
                                .rename(columns={'level_1':'index', 0:'product'}) \
                                .set_index('index')
    data = data.reset_index() \
                .join(data_flat_df, 'index') \
                .drop('index', axis = 1)

    print('data preprocessing')
    preprocess_data = MyPipeline().get_pipeline().fit_transform(data)
    preprocess_df = pd.DataFrame(preprocess_data, 
                    columns=['employee_idx', 'country_residence', 'sex',  'relation_type', 'residence_idx', 
                            'foreigner_idx', 'spouse_idx', 'channel', 'deceased_idx', 'province_name', 
                            'segmentation', 'age', 'gross_income_household', 'seniority', 'type', 
                            'month', 'partition', 'customer_id', 'new_customer','primary', 
                            'address_type', 'province_code','activity_idx', 'prod_list', 'prev_prod_list',
                            'new_product', 'product'])
    return preprocess_df

if __name__ == '__main__':
    data_dir = '../dataset/'
    train = pd.read_csv(data_dir + 'train_ver2.csv', low_memory=False)
    test = pd.read_csv(data_dir + 'test_ver2.csv', low_memory=False)
    data = generate_data(train, test)
    data.to_csv(data_dir + 'rec_data.csv', index=False)