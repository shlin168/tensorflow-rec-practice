import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from pipeline_transformer import MyTranformer, MultiColumnLabelEncoder


class MyPipeline(object):
    __feature_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('label_encoder', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_col, validate=False, 
                    kw_args=dict(cols=['employee_idx', 'country_residence', 'sex', 
                                        'relation_type', 'residence_idx', 'foreigner_idx', 
                                        'spouse_idx', 'channel', 'deceased_idx', 'province_name']))),
                ('encoding', MultiColumnLabelEncoder())
            ])),
            ('segmentation', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='segmentation'))),
                ('split_first', FunctionTransformer(MyTranformer.split_n_get_first_item, validate=False)),
                ('str2float', FunctionTransformer(MyTranformer.str2float, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False))
            ])),
            ('age', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='age'))),
                ('trim_if_string', FunctionTransformer(MyTranformer.trim_if_string, validate=False)),
                ('NA2na', FunctionTransformer(MyTranformer.NA2na, validate=False, kw_args=dict(val='NA'))),
                ('str2float', FunctionTransformer(MyTranformer.str2float, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False))
            ])),
            ('gross_income_household', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='gross_income_household'))),
                ('trim_if_string', FunctionTransformer(MyTranformer.trim_if_string, validate=False)),
                ('NA2na', FunctionTransformer(MyTranformer.NA2na, validate=False, kw_args=dict(val='NA'))),
                ('str2float', FunctionTransformer(MyTranformer.str2float, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False))
            ])),
            ('seniority', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='seniority'))),
                ('trim_if_string', FunctionTransformer(MyTranformer.trim_if_string, validate=False)),
                ('NA2na', FunctionTransformer(MyTranformer.NA2na, validate=False, kw_args=dict(val='NA'))),
                ('str2float', FunctionTransformer(MyTranformer.str2float, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False))
            ])),
            ('type', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='type'))),
                ('dealspecific', FunctionTransformer(MyTranformer.deal_specific_val, validate=False, 
                                                    kw_args=dict(input_val='P', output_val=5.0))),
                ('str2float', FunctionTransformer(MyTranformer.str2float, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False))
            ])),
            ('month', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_single_col, validate=False, kw_args=dict(col='partition'))),
                ('getmonth',  FunctionTransformer(MyTranformer.get_month, validate=False)),
                ('toframe',  FunctionTransformer(MyTranformer.to_frame, validate=False)),
                ('rename', FunctionTransformer(MyTranformer.rename_col, validate=False, kw_args=dict(col_map={'partition':'month'}))),
            ])),
            ('select', Pipeline([
                ('selector', FunctionTransformer(MyTranformer.select_col, validate=False, 
                                kw_args=dict(cols=['partition', 'customer_id', 'new_customer','primary',
                                                    'address_type','province_code','activity_idx', 'prod_list',
                                                    'prev_prod_list', 'new_product', 'product'])))
            ]))
        ]))
    ])

    def get_pipeline(self):
        return self.__feature_pipeline