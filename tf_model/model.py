from __future__ import print_function
import tensorflow as tf

class Model(object):

    @staticmethod
    def train_input_fn(features, labels, batch_size):
        """An input function for training"""
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.shuffle(1000).batch(batch_size).repeat()
        return dataset.make_one_shot_iterator().get_next()

    @staticmethod
    def eval_input_fn(features, labels, batch_size):
        """An input function for evaluation"""
        if labels is None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        return dataset.batch(batch_size)

    @staticmethod
    def rename_dup_cols(df, features, label, dup_cols_map):
        '''Deal with duplicate columns from same base
            dup_cols_map = {
                'bucket': ['age']
            }
        '''
        X = {col: df[col].values for col in features}
        for col_type in dup_cols_map:
            for col in dup_cols_map[col_type]:
                X['{}_{}'.format(col, col_type)] = df[col].values
        y = df[label].values
        return X, y

    @staticmethod
    def create_model(features, labels, mode, params):
        # create network
        net = tf.feature_column.input_layer(features, params['feature_columns'])
        for units in params['hidden_units']:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            
        # last layer (prediction)
        logits = tf.layers.dense(net, params['n_classes'], activation=None)
        
        # compute prediction
        if params['k'] == 1:
            predicted_classes = tf.argmax(logits, 1)
        else:
            predicted_classes = logits       #tf.nn.top_k(logits, k=params['k'])[1]  # get top k class
        if mode == tf.estimator.ModeKeys.PREDICT:
            if params['k'] == 1:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }
            else:
                predictions = {
                    'class_ids': predicted_classes,
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits,
                }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
        # compute loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        
        # compute evaluation metrics
        if params['k'] == 1:
            accuracy = tf.metrics.accuracy(labels=labels,
                                            predictions=predicted_classes,
                                            name='acc_op')
        else:
            accuracy = tf.metrics.average_precision_at_k(labels=labels,
                                                        predictions=predicted_classes,
                                                        k=params['k'],
                                                        name='acc_op')
        
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
        
        # create training op
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)