#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import numpy as np
import pandas as pd
import sys
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

"""
Implementation of LIMES and baseline methods for Class-Prior Shift in Continuous Learning.
"""

# class that allows to incorporate bias correction term in classical Linear layer,
# if bias correction term is not needed, can be simply set to zero
class Linear(tensorflow.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        self.use_bias = use_bias
        super(Linear, self).__init__()
        w_init = tensorflow.random_normal_initializer()
        self.w = tensorflow.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tensorflow.zeros_initializer()
        self.b = tensorflow.Variable(initial_value=b_init(shape=(units, ),
                                                          dtype="float32"),
                                     trainable=use_bias)

        self.c = tensorflow.Variable(
            tensorflow.zeros_initializer()(shape=(units, ), dtype="float32"),
            trainable=False,
        )

    def call(self, inputs):  
        return tensorflow.matmul(inputs, self.w) + self.b + self.c
        

def create_model(source, n_classes=250):
    if source == "tweet_location":
        n_dim = 1024
    else:
        n_dim = 512

    x = Input(shape=(n_dim, ))
    z = Linear(n_classes, n_dim)(x)
    model = Model(inputs=x, outputs=z)
    return model

def create_many_models(source, n_classes=250, num_models=24):
    if source == "tweet_location":
        n_dim = 1024
    else:
        n_dim = 512
    
    all_models = []
    for i in range(num_models):
        x = Input(shape=(n_dim, ))
        z = Linear(n_classes, n_dim)(x)
        all_models.append( Model(inputs=x, outputs=z) )
    return all_models

def distance(p1, p2):
    return np.sum(np.abs(p1 - p2))

def search_similar(W):
    dist = [distance(W[-1], w_i) for w_i in W[:-1]]
    pk_index = np.argmin(dist)
    return pk_index

def evaluation(t, df_results):
    subset_per_hour = df_results[df_results.hour == t]
    acc_point_in_time = np.mean(
        subset_per_hour.true_label == subset_per_hour.pred)
    return acc_point_in_time


def update_weight_inplace(W, hour, y_true):
    counts = np.bincount(y_true, minlength=W.shape[-1])
    W[hour] += counts
    return


def load_file(filename, source):
    data = np.load(filename)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    if source == "tweet":
        X = X[:, :512]
    elif source == "location":
        X = X[:, -512:]
    elif source == "tweet_location":
        X = X[:, :]
    return X, y


def train(args):
    n_classes = args.nclasses
    batch_size = args.batchsize
    
    if args.multi:
        all_models = create_many_models(args.source, n_classes, args.nmodels)
        for model in all_models:
            model.summary(print_fn=lambda x: print(x,file=sys.stderr))
    else:
        model = create_model(args.source, n_classes)
        model.summary(print_fn=lambda x: print(x,file=sys.stderr))

    # Instantiate an optimizer.
    optimizer = tensorflow.keras.optimizers.Adam()
    # Instantiate a loss function.
    loss_fn = tensorflow.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)


    file_path = args.path + args.dates

    filenames_train = glob.glob(
        file_path + "/train/embeddings_train_2020_0[4567]_??_??-sub{}.npy".format(args.seed%10))
    filenames_train = sorted(filenames_train,
                             reverse=args.rev) 
    filenames_val = [f.replace("train", "val") for f in filenames_train]
    # ensure perfect match between train and val filename
    n_files = len(filenames_train)

    indexes = []
    acc_in_time = []

    W = np.ones((0, n_classes))
    
    
    for counter in range(n_files):
        
        # first construct or pick model for current hour
        if args.multi:
            hours_per_model = 24//args.nmodels
            model_id = (counter//hours_per_model)%args.nmodels
            model = all_models[model_id]        # current model based on hour

     
        if args.mode == "limes" and counter > 1:
            pk_index = search_similar(W)
            p_t = W[pk_index + 1, :]
            indexes.append(pk_index)
        else:  
            p_t = np.ones(n_classes) / n_classes

        c = np.log(p_t) - np.log(1 / n_classes)
        
        cur_W, cur_b, _ = model.layers[-1].get_weights()
        model.layers[-1].set_weights([cur_W, cur_b, c])


        # evaluate model on the hour that the model was not previously trained
        if counter >= 0:   #could be >0 but for consistency across all periods, we use >=0
            X_val, y_val = load_file(filenames_val[counter], args.source)
            val_logits = model(X_val, training=False).numpy()

            y_pred_val = np.argmax(val_logits, axis=1)
            df_results = pd.DataFrame({
                "hour": [counter] * len(y_val),
                "true_label": y_val,
                "pred": y_pred_val
            })

            acc_in_time.append(evaluation(counter, df_results))
            print(counter, acc_in_time[-1])

           
        @tensorflow.function  # to speed up training
        def train_step(x, y, model, loss_fn):
            with tensorflow.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = loss_fn(y, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss_value

        # train in batches on data of the hour that was the evaluation performed on
        W = np.append(W, np.ones((1, n_classes)), axis=0)

        if args.mode == "reset": # start with fresh model every time
            del model
            model = create_model(args.source, args.nclasses)

        # proper training
        X, y = load_file(filenames_train[counter], args.source)
        for pos in range(0, len(y), batch_size):
            x_batch_train = X[pos:pos + batch_size]
            y_batch_train = y[pos:pos + batch_size]
            update_weight_inplace(W, -1, y_batch_train)  # update information about the number of samples from different classes in data just used for training
            loss_value = train_step(x_batch_train, y_batch_train, model,
                                    loss_fn)

        W[-1] /= np.sum(W[-1])  # normalize counts



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        '-s',
                        type=int,
                        default=0,
                        help="Random seed")
    parser.add_argument('--dates',
                        '-d',
                        type=str,
                        default='0_5',
                        help="data subset",
                        choices=['0_5', '10_15'])
    parser.add_argument('--mode',
                        '-m',
                        type=str,
                        default='zero',
                        help="adaptation mode",
                        choices=['zero', 'limes', 'multi', 'reset'])
    parser.add_argument('--path',
                        '-p',
                        type=str,
                        help="directory to dataset that is later concatenated with dates argument")
                        
    parser.add_argument('--source',
                        '-S',
                        type=str,
                        default='tweet',
                        help="feature set",
                        choices=['tweet', 'location', 'tweet_location'])
    parser.add_argument('--nclasses',
                        '-n',
                        type=int,
                        default=250,
                        help="number of classes")
    parser.add_argument('--batchsize',
                        '-b',
                        type=int,
                        default=100,
                        help="batchsize")
    parser.add_argument('--nmodels',
                        type=int,
                        default=24,
                        help="batchsize")
    parser.add_argument('--rev',
                        '-r',
                        action='store_true',
                        help="Invert data order (default: false)")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    if args.mode == "multi":
        args.multi = True
        args.mode = "zero"
    else:
        args.multi = False
    
    if 24%args.nmodels != 0:
        print(f"Warning: 24 is not a multiple of number of models {args.nmodels}", file=sys.stderr)
    
    np.random.seed(args.seed)
    tensorflow.random.set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
