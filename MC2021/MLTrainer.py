#!/usr/bin/env python3
#
# ML Training code for the ECal data.
# For details see the notebook: MLTests.ipynb
#
import sys
import os
import argparse
import time
import numpy as np
import json
from json import JSONEncoder
import feather
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# import tensorflow as tf
from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class NumpyArrayEncoder(JSONEncoder):
    """This is a helper class deriving from JSONEncoder to help write np.array objects to disk in JSON format.
    The code came from: https://pynative.com/python-serialize-numpy-ndarray-into-json/"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def main(argv=None):
    """Main code is here."""
    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    parser = argparse.ArgumentParser(
        description="""This Python script runs a ML trainer for the ECal data.""",
        epilog="""
            For more info, read the script ^_^, or email maurik@physics.unh.edu.""")

    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-t', '--train', type=float, help="Percentage of data to use for training, rest is validation.",
                        default=50.)
    parser.add_argument('-s', '--split', type=int, help="Split the fit batch into N parts so there are more updates.",
                        default=1)
    parser.add_argument('-n', '--numepocs', type=int, help="Number of epocs to optimize over.",
                        default=10)
    parser.add_argument('-m', '--model', type=int, help="Model to use: 1=Linear, 2=NN2 3=DeepNN, 4=Deep Wide", default=3)
    parser.add_argument('-a', '--alpha', type=float, help="Regularization parameter for the NN.", default=1e-30)
    parser.add_argument('--skipval', action="store_true", help="Skip validation for each step.")
    parser.add_argument('-cp', '--checkpoint', type=int, help="Save model json after N epochs.", default=0)
    parser.add_argument('--cont', action="store_true", help="Continue last run with last saved weights.")
    parser.add_argument('--root', action="store_true",
                        help="As a last step, create a ROOT RDataFrame and store to file.")
    parser.add_argument('--rate', type=float, help="Set the training rate. ", default=0.00001)
    parser.add_argument('--file', type=str, help="Data file name to read/write model parameters.",
                        default=None)
    parser.add_argument('input_file', type=str, help="Input files with a Pandas DataFrame in the feather format")

    args = parser.parse_args(argv[1:])

    print(f"Tensorflow version: {tf.__version__}")

    input_file = args.input_file
    print("Input file: ", input_file)

    if args.file is None:
        data_file_name = os.path.splitext(os.path.basename(input_file))[0] + "_M" + str(args.model) + ".json"
    else:
        data_file_name = args.file

    data_file_name_root = os.path.splitext(data_file_name)[0]

    # Prepare the data. First load it.
    df = pd.read_feather(input_file)
    # We now extend the data. The true_e is highly truncated, which we can repeat here,
    # but it is needed only for visulalization.
    # df['true_e'] = df['true_e'].round(1)                       # Rounded version to one decimal.
    df['one_over_e'] = 1/df['energy']
    df['one_over_sqrt_e'] = 1/np.sqrt(df['energy'])
    ran_loc = np.random.permutation(len(df))                   # To randomize the entries in the data set.

    # Split the data into what you may know, and the target
    dfc = df[["energy", "x", "y", "nhits", "seed_e", "one_over_e", "one_over_sqrt_e"]].copy()
    dfy = df[['score_e']].copy()   # You can also train for "true_e"

    # Now split the data into a training and a validation set.
    split_frac = args.train/100.
    split_point = int(len(ran_loc)*split_frac)
    fit_loc = ran_loc[0:split_point]
    if split_frac < 25.:   # No need to validate on more than the training data.
        val_loc = ran_loc[split_point:2*split_point]
    else:
        val_loc = ran_loc[split_point:]
    dfc_fit = dfc.iloc[fit_loc]
    dfy_fit = dfy.iloc[fit_loc]
    dfc_val = dfc.iloc[val_loc]
    dfy_val = dfy.iloc[val_loc]

    #
    # Here we start the actual ML part.
    # This one for Keras - Tensorflow
    #
    if args.cont:
        with open(data_file_name, "r") as read_file:
            decodedArray = json.load(read_file)
            weights_store = decodedArray['weights_store']
            weights = []
            for w in weights_store[-1]:
                weights.append(np.array(np.array(w)))
            loss_store = decodedArray['loss_store']
            fit_mse_store = decodedArray['fit_mse_store']
            val_mse_store = decodedArray['val_mse_store']
        print("Starting values:")
        print(f"Mean square error for the fit    = {fit_mse_store[-1]}")
        print(f"Mean square error for validation = {val_mse_store[-1]}")
        print(f"Loss function                    = {loss_store[-1]}")
    else:
        weights_store = []
        loss_store = [0]
        fit_mse_store = [0]
        val_mse_store = [0]
        print("Starting without restoring weights.")

    # Build the models
    model = None
    if args.alpha > 1e-30:
        reg = tf.keras.regularizers.l2(args.alpha)
    else:
        reg = None

    if args.model == 1:
        model = keras.Sequential([
            layers.Dense(units=1, activation="linear", input_shape=(7,),
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros())
        ])
    elif args.model == 2:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(20, activation="linear",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros()),
            layers.Dense(1, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
    elif args.model == 3:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(100, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(100, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(1, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
    elif args.model == 4:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(460, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(1, activation="elu",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])

    elif args.model == 5 or args.model == 6:
        if args.model == 5:
            initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=0.1)
        else:
            initializer = tf.keras.initializers.Identity()
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),

            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(1, activation="linear",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
    elif args.model == 7:

        if args.model == 7:
            initializer = "he_normal"
        else:
            initializer = tf.keras.initializers.Identity()
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01),
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(28, activation="elu",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dense(1, activation="linear",
                         kernel_initializer=initializer,
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
# keras.layers.BatchNormalization()
    else:
        print("Model not implemented")
        exit(1)

    model.compile(
        #optimizer=tf.optimizers.SGD(learning_rate=args.rate),
        optimizer=tf.optimizers.Adam(learning_rate=args.rate),
        loss="mse"
        # tf.keras.losses.MeanSquaredError() # alternate: 'mean_absolute_error'='mae', 'mean_squared_error' = 'mse'
    )

    # Now run the iterations
    # TODO: Set this up smarter by using the history feature and having validation done with a callback.
    N_epocs = args.numepocs
    if not args.cont:
        # weights_store = [model.get_weights()]
        # TODO: Make this possible for any of the models.
        if args.model == 6:
            print()
            print("Setting the weights to the linear model")
            print()
            linreg = LinearRegression()
            linreg.fit(dfc_fit, dfy_fit)
            lin_coeffs = linreg.coef_[0]
            lin_const = linreg.intercept_[0]
            weights = model.get_weights()
            for ii in range(len(lin_coeffs)):
                weights[0][ii][ii] = lin_coeffs[ii]/2.
                weights[0][ii][ii + len(lin_coeffs)] = -lin_coeffs[ii]/2.
                weights[0][ii][ii + 2*len(lin_coeffs)] = lin_coeffs[ii]/2.
                weights[0][ii][ii + 3*len(lin_coeffs)] = -lin_coeffs[ii]/2.
                weights[-2][ii] = 1.
                weights[-2][ii + len(lin_coeffs)] = -1.
                weights[-2][ii + 2*len(lin_coeffs)] = 1.
                weights[-2][ii + 3*len(lin_coeffs)] = -1.
            # %%
            weights[-1][0] = lin_const

            for ii in range(len(weights)):
                # Add some randomness to the initial weights
                weights[ii] = weights[ii] + np.random.normal(0, 1e-6, weights[ii].shape)

            model.set_weights(weights)
    else:
        model.set_weights(weights)

    if args.debug:
        model.summary()

    splits = [0]
    for i in range(args.split):
        splits.append(int((i+1)*len(dfc_fit)/args.split))

    for i_epoc in range(N_epocs):
        for i_split in range(1, len(splits)):
            # Run the model once.
            if args.debug:
                print(f"[{i_epoc:2d}.{i_split:2d}] ")
            history = model.fit(dfc_fit.iloc[splits[i_split-1]:splits[i_split]],
                                dfy_fit.iloc[splits[i_split-1]:splits[i_split]],  verbose=args.debug, epochs=1)

            loss_store.append(history.history['loss'][-1])
            # Ypred_fit = model.predict(dfc_fit.iloc[splits[i_split-1]:splits[i_split]])
            # fit_mse_store.append(mean_squared_error(Ypred_fit, dfy_fit.iloc[splits[i_split-1]:splits[i_split]]))
            fit_mse_store.append(history.history['loss'][-1])
            if not args.skipval:
                Ypred_val = model.predict(dfc_val.iloc[splits[i_split-1]:splits[i_split]])
                val_mse_store.append(mean_squared_error(Ypred_val, dfy_val.iloc[splits[i_split-1]:splits[i_split]]))
            else:
                val_mse_store.append(0)

        if args.checkpoint > 0 and (i_epoc+1) % args.checkpoint == 0:
            if args.debug:
                print("Storing checkpoint.")
            outData = {"loss_store": loss_store, "fit_mse_store": fit_mse_store, "val_mse_store": val_mse_store,
                       "weights_store": weights_store}
            with open(data_file_name, "w") as f:
                json.dump(outData, f, cls=NumpyArrayEncoder)

    print("Computing non-batched loss:")
    Ypred_val = model.predict(dfc_val, verbose=args.debug)
    Ypred_fit = model.predict(dfc_fit, verbose=args.debug)
    val_mse_store[-1] = mean_squared_error(Ypred_val, dfy_val)
    fit_mse_store[-1] = mean_squared_error(Ypred_fit, dfy_fit)
    print("Final values:")
    print(f"Loss                             = {loss_store[-1]}")
    print(f"Mean square error for the fit    = {fit_mse_store[-1]}")
    print(f"Mean square error for validation = {val_mse_store[-1]}")

    outData = {"loss_store": loss_store, "fit_mse_store": fit_mse_store, "val_mse_store": val_mse_store,
               "weights_store": weights_store}

    with open(data_file_name, "w") as write_file:
        json.dump(outData, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized info into "+data_file_name)

    if args.root:
        print("Storing the prediction output in ROOT files.")
        if len(dfc_fit) + len(dfc_val) < len(df) - 3:   # Allow for some rounding error.
            print("WARNING: Not all data is in fit + validation sets. Run with -t 50 to get a 50/50 split and all data.")
        import ROOT as R
        write_keys = ['true_e', 'score_e', 'energy', 'energy_cor',
                      'nhits', 'seed_e', 'one_over_e', 'one_over_sqrt_e']
        tmp_data = {key: df.iloc[val_loc][key].values for key in write_keys}
        tmp_data['energy_NN'] = Ypred_val[:, 0]
        rdf_NN = R.RDF.FromNumpy(tmp_data)
        rdf_NN.Snapshot("EcalTraining", data_file_name_root+"_val.root")
        tmp_data = {key: df.iloc[fit_loc][key].values for key in write_keys}
        tmp_data['energy_NN'] = Ypred_fit[:, 0]
        rdf_NN = R.RDF.FromNumpy(tmp_data)
        rdf_NN.Snapshot("EcalTraining", data_file_name_root+"_fit.root")


if __name__ == "__main__":
    sys.exit(main())
