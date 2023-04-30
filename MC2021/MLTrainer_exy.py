#!/usr/bin/env python3
#
# ML Training code for the ECal data.
# For details see the notebook: MLTests.ipynb
# This code will train for E, x and y simultaneously.
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

__version__ = "1.0.0"

class NumpyArrayEncoder(JSONEncoder):
    """This is a helper class deriving from JSONEncoder to help write np.array objects to disk in JSON format.
    The code came from: https://pynative.com/python-serialize-numpy-ndarray-into-json/"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def main(argv=None):

    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    parser = argparse.ArgumentParser(
        description="""This Python script runs a ML trainer for the ECal data. Version:""" + __version__,
        epilog="""
            For more info, read the script ^_^, or email maurik@physics.unh.edu.""")

    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-t', '--train', type=float, help="Percentage of data to use for training, rest is validation.",
                        default=50.)
    parser.add_argument('-s', '--split', type=int, help="Split the fit batch into N parts so there are more updates.",
                        default=1)
    parser.add_argument('-n', '--numepocs', type=int, help="Number of epocs to optimize over.",
                        default=10)
    parser.add_argument('-f', '--freeze', type=str, help="Batch size for the fit.", default=None)
    parser.add_argument('-r', '--random', type=float, help="Init weights randomly with std, instead of linear fit.",
                        default=0.)
    parser.add_argument('-a', '--activation', type=str, help="Activation function to use", default="elu")
    parser.add_argument('-m', '--model', type=int, help="Model to use. [1-6] (6)", default=6)
    parser.add_argument('--alpha', type=float, help="Regularization parameter for the NN.", default=0.)
    parser.add_argument('--skipval', action="store_true", help="Skip validation for each step.")
    parser.add_argument('--nonorm', action="store_true", help="Do not normalize the data.")
    parser.add_argument('-cp', '--checkpoint', type=int, help="Save model json after N epochs.", default=0)
    parser.add_argument('--cont', action="store_true", help="Continue last run with last saved weights.")
    parser.add_argument('--root', action="store_true",
                        help="As a last step, create a ROOT RDataFrame and store to file.")
    parser.add_argument('--rate', type=float, help="Set the training rate. (1e-5) ", default=1e-5)
    parser.add_argument('--momentum', type=float, help="Set the training momentum. (0.9)", default=0.9)
    parser.add_argument('--file', type=str, help="Data file name to read/write model parameters.",
                        default=None)
    parser.add_argument('input_file', type=str, help="Input files with a Pandas DataFrame in the feather format")
    args = parser.parse_args(argv[1:])

    if args.debug > 0:
        print(f"MLTrainer_exy.py (version {__version__})")
        print(f"Tensorflow version: {tf.__version__}")
        print(f"Keras version: {keras.__version__}")
        print()

    input_file = args.input_file

    if args.debug > 0:
        print("Input file: ", input_file)
        print("Freeze: ", args.freeze)
        print()

    if args.file is None:
        data_file_name = os.path.splitext(os.path.basename(input_file))[0] + "_M" + str(args.model) + ".json"
    else:
        data_file_name = args.file

    data_file_name_root = os.path.splitext(data_file_name)[0]

    # Prepare the data. First load it.
    df = pd.read_feather(input_file)
    # Extend the data with 1/E and 1/sqrt(E), these are from Andrea's fits.
    df['one_over_e'] = 1/df['energy']
    df['one_over_sqrt_e'] = 1/np.sqrt(df['energy'])

    if args.cont:
        if args.debug > 1:
            print("Loading weights from file: ", data_file_name)
        with open(data_file_name, "r") as read_file:
            decodedArray = json.load(read_file)
            weights_store = decodedArray['weights_store']
            weights = []
            for w in weights_store[-1]:
                weights.append(np.array(np.array(w)))
            loss_store = decodedArray['loss_store']
            fit_mse_store = decodedArray['fit_mse_store']
            val_mse_store = decodedArray['val_mse_store']
            if "means" in decodedArray:
                means = np.array(decodedArray['means'])

            else:
                means = df.mean().values
            if "standard_devs" in decodedArray:
                standard_devs = np.array(decodedArray['standard_devs'])
            else:
                standard_devs = df.std().values
    else:
        weights_store = []
        loss_store = [0]
        fit_mse_store = [[0] * 3]
        val_mse_store = [[0] * 3]
        means = df.mean().values
        standard_devs = df.std().values

    if not args.nonorm:
        # Standardize the data
        dfn = (df - means)/standard_devs
    else:
        dfn = df
        means = np.array([0.]*len(df.columns))
        standard_devs = np.array([1.]*len(df.columns))

    np.random.seed(42)
    ran_loc = np.random.permutation(len(dfn))                   # To randomize the entries in the data set.

    # Split the data into what you may know, and the target
    dfc = dfn[["energy", "x", "y", "nhits", "seed_e", "one_over_e", "one_over_sqrt_e"]]
    dfy = dfn[['score_e', 'score_x', 'score_y']]            # You can also train for "true_e"

    # Now split the data into a training and a validation set.
    split_frac = args.train/100.
    split_point = int(len(ran_loc)*split_frac)
    fit_loc = ran_loc[0:split_point]
    if split_frac < 50.:   # No need to validate on more than the training data.
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

    print("Values from last run:")
    print(f"Losses for the last fit           = {loss_store[-1]:12.6f}")
    print(f"Mean square errors for the fit    = {fit_mse_store[-1][0]:12.6f}, {fit_mse_store[-1][1]:12.6f}, {fit_mse_store[-1][2]:12.6f}")
    print(f"Mean square errors for validation = {val_mse_store[-1][0]:12.6f}, {val_mse_store[-1][1]:12.6f}, {val_mse_store[-1][2]:12.6f}")

    activation_function = args.activation
    if activation_function == "leaky_relu":
        # activation_function = tf.nn.leaky_relu(alpha=0.01)
        activation_function = tf.keras.layers.LeakyReLU(alpha=0.01)
        
    # Build the models
    model = None
    if args.alpha > 1e-30:
        reg = tf.keras.regularizers.l2(args.alpha)
    else:
        reg = None

    if args.model == 1:
        model = keras.Sequential([
            layers.Dense(3, activation="linear", input_shape=(7,),
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros())
        ])
    elif args.model == 2:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(20, activation="linear",
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros()),
            layers.Dense(3, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
    elif args.model == 3:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(100, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(100, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(3, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])
    elif args.model == 4:
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(460, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(460, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(3, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=args.random),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])

    elif args.model == 5 or args.model == 6:
        width = 28*3
        model = keras.Sequential([
            keras.Input(shape=(7,)),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),

            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),
            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),

            layers.Dense(width, activation=activation_function,
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg),

            layers.Dense(3, activation="linear",
                         kernel_initializer=tf.keras.initializers.Identity(),
                         bias_initializer=tf.keras.initializers.Zeros(),
                         kernel_regularizer=reg)
        ])

    else:
        print("Model not implemented")
        exit(1)

    # --------------------------- Freezer --------------------------------
    # Parse the freeze argument
    if args.freeze is None or args.freeze.lower() == "none":
        freeze = None
    else:
        freeze = eval(args.freeze)  # Parse the freeze argument as list or number

    if type(freeze) is int:
        freeze = list(range(freeze, len(model.layers)))

    if type(freeze) is list or type(freeze) is tuple:
        if args.debug:
            print("Freezing: ", freeze)
        for i in freeze:
            model.get_layer(index=i).trainable = False

    model.compile(
        optimizer=tf.optimizers.SGD(learning_rate=args.rate, momentum=args.momentum),
        #optimizer=tf.optimizers.Adam(learning_rate=args.rate),
        loss="mse"
        # tf.keras.losses.MeanSquaredError() # alternate: 'mean_absolute_error'='mae', 'mean_squared_error' = 'mse'
    )

    # if we did not load the weights from a file, get a first guess by using a linear fit.
    if len(weights_store) == 0:
        weights_store = [model.get_weights()]
        if args.random < 1e-20:
            print()
            print("Setting the weights to the linear model")
            print()
            linreg = LinearRegression()
            linreg.fit(dfc_fit, dfy_fit)
            lin_coeffs = linreg.coef_
            print(lin_coeffs)
            lin_const = linreg.intercept_
            print(lin_const)

            if args.debug > 1:
                tmp_pred = linreg.predict(dfc_fit)
                print("Initial MSE of linear fit: ", end="")
                for ii in range(3):
                    print(f"{mean_squared_error(dfy_fit.iloc[:, ii], tmp_pred[:, ii]):12.6f}, ", end="")
                print()
            #
            # The input is 7 variables wide, the model uses dense layers that 28*3 wide (7*4*3).
            # Since the model uses "Relu" or "Elu" activation, we need to use one neuron for the positive and
            # one for the negative part of the input values. We double this up to have two complete paths to the
            # final layer. The intermediate layers all start out as identities. In the final layer we then need to
            # use alternate 1 and -1 to re-constitute the split paths of the data.
            #
            weights = model.get_weights()
            for jj in range(len(lin_coeffs)):
                for ii in range(len(lin_coeffs[jj])):
                    diag = ii + jj*4*len(lin_coeffs[jj])
                    weights[0][ii][diag + 0*len(lin_coeffs[jj])] = lin_coeffs[jj][ii]/2.
                    weights[0][ii][diag + 1*len(lin_coeffs[jj])] = -lin_coeffs[jj][ii]/2.
                    weights[0][ii][diag + 2*len(lin_coeffs[jj])] = lin_coeffs[jj][ii]/2.
                    weights[0][ii][diag + 3*len(lin_coeffs[jj])] = -lin_coeffs[jj][ii]/2.
                    #
                    weights[-2][diag + 0*len(lin_coeffs[jj])][jj] = 1.
                    weights[-2][diag + 1*len(lin_coeffs[jj])][jj] = -1.
                    weights[-2][diag + 2*len(lin_coeffs[jj])][jj] = 1.
                    weights[-2][diag + 3*len(lin_coeffs[jj])][jj] = -1.
                # %%
                weights[-1][jj] = lin_const[jj]

            model.set_weights(weights)
        else:
            if args.debug:
                print()
                print(f"Setting the weights to random values with stdev= {args.random}")
                print()
            weights = model.get_weights()
            for ii in range(len(model.layers)):
                if ii not in freeze:
                    # Set initial weights to distributed values.
                    weights[2*ii] = np.random.normal(0., args.random, weights[2*ii].shape)
                    weights[2*ii+1] = np.random.normal(0., args.random, weights[2*ii+1].shape)
                else:
                    # Frozen weights are set to Identity matrix for the even weights (the matrix) and
                    # zero for the odd weights (the bias).
                    for n in range(0, weights[2*ii].shape[0] * weights[2*ii].shape[1],
                                   min(weights[2*ii].shape[0], weights[2*ii].shape[1])):
                        i = int(n / weights[2*ii].shape[1])
                        j = int(n / weights[2*ii].shape[0])
                        # For the last layer, we alternate 1 and -1 to take into account the relu type data split.
                        if (ii < len(model.layers)-1) or int(i/weights[0].shape[0]) % 2 == 0:
                            weights[2 * ii][i][j] = 1.
                        else:
                            weights[2 * ii][i][j] = -1.
                    weights[2*ii+1] = np.zeros(weights[2*ii+1].shape)
                    pass

            model.set_weights(weights)

    else:
        model.set_weights(weights)

    if args.debug:
        print(f"Activation function = {activation_function}  st.dev = {args.random}  rate = {args.rate} "
              f" momentum={args.momentum}\n")
        model.summary()

    if args.debug > 2:
        print("Weights in the model at start:")
        weights = model.get_weights()
        for ii in range(len(model.layers)):
            print(f"Layer {ii:2d}:", weights[ii].shape, weights[ii + 1].shape)
            print(weights[2 * ii])
            print(weights[2 * ii + 1])

    if args.debug > 1:
        print("Checking the weights initalization to linear fit.")
        tmp_pred = model.predict(dfc_fit)
        print("Starting mse of model:", mean_squared_error(dfy_fit.iloc[:, 0], tmp_pred[:, 0]))

    splits = [0]
    for i in range(args.split):
        splits.append(int((i+1)*len(dfc_fit)/args.split))

    for i_epoc in range(args.numepocs):
        for i_split in range(1, len(splits)):
            # Run the model once.
            if args.debug:
                print(f"[{i_epoc:2d}.{i_split:2d}] ")
            if args.debug > 0:
                fit_debug = 1
            else:
                fit_debug = 0
            history = model.fit(dfc_fit.iloc[splits[i_split-1]:splits[i_split]],
                                dfy_fit.iloc[splits[i_split-1]:splits[i_split]],  verbose=fit_debug, epochs=1)

            loss_store.append(history.history['loss'][-1])
            if not args.skipval:
                print(f"Losses for the last fit           = {loss_store[-1]:12.6f}")
                Ypred_fit = model.predict(dfc_fit.iloc[splits[i_split-1]:splits[i_split]])
                mse_fit = []
                for ii in range(len(dfy_fit.iloc[0])):
                    mse_fit.append(mean_squared_error(Ypred_fit[:, ii],
                                                      dfy_fit.iloc[splits[i_split-1]:splits[i_split], ii]))
                fit_mse_store.append(mse_fit)
                print(
                    f"Mean square errors for the fit    = {mse_fit[0]:12.6f}, {mse_fit[1]:12.6f}, {mse_fit[2]:12.6f}")

                Ypred_val = model.predict(dfc_val.iloc[splits[i_split-1]:splits[i_split]])
                mse_val = []
                for ii in range(len(dfy_val.iloc[0])):
                    mse_val.append(mean_squared_error(Ypred_val[:, ii],
                                                      dfy_val.iloc[splits[i_split-1]:splits[i_split], ii]))
                val_mse_store.append(mse_val)
                print(
                    f"Mean square errors for validation = {mse_val[0]:12.6f}, {mse_val[1]:12.6f}, {mse_val[2]:12.6f}")

            else:
                fit_mse_store.append([0]*len(dfy_fit.iloc[0]))
                val_mse_store.append([0]*len(dfy_val.iloc[0]))

        if args.checkpoint > 0 and (i_epoc+1) % args.checkpoint == 0:
            weights_store.append(model.get_weights())
            if args.debug:
                print("Storing checkpoint.")
            outData = {"loss_store": loss_store, "fit_mse_store": fit_mse_store, "val_mse_store": val_mse_store,
                       "weights_store": weights_store}
            with open(data_file_name, "w") as f:
                json.dump(outData, f, cls=NumpyArrayEncoder)

    print("Computing non-batched loss:")
    print("Final values:")
    print(f"Loss                             = {loss_store[-1]:12.6g}")
    Ypred_fit = model.predict(dfc_fit, verbose=args.debug)
    mse = []
    for ii in range(len(dfy_fit.columns)):
        mse.append(mean_squared_error(Ypred_fit[:, ii], dfy_fit.iloc[:, ii]))
    fit_mse_store[-1] = mse
    print(f"Mean square error for the fit    = {mse[0]:12.6f}, {mse[1]:12.6f}, {mse[2]:12.6f}")
    Ypred_val = model.predict(dfc_val, verbose=args.debug)
    mse = []
    for ii in range(len(dfy_val.columns)):
        mse.append(mean_squared_error(Ypred_val[:, ii], dfy_val.iloc[:, ii]))
    val_mse_store[-1] = mse

    print(f"Mean square error for validation = {mse[0]:12.6f}, {mse[1]:12.6f}, {mse[2]:12.6f}")

    weights_store.append(model.get_weights())
    print("Storing final weights.")
    outData = {"loss_store": loss_store, "fit_mse_store": fit_mse_store, "val_mse_store": val_mse_store,
               "weights_store": weights_store, "means": np.array(means), "standard_devs": np.array(standard_devs)}

    with open(data_file_name, "w") as write_file:
        json.dump(outData, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized info into "+data_file_name)

    if args.root:
        print("Storing the prediction output in ROOT files:"
              + data_file_name_root + "_fit.root and " + data_file_name_root + "_val.root")
        if len(dfc_fit) + len(dfc_val) < len(df) - 3:   # Allow for some rounding error.
            print("WARNING: Not all data is in fit + validation sets. Run with -t 50 to get a 50/50 split and all data.")

        write_keys = ['true_e', 'score_e', 'score_x', 'score_y', 'energy', 'energy_cor', 'x', 'y', 'x_cor', 'y_cor',
                      'nhits', 'seed_e', 'one_over_e', 'one_over_sqrt_e']

        if not args.nonorm:
            # We need to unwind the normalization.
            # In the original dataframe df, from which the mean and std come, the order was:
            # "energy", "energy_cor", "x", "y", ...
            pred_means = means[[11, 12, 13]]
            pred_std = standard_devs[[11, 12, 13]]
            Ypred_val = Ypred_val * pred_std + pred_means
            Ypred_fit = Ypred_fit * pred_std + pred_means

        tmp_data = {key: df.iloc[val_loc][key].values for key in write_keys}
        tmp_data['energy_NN'] = Ypred_val[:, 0].copy()
        tmp_data['x_NN'] = Ypred_val[:, 1].copy()
        tmp_data['y_NN'] = Ypred_val[:, 2].copy()
        import ROOT as R
        rdf_NN = R.RDF.FromNumpy(tmp_data)
        rdf_NN.Snapshot("EcalTraining", data_file_name_root+"_val.root")

        if args.debug > 2:
            # prt = rdf_NN.Display(['score_e', 'energy', 'energy_NN', 'x_NN', 'y_NN'], 20)
            # prt.Print()
            for ii in range(20):
                print(f"{tmp_data['score_e'][ii]:12.6f} {tmp_data['energy'][ii]:12.6f} {Ypred_val[ii,0]:12.6f}"
                      f" {Ypred_val[ii, 1]:12.6f} {Ypred_val[ii, 2]:12.6f} ")

        tmp_data = {key: df.iloc[fit_loc][key].values for key in write_keys}
        tmp_data['energy_NN'] = Ypred_fit[:, 0].copy()
        tmp_data['x_NN'] = Ypred_fit[:, 1].copy()
        tmp_data['y_NN'] = Ypred_fit[:, 2].copy()
        rdf_NN = R.RDF.FromNumpy(tmp_data)
        rdf_NN.Snapshot("EcalTraining", data_file_name_root+"_fit.root")


if __name__ == "__main__":
    sys.exit(main())
