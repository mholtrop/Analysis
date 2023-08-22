#!/usr/bin/env python3
#
# ML Training code for the ECal data.
# For details see the notebook: MLTests.ipynb
# This code will train for E, x and y simultaneously.
#
###
# Sparsify the input data from the full ROOT files can be done with C++ ROOT:
# .L ../lib/libEcal_Analysis.dylib
# ROOT::EnableImplicitMT();
# using namespace ROOT;
# auto ch = new TChain("MiniDST"); ch->Add("/data/HPS/data/MC/ele_2019/ele_*.root")
# auto rdf = RDataFrame(*ch)
# rdf.Snapshot("MiniDST","electrons_sparse.root", {"ecal_cluster_energy", "ecal_cluster_mc_id",
#         "ecal_cluster_seed_energy", "ecal_cluster_seed_ix", "ecal_cluster_seed_iy", "ecal_cluster_uncor_energy",
#         "ecal_cluster_uncor_hits", "ecal_cluster_x", "ecal_cluster_y", "ecal_hit_energy", "ecal_hit_index_x",
#         "ecal_hit_index_y", "event_number", "mc_part_energy", "mc_score_px", "mc_score_py", "mc_score_pz",
#         "mc_score_x", "mc_score_y", "mc_score_z"})
#
# This reduces the total size from 20GB to 5GB (~2M events) and the load time for 200000 uncut clusters from
# ~43s to ~21s.
#
import sys
import os
import argparse
import time
import numpy as np
import json
from json import JSONEncoder

try:
    import tensorrt
except ImportError:
    pass
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.regularizers import L1

from matplotlib import pyplot as plt

class NumpyArrayEncoder(JSONEncoder):
    """This is a helper class deriving from JSONEncoder to help write np.array objects to disk in JSON format.
    The code came from: https://pynative.com/python-serialize-numpy-ndarray-into-json/"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def main(argv=None):

    import ROOT as R
    print(f"ROOT version is {R.gROOT.GetVersion()}")
    __version__ = "1.0.1"

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
    parser.add_argument('-N', '--numevents', type=int, help="Number of events to use.", default=100000)
    parser.add_argument('-t', '--train', type=float, help="Percentage of data to use for training, rest is validation.",
                        default=50.)
    parser.add_argument('-b', '--minibatch', type=int, help="Mini batch size.", default=32)
    parser.add_argument('-s', '--split', type=int, help="Split the fit batch into N parts so there are more updates.",
                        default=1)
    parser.add_argument('-n', '--numepocs', type=int, help="Number of epocs to optimize over.",
                        default=10)
    parser.add_argument('-f', '--freeze', type=str, help="Freeze layers. ", default=None)
    parser.add_argument('-c', '--cuts', type=int, help="Cuts to us. 0=None, 1=Fiducial, 2=Anti-Fiducial", default=0)
    # parser.add_argument('-r', '--random', type=float, help="Init weights randomly with std, instead of linear fit.",
    #                    default=0.)
    parser.add_argument('-a', '--activation', type=str, help="Activation function to use", default="leaky_relu")
    parser.add_argument('-m', '--model', type=int, help="Model to use. [1-6] (2)", default=2)
    parser.add_argument('--alpha', type=float, help="Regularization parameter for the NN.", default=0.)
    parser.add_argument('--skipval', action="store_true", help="Skip validation for each step.")
    parser.add_argument('--nonorm', action="store_true", help="Do not normalize the data.")
    parser.add_argument('-cp', '--checkpoint', type=int, help="Write a checkpoint file after N epochs.", default=0)
    parser.add_argument('--cont', action="store_true", help="Continue last run with last saved checkpoint.")
    parser.add_argument('--json', action="store_true", help="Write/Load the model parameters to a JSON file.")
    parser.add_argument('--jsonfile', type=str, help="JSON file to read/write model parameters from/to.",
                        default=None)
    parser.add_argument('--root', action="store_true",
                        help="As a last step, create a ROOT RDataFrame and store to file.")
    parser.add_argument('--mcpart', action="store_true", help="Include the MC particle energy in output.")
    parser.add_argument('--rate', type=float, help="Set the training rate. (1e-4) ", default=1e-4)
    parser.add_argument('--momentum', type=float, help="Set the training momentum. (0.9)", default=0.9)
    parser.add_argument('input_files', type=str, nargs='+', help="Input files with data in the ROOT format.")
    args = parser.parse_args(argv[1:])

    R.gSystem.Load("/data/HPS/lib/libMiniDST")
    R.gSystem.Load("/data/HPS/Analysis/lib/libEcal_Analysis")
    R.gInterpreter.ProcessLine('''auto EAC = Ecal_Analysis_Class();''')  # This is key. It puts the EAC in C++ space.

    if args.debug > 0:
        print(f"MLTrainer_exy.py  : {__version__}")
        print(f"Tensorflow version: {tf.__version__}")
        print(f"Keras version     : {keras.__version__}")
        print(f"ROOT version      : {R.gROOT.GetVersion()}")
        print(f"ECAL Class version: {R.EAC.Version()}")
        print()

    input_files = args.input_files

    if args.debug > 0:
        print("Input files: ", input_files)
        print("Freeze: ", args.freeze)
        print()

    if args.jsonfile is None:
        data_file_name = f"CNN_Model_{args.model}.json"
    else:
        data_file_name = args.jsonfile

    data_file_name_root = os.path.splitext(data_file_name)[0]

    # Parameters that should be added to the options or are defaults.
    total_number_of_clusters = args.numevents  # Number of clusters in the output data.
    n_training = int(total_number_of_clusters * args.train//100)  # n_fid_clus//2
    n_validation = int(total_number_of_clusters * (100 - args.train)//100)  # n_fid_clus//2

    mini_batch_size = args.minibatch
    y_size = 12
    x_size = 47
    x_scaling = 100  # Scale the x and y variables to reduce the contribution to the loss.
    y_scaling = 100
    epoch = 0

    if not args.nonorm:
        # Standardize the data
        pass
    else:
        # Do not standardize the data
        pass

    #
    # Here we start the actual ML part.
    #
    activation_function = args.activation
    if activation_function == "leaky_relu":
        # activation_function = tf.nn.leaky_relu(alpha=0.01)
        activation_function = tf.keras.layers.LeakyReLU(alpha=0.01)

    # Build the models
    model = None
    if args.alpha > 1e-30:
        reg = tf.keras.regularizers.l1(args.alpha)
    else:
        reg = None

    def float_matches(y_true, y_pred, accuracy=2.e-2):
        """Creates int Tensor, 1 if y_true is equal to y_pred within accuracy, 0 if not.

        Args:
          y_true: Ground truth values, of shape (batch_size, d0, .. dN).
          y_pred: The predicted values, of shape (batch_size, d0, .. dN).
          accuracy: (Optional) Float representing the fractional accuracy for deciding
            whether prediction values are 1 or 0.

        Returns:
          Binary matches, of shape (batch_size, d0, .. dN).
        """
        y_pred = tf.convert_to_tensor(y_pred)
        accuracy = tf.cast(accuracy, y_pred.dtype)
        result = tf.less(tf.abs(y_true - y_pred) / y_true, accuracy)
        return tf.cast(result, keras.backend.floatx())

    class IntCompareAccuracy(tf.keras.metrics.Accuracy):
        """Compare whether the predicted float values round to the truth integer values."""

        def __init__(self, name='int_compare_accuracy', **kwargs):
            super(IntCompareAccuracy, self).__init__(name=name, **kwargs)

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred + 0.5, tf.int32)
            super().update_state(y_true, y_pred)

    class FloatCompareAccuracy(tf.keras.metrics.MeanMetricWrapper):
        """Compare whether the predicted float values are within Accuracy_tolerance of the truth float values."""

        def __init__(self, name='float_compare_accuracy', **kwargs):
            super().__init__(float_matches, name=name, **kwargs)

    if args.model == 1:
        checkpoint_path = "check_points/CNN_Model_1-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        class MyModel(Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self.flatten = Flatten()
                self.d1 = Dense(564, activation=activation_function, kernel_regularizer=reg)
                self.d2 = Dense(282, activation=activation_function, kernel_regularizer=reg)
                self.d3 = Dense(64, activation=activation_function, kernel_regularizer=reg)
                #self.dropout2 = Dropout(0.1)
                self.d4 = Dense(16, activation=activation_function, kernel_regularizer=reg)
                self.d5 = Dense(1, kernel_regularizer=reg)

            def call(self, x):
                x = self.flatten(x)
                x = self.d1(x)
                x = self.d2(x)
                x = self.d3(x)
                x = self.d4(x)
                x = self.d5(x)
                return x

    elif args.model == 2:
        checkpoint_path = "check_points/CNN_Model_2-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        class MyModel(Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = Conv2D(32, 5, activation=activation_function, kernel_regularizer=reg)
                self.conv2 = Conv2D(16, 3, activation=activation_function, kernel_regularizer=reg)
                self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
                self.conv3 = Conv2D(128, 3, activation=activation_function, kernel_regularizer=reg)
                self.dropout1 = Dropout(0.3)
                self.flatten = Flatten()
                self.d1 = Dense(128, activation=activation_function, kernel_regularizer=reg)
                self.dropout2 = Dropout(0.3)
                self.d2 = Dense(16, activation=activation_function, kernel_regularizer=reg)
                self.d3 = Dense(1, kernel_regularizer=reg)

            def call(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.maxpool1(x)
                x = self.conv3(x)
                x = self.dropout1(x)
                x = self.flatten(x)
                x = self.d1(x)
                x = self.dropout2(x)
                x = self.d2(x)
                x = self.d3(x)
                return x
    elif args.model == 3:
        checkpoint_path = "check_points/CNN_Model_3-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        class MyModel(Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = Conv2D(32, 5, activation=activation_function, kernel_regularizer=reg)
                self.conv2 = Conv2D(16, 3, activation=activation_function, kernel_regularizer=reg)
                self.maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
                self.conv3 = Conv2D(128, 3, activation=activation_function, kernel_regularizer=reg)
                self.dropout1 = Dropout(0.3)
                self.flatten = Flatten()
                self.d1 = Dense(128, activation=activation_function, kernel_regularizer=reg)
                self.dropout2 = Dropout(0.3)
                self.d2 = Dense(16, activation=activation_function, kernel_regularizer=reg)
                self.d3 = Dense(1, kernel_regularizer=reg)

            def call(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.maxpool1(x)
                x = self.conv3(x)
                x = self.dropout1(x)
                x = self.flatten(x)
                x = self.d1(x)
                x = self.dropout2(x)
                x = self.d2(x)
                x = self.d3(x)
                return x

    else:
        print("Model not implemented")
        exit(1)

    # Create an instance of the model
    model = MyModel()
    model.build(input_shape=(None, 12, 47, 1))

    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate, ema_momentum=args.momentum)

    train_loss = tf.keras.metrics.Mean(name='loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    train_accuracy = FloatCompareAccuracy(name='accuracy')
    validation_loss = tf.keras.metrics.Mean(name='validation_loss')
    # validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')
    validation_accuracy = FloatCompareAccuracy(name='validation_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        validation_loss(t_loss)
        validation_accuracy(labels, predictions)

    # --------------------------- Restore from JSON file --------------------------------
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}


    if args.cont:
        if args.debug > 1:
            print("Loading weights from file: ", data_file_name)
        weights = model.get_weights()
        with open(data_file_name, "r") as read_file:
            decodedArray = json.load(read_file)
            weights_store = decodedArray['model_weights']
            for i in range(len(weights_store)):
                weights[i] = np.array(weights_store[i])
            model.set_weights(weights)
            if "history" in decodedArray:
                history = decodedArray["history"]
            if "epoch" in decodedArray:
                epoch = decodedArray["epoch"]

    else:
        weights_store = []



    # --------------------------- Freezer --------------------------------
    # Parse the freeze argument
    if args.freeze is None or args.freeze.lower() == "none":
        freeze = []
    else:
        freeze = eval(args.freeze)  # Parse the freeze argument as list or number

    if type(freeze) is int:
        freeze = list(range(freeze, len(model.layers)))

    if type(freeze) is list or type(freeze) is tuple:
        if args.debug:
            print("Freezing: ", freeze)
        for i in freeze:
            model.get_layer(index=i).trainable = False


    if args.debug:
        print(f"Activation function = {activation_function}")
        print("st.dev = {args.random}  rate = {args.rate}  momentum={args.momentum}\n")
        model.summary()

    if args.debug > 2:
        print("Weights in the model at start:")
        weights = model.get_weights()
        for ii in range(len(weights)//2):
            print(f"Layer {ii:2d}:", weights[ii].shape, weights[ii + 1].shape)
            print(weights[2 * ii])
            print(weights[2 * ii + 1])

# --------------------------- Prepare the Data --------------------------------
# Needs optimization!!!!!!!
    ch = R.TChain("MiniDST")
    for f in input_files:
        ch.Add(f)
    mdst = R.MiniDst()  # Initiate the class
    #
    # Turning these branches off should speed up reading the data, but I do not observe any effect.
    #
    mdst.use_hodo_hits = False
    mdst.use_hodo_clusters = False
    mdst.use_svt_hits = False
    mdst.use_kf_tracks = False
    mdst.use_gbl_tracks = False
    mdst.use_matched_tracks = False
    mdst.use_gbl_kink_data = False
    mdst.use_kf_particles = False
    mdst.use_gbl_particles = False
    mdst.use_mc_particles = True  # Tell it to look for the MC Particles in the TTree
    mdst.use_ecal_cluster_uncor = True
    mdst.use_mc_scoring = True
    mdst.DefineBranchMap()  # Define the map of all the branches to the contents of the TTree
    mdst.SetBranchAddressesOnTree(ch)  # Connect the TChain (which contains the TTree) to the class.
    if args.debug:
        print(f"MminiDST version = {mdst._version_()}")
        print(f"Number of events in TChain: {ch.GetEntries():8d}")
    event = 0  # Starting event number.

    # We copy the data here into large Numpy arrays, maybe we need to do better.

    # Input training data: the last 1 is for 1 color channel, so we can use standard CNN trainer.
    ecal_hits = np.zeros([total_number_of_clusters, y_size, x_size, 1], dtype=np.float32)
    ecal_is_fiducial = np.zeros([total_number_of_clusters], dtype=bool)
    ecal_truth = np.zeros([total_number_of_clusters, 3])  # Truth is energy, x, y
    ecal_energy = np.zeros([total_number_of_clusters, 3])  # These are for verification of the end results.
    ecal_x = np.zeros([total_number_of_clusters, 2])
    ecal_y = np.zeros([total_number_of_clusters, 2])

    R.EAC.mc_score_indexes_are_sorted = True
    out_evt = 0
    n_event = 0
    more_clusters_than_score_clusters = 0  # Count errors.
    max_event = ch.GetEntries()
    while out_evt < total_number_of_clusters:
        ch.GetEntry(event)
        if event % 10000 == 0:
            print(".", end="", flush=True)
        if event % 100000 == 0:
            print(f"Event: {event:7d}", flush=True)
        cl_idx = R.EAC.get_score_cluster_indexes(mdst.mc_score_pz, mdst.mc_score_x, mdst.mc_score_y,
                                                 mdst.mc_score_z, mdst.ecal_cluster_x, mdst.ecal_cluster_y)
        score_e = R.EAC.get_score_cluster_e(cl_idx, mdst.mc_score_px, mdst.mc_score_py, mdst.mc_score_pz)
        score_x = R.EAC.get_score_cluster_loc(cl_idx, mdst.mc_score_x, mdst.mc_score_pz)
        score_y = R.EAC.get_score_cluster_loc(cl_idx, mdst.mc_score_y, mdst.mc_score_pz)

        is_fiducial = R.EAC.fiducial_cut(mdst.ecal_cluster_seed_ix, mdst.ecal_cluster_seed_iy)

        n_clust = len(mdst.ecal_cluster_uncor_energy)
        if n_clust > len(score_e):
            n_clust = len(score_e)  # This would be very rare.
            more_clusters_than_score_clusters += 1
        for i_ecal in range(n_clust):
            for i_echit in range(len(mdst.ecal_cluster_uncor_hits[i_ecal])):
                i_hit = mdst.ecal_cluster_uncor_hits[i_ecal][i_echit]
                x_loc = mdst.ecal_hit_index_x[i_hit]
                if x_loc < 0:
                    x_loc += 24
                else:
                    x_loc += 23
                y_loc = mdst.ecal_hit_index_y[i_hit] + 6
                ecal_hits[out_evt][y_loc, x_loc][0] = mdst.ecal_hit_energy[i_hit]
            if args.mcpart:
                mc_id = mdst.ecal_cluster_mc_id[i_ecal]
                ecal_energy[out_evt][2] = mdst.mc_part_energy[mc_id]
            else:
                ecal_energy[out_evt][2] = 0

            i_cl = cl_idx[i_ecal]

            # ecal_truth[out_evt][0] = mdst.mc_part_energy[mc_id]   ## To train for MC_particle energy truth
            # This requires the newer version ROOT data that contains the mdst.ecal_cluster_mc_id data.

            ecal_is_fiducial[out_evt] = is_fiducial[i_ecal]
            ecal_truth[out_evt][0] = score_e[i_ecal]
            ecal_truth[out_evt][1] = score_x[i_ecal] / x_scaling
            ecal_truth[out_evt][2] = score_y[i_ecal] / y_scaling
            ecal_energy[out_evt][0] = mdst.ecal_cluster_energy[i_ecal]
            ecal_energy[out_evt][1] = score_e[i_ecal]

            ecal_x[out_evt] = [mdst.ecal_cluster_x[i_ecal] / x_scaling, score_x[i_ecal] / x_scaling]
            ecal_y[out_evt] = [mdst.ecal_cluster_y[i_ecal] / y_scaling, score_y[i_ecal] / y_scaling]
            if args.cuts == 0:
                out_evt += 1
            elif args.cuts == 1 and is_fiducial[i_ecal]:
                out_evt += 1
            elif args.cuts == 2 and not is_fiducial[i_ecal]:
                out_evt += 1

            if out_evt >= total_number_of_clusters:
                break
        event += 1
        if event >= max_event:
            break
        n_event += 1

    print(f"n_event = {n_event}  out_evt = {out_evt}  current event = {event}")
    print(f"We got more ECal clusters than score clusters {more_clusters_than_score_clusters} times.")

    # Choose the data_slice, i.e. make a selection of which events we actually use. An array of ones means all events.
    # You can also make some cut here, i.e. choose fiducial region only.
    data_slice = None
    if args.cuts == 0:
        print("Selected all clusters.")
        data_slice = np.ones_like(ecal_is_fiducial)
    elif args.cuts == 1:
        print("Selected fiducial clusters.")
        data_slice = (ecal_is_fiducial == True)
    elif args.cuts == 2:
        print("Selected anti-fiducial clusters.")
        data_slice = (ecal_is_fiducial == False)
    else:
        print("No valid cut selection. Use -c 0, 1 or 2.")
        return 1

    # plt.imshow(np.sum(ecal_hits[:10000], axis=0), cmap="YlOrRd")
    # plt.title("All Ecal hits")
    # plt.show()
    # plt.imshow(np.sum(ecal_hits[data_slice][:10000], axis=0), cmap="YlOrRd")
    # plt.title("Ecal hits for clusters in slice")
    # plt.show()

    n_fid_clus = len(ecal_hits[data_slice])
    print(f"Used ECal clusters: {n_fid_clus:d} = {100 * n_fid_clus / out_evt:4.2f}%, "
          f"train: {n_training:d}, val: {n_validation:d}")
    if n_training + n_validation > n_fid_clus:
        print(f"We don't have that much data loaded. Load more. {n_training} + {n_training} > {n_fid_clus}")
        n_training = n_fid_clus // 2
        n_validation = n_fid_clus // 2
    fid_ecal_hits = ecal_hits[data_slice]
    fid_ecal_truth = ecal_truth[data_slice]
    x_train = fid_ecal_hits[:n_training]
    y_train = fid_ecal_truth[:n_training, 0:1]
    x_test = fid_ecal_hits[n_training:n_training + n_validation]
    y_test = fid_ecal_truth[n_training:n_training + n_validation, 0:1]

# --------------------------- Training --------------------------------
    if args.debug > 0:
        fit_debug = 1
    else:
        fit_debug = 0

    splits = [0]
    for i in range(args.split):
        splits.append(int((i+1)*len(x_train)/args.split))

    print(f"Start training {args.numepocs} epochs with batch size {mini_batch_size} and {args.split} splits")

    for i_epoc in range(args.numepocs):
        epoch += 1
        if args.debug:
            print(f"[{i_epoc:2d}]", end="")

        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()

        for i in range(len(x_train) // mini_batch_size):
            if args.debug and i % (len(x_train) // mini_batch_size // 10) == 0:
                print(f".", end="", flush=True)
            train_step(x_train[i * mini_batch_size:(i + 1) * mini_batch_size],
                       y_train[i * mini_batch_size:(i + 1) * mini_batch_size])

        if not args.skipval:
            for i in range(len(x_test) // mini_batch_size):
                if i % (len(x_test) // mini_batch_size // 10) == 0:
                    print(f"+", end="", flush=True)
                test_step(x_test[i * mini_batch_size:(i + 1) * mini_batch_size],
                          y_test[i * mini_batch_size:(i + 1) * mini_batch_size])

        history['loss'] += [float(train_loss.result().numpy())]
        history['accuracy'] += [float(train_accuracy.result().numpy())]
        history['val_loss'] += [float(validation_loss.result().numpy())]
        history['val_accuracy'] += [float(validation_accuracy.result().numpy())]
        print(
            f' Epoch {epoch}, '
            f'Loss: {history["loss"][-1]:10.7f}, '
            f'Acc: {history["accuracy"][-1] * 100:6.3f}%, '
            f'Val Loss: {history["val_loss"][-1] :10.7f}, '
            f'Val Acc: {history["val_accuracy"][-1] * 100:6.3f}%'
            , flush=True)

        if args.checkpoint > 0 and (i_epoc) % args.checkpoint == 0:
            if args.debug:
                print("Storing checkpoint.")
            model.save_weights(checkpoint_path.format(epoch=i_epoc))

    if len(history['loss']) > 0:
        print("\nFinal values:")
        print(f"Last Loss      : {history['loss'][-1]:12.6g}  Accuracy: {history['accuracy'][-1] * 100:6.3f}%")
        pred_train = model(x_train, training=False)
        loss_train = loss_object(y_train, pred_train)
        acc_train = validation_accuracy(y_train, pred_train)
        print(f"Training Loss  : {float(loss_train.numpy()):12.6g}  Accuracy: {float(acc_train.numpy()) * 100:6.3f}%")
        pred = model(x_test, training=False)
        loss = loss_object(y_test, pred)
        accuracy = validation_accuracy(y_test, pred)
        print(f"Validation Loss: {float(loss.numpy()):12.6g}  Accuracy: {float(accuracy.numpy())*100:6.3f}%")

        model_weights = model.get_weights()
        for h in history:
            if type(history[h]) is list:
                for i in range(len(history[h])):
                    history[h][i] = float(history[h][i])

        outdata = {"epochs": epoch, "history": history, "model_weights": model_weights}
        with open(data_file_name, "w") as write_file:
            json.dump(outdata, write_file, cls=NumpyArrayEncoder)
        print("Done writing serialized info into "+data_file_name)

    if args.root:
        print("Storing the prediction output in ROOT file:"
              + data_file_name_root + ".root ")
        opts = R.RDF.RSnapshotOptions()
        opts.fMode = "RECREATE"
        rdf = R.RDF.FromNumpy(
            {'ecal_cluster_energy': ecal_energy[data_slice][n_training:n_training + n_validation, 0].copy(),
             'score_e': ecal_energy[data_slice][n_training:n_training + n_validation, 1].copy(),
             'mc_part_energy': ecal_energy[data_slice][n_training:n_training + n_validation, 2].copy(),
             'truth_e': y_test[:n_validation, 0].copy(), 'pred_e': pred[:n_validation , 0].numpy().copy()})
        rdf.Snapshot("Validation", data_file_name_root + ".root", (""), opts)
        rdf_train = R.RDF.FromNumpy({'ecal_cluster_energy': ecal_energy[data_slice][:n_training, 0].copy(),
                                     'score_e': ecal_energy[data_slice][:n_training, 1].copy(),
                                     'mc_part_energy': ecal_energy[data_slice][:n_training, 2].copy(),
                                     'truth_e': y_train[:n_training, 0].copy(),
                                     'pred_e': pred_train[:, 0].numpy().copy()})
        opts.fMode = "UPDATE"
        rdf_train.Snapshot("Training", data_file_name_root + ".root", (""), opts)


if __name__ == "__main__":
    sys.exit(main())
