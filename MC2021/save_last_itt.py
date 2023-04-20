#!/usr/bin/env python3

import numpy as np
from json import JSONEncoder
import json
import os
import argparse

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def main():

    parser = argparse.ArgumentParser("Save only the constants of the last iteration")
    parser.add_argument('input', type=str, help="Input file name")
    parser.add_argument('output', type=str, nargs="?", help="Output file name", default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.basename( os.path.splitext(args.input)[0] ) + "_last.json"

    print(f"Transform {args.input} to {args.output}")

    weights_store = None
    loss_store = None
    fit_mse_store = None
    val_mse_store = None
    with open(args.input, "r") as read_file:
        decoded_array = json.load(read_file)
        weights_store = decoded_array['weights_store']
        loss_store = decoded_array['loss_store']
        fit_mse_store = decoded_array['fit_mse_store']
        val_mse_store = decoded_array['val_mse_store']

    with open(args.output, "w") as write_file:
        json.dump({'weights_store': weights_store[-1:],
                   'loss_store': loss_store,
                   'fit_mse_store': fit_mse_store,
                   'val_mse_store': val_mse_store}, write_file)


if __name__ == "__main__":
    main()
