#!/usr/bin/env python3
#
# Transform the root files to a Pandas feather file.
#
import sys
sys.path.append("../Python")
import ROOT as R
from array import array
import time
import numpy as np
import pandas as pd
import argparse


def main(argv=None):
    """Main code is here."""
    R.EnableImplicitMT()
    R.gSystem.Load("lib/libMC2021")


    if argv is None:
        argv = sys.argv
    else:
        argv = argv.split()
        argv.insert(0, sys.argv[0])  # add the program name.

    parser = argparse.ArgumentParser(
        description="""This Python script converts a MC root file to a Pandas feather file.""",
        epilog="""
            For more info, read the script ^_^, or email maurik@physics.unh.edu.""")

    parser.add_argument('-d', '--debug', action="count", help="Be more verbose if possible. ", default=0)
    parser.add_argument('-f', '--fiducialcut', action="store_true", help="Apply fiducial cut.", default=False)
    parser.add_argument('-1', '--onecluster', action="store_true", help="Only use events with one cluster.",
                        default=False)
    parser.add_argument('-o', '--output', type=str, help="Output file name.", default="output.feather")
    parser.add_argument('input_files', nargs='+', type=str, help="Input root files.")

    args = parser.parse_args(argv[1:])

    R.gInterpreter.ProcessLine('''auto EAC = Ecal_Analysis_Class();''')  # This is key. It puts the EAC in C++ space.
    print(f"Ecal_Analysis_Class version is {R.EAC.Version()}")

    ch = R.TChain("MiniDST")
    for f in args.input_files:
        ch.Add(f)
    print(f"Added {ch.GetNtrees()} files to the chain for a total of {ch.GetEntries()/1e6:6.3f}M events.")
    print(f"Output going to {args.output}")

    df = R.RDataFrame(ch)
    dfx = R.EAC.extend_dataframe(R.RDF.AsRNode(df))

    if args.fiducialcut:
        dfx = dfx.Define("fiducial_cut",
                         "auto f=EAC.fiducial_cut(ecal_cluster_seed_ix,ecal_cluster_seed_iy); return f;") \
             .Filter(
            "for(size_t i=0; i< fiducial_cut.size(); ++i) if( fiducial_cut[i] == 0){ return false;} return true;")

    if args.onecluster:
        dfx = dfx.Filter("ecal_cluster_uncor_energy.size() == 1") \
            .Filter("ecal_cluster_energy.size() == 1") \
            .Filter("mc_score_cluster_indexes.size()==1")

        dfx = dfx.Define("mc_part_primary_energy", "return mc_part_energy[mc_part_primary_index[0]]")
        dfx = dfx.Define("energy", "ecal_cluster_uncor_energy[0]") \
            .Define("energy_cor", "ecal_cluster_energy[0]") \
            .Define("x", "ecal_cluster_uncor_x[0]") \
            .Define("y", "ecal_cluster_uncor_y[0]") \
            .Define("x_cor", "ecal_cluster_x[0]") \
            .Define("y_cor", "ecal_cluster_y[0]") \
            .Define("nhits", "ecal_cluster_uncor_nhits[0]") \
            .Define("seed_e", "ecal_cluster_uncor_seed_energy[0]") \
            .Define("seed_ix", "ecal_cluster_uncor_seed_ix[0]") \
            .Define("seed_iy", "ecal_cluster_uncor_seed_iy[0]") \
            .Define("true_e", "mc_part_primary_energy") \
            .Define("score_e", "mc_score_cluster_e[0]") \
            .Define("score_x", "mc_score_cluster_x[0]") \
            .Define("score_y", "mc_score_cluster_y[0]")

        cols = dfx.AsNumpy(['energy', 'energy_cor', 'x', 'y', 'x_cor', 'y_cor',
                            'nhits', 'seed_e', 'seed_ix', 'seed_iy', 'true_e',
                            'score_e', 'score_x', 'score_y'])
        panda_df = pd.DataFrame(cols)

    else:
        dfx = dfx.Define("mc_part_primary_energy", """
        vector<double> out;
        for(size_t i=0; i< mc_part_primary_index.size(); ++i){
            out.push_back(mc_part_energy[mc_part_primary_index[i]]);
        }
        return out;""")
        cols = dfx.AsNumpy(['ecal_cluster_uncor_energy', 'ecal_cluster_x', 'ecal_cluster_y',
                            'ecal_cluster_uncor_nhits', 'ecal_cluster_uncor_seed_energy',
                            'ecal_cluster_uncor_seed_ix', 'ecal_cluster_uncor_seed_iy',
                            'mc_part_primary_energy', 'mc_score_cluster_e', 'mc_score_cluster_x',
                            'mc_score_cluster_y'])
        panda_df = pd.DataFrame(cols)

    # now = time.time()
    # panda_df.to_feather(args.output+".feather")
    # delta_t = time.time() - now
    # print(f"Saved to {args.output}.feather in {delta_t:6.3f} seconds.")
    now = time.time()
    panda_df.to_hdf(args.output, 'data')
    delta_t = time.time() - now
    if args.debug:
        print(f"Saved to {args.output} in {delta_t:6.3f} seconds.")


if __name__ == "__main__":
    sys.exit(main())

