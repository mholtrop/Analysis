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
    parser.add_argument('-c', '--corrected', action="store_true", help="Store the corrected energy and position "
                                                                       "for the clusters.", default=False)
    parser.add_argument('-o', '--output', type=str, help="Output file name.", default="output.feather")
    parser.add_argument('input_files', nargs='+', type=str, help="Input root files.")

    args = parser.parse_args(argv[1:])

    R.gSystem.Load("lib/libMC2021")
    R.gInterpreter.ProcessLine('''auto EAC = Ecal_Analysis_Class();''')  # This is key. It puts the EAC in C++ space.
    print(f"Ecal_Analysis_Class version is {R.EAC.Version()}")
    R.EAC.mc_score_indexes_are_sorted = True

    now = time.time()

    ch = R.TChain("MiniDST")
    for f in args.input_files:
        ch.Add(f)
    print(f"Added {ch.GetNtrees()} files to the chain for a total of {ch.GetEntries()/1e6:6.3f}M events.")
    print(f"Output going to {args.output}")

    if args.onecluster:
        R.EnableImplicitMT()
        df = R.RDataFrame(ch)
        dfx = R.EAC.extend_dataframe(R.RDF.AsRNode(df))

        if args.fiducialcut:
            dfx = dfx.Define("fiducial_cut",
                             "auto f=EAC.fiducial_cut(ecal_cluster_seed_ix,ecal_cluster_seed_iy); return f;") \
                .Filter(
                "for(size_t i=0; i< fiducial_cut.size(); ++i) if( fiducial_cut[i] == 0){ return false;} return true;")

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
        # We have multiple clusters in an event. Each cluster needs to be on its own row in the Pandas dataframe.

        R.gSystem.Load("/data/HPS/lib/libMiniDst")
        mdst = R.MiniDst()  # Initiate the class
        mdst.use_mc_particles = True  # Tell it to look for the MC Particles in the TTree
        mdst.use_ecal_cluster_uncor = True
        mdst.use_mc_scoring = True
        mdst.DefineBranchMap()  # Define the map of all the branches to the contents of the TTree
        mdst.SetBranchAddressesOnTree(ch)  # Connect the TChain (which contains the TTree) to the class.
        print(f"MminiDST version = {mdst._version_()}")

        # Some investigation (google search) shows that adding rows to a dataframe is not efficient.
        # A memory efficient way is the pre-allocate Numpy arrays and then fill these, then convert to a dataframe.
        mem_space = ch.GetEntries()*3  # Expecting no more than 3 clusters per event.
        evt_num = np.zeros(mem_space, dtype=np.int32)
        energy = np.zeros(mem_space, dtype=np.float32)
        x = np.zeros(mem_space, dtype=np.float32)
        y = np.zeros(mem_space, dtype=np.float32)
        nhits = np.zeros(mem_space, dtype=np.int32)
        seed_e = np.zeros(mem_space, dtype=np.float32)
        seed_ix = np.zeros(mem_space, dtype=np.int32)
        seed_iy = np.zeros(mem_space, dtype=np.int32)
        true_e = np.zeros(mem_space, dtype=np.float32)
        true_pdg = np.zeros(mem_space, dtype=np.int32)
        true_pdg_purity = np.zeros(mem_space, dtype=np.float32)
        score_e = np.zeros(mem_space, dtype=np.float32)
        score_x = np.zeros(mem_space, dtype=np.float32)
        score_y = np.zeros(mem_space, dtype=np.float32)

        # Loop over the events and store the data from each cluster the arrays.
        n_idx = 0
        for i_evt in range(ch.GetEntries()):
            ch.GetEntry(i_evt)
            if i_evt % 10000 == 0:
                print(f"Processing event {i_evt:6d}/{ch.GetEntries()}")

            # Use the Ecal_Analysis_Class to find the score indexes that match the clusters in the event.
            cl_idx = R.EAC.get_score_cluster_indexes(mdst.mc_score_pz, mdst.mc_score_x, mdst.mc_score_y,
                                                     mdst.mc_score_z, mdst.ecal_cluster_x, mdst.ecal_cluster_y)

            n_clusters = mdst.ecal_cluster_uncor_energy.size()
            if n_clusters > cl_idx.size():
                if args.debug > 0:
                    print(f"Warning: There are fewer score plane clusters {cl_idx.size()} than ECAl clusters {n_clusters}")
                n_clusters = cl_idx.size()

            for i_cl in range(n_clusters):
                evt_num[n_idx] = mdst.event_number
                if args.corrected:
                    energy[n_idx] = mdst.ecal_cluster_energy[i_cl]
                    x[n_idx] = mdst.ecal_cluster_x[i_cl]
                    y[n_idx] = mdst.ecal_cluster_y[i_cl]
                    nhits[n_idx] = mdst.ecal_cluster_nhits[i_cl]
                    seed_e[n_idx] = mdst.ecal_cluster_seed_energy[i_cl]
                    seed_ix[n_idx] = mdst.ecal_cluster_seed_ix[i_cl]
                    seed_iy[n_idx] = mdst.ecal_cluster_seed_iy[i_cl]
                else:
                    energy[n_idx] = mdst.ecal_cluster_uncor_energy[i_cl]
                    x[n_idx] = mdst.ecal_cluster_uncor_x[i_cl]
                    y[n_idx] = mdst.ecal_cluster_uncor_y[i_cl]
                    nhits[n_idx] = mdst.ecal_cluster_uncor_nhits[i_cl]
                    seed_e[n_idx] = mdst.ecal_cluster_uncor_seed_energy[i_cl]
                    seed_ix[n_idx] = mdst.ecal_cluster_uncor_seed_ix[i_cl]
                    seed_iy[n_idx] = mdst.ecal_cluster_uncor_seed_iy[i_cl]

                mc_part_id = mdst.ecal_cluster_mc_id[i_cl]
                true_e[n_idx] = mdst.mc_part_energy[mc_part_id]
                true_pdg[n_idx] = mdst.ecal_cluster_mc_pdg[i_cl]
                true_pdg_purity[n_idx] = mdst.ecal_cluster_mc_pdg_purity[i_cl]

                i_sc = cl_idx[i_cl][0]
                score_e[n_idx] = np.sqrt(mdst.mc_score_px[i_sc]**2 + mdst.mc_score_py[i_sc]**2 +
                                         mdst.mc_score_pz[i_sc]**2 + mdst.mc_part_mass[mc_part_id]**2)
                score_x[n_idx] = mdst.mc_score_x[i_sc]
                score_y[n_idx] = mdst.mc_score_y[i_sc]
                n_idx += 1
                if n_idx >= mem_space:
                    print(f"Warning: Reached maximum number of clusters {mem_space}")
                    break

    print("Create and fill the Dataframe.")
    panda_df = pd.DataFrame({'evt_num': evt_num, "energy": energy, "x": x, "y": y,
                             "nhits": nhits, "seed_e": seed_e, "seed_ix": seed_ix,
                             "seed_iy": seed_iy, "true_e": true_e, "true_pdg": true_pdg,
                             "true_pdg_purity": true_pdg_purity, "score_e": score_e,
                             "score_x": score_x, "score_y": score_y})

    delta_t = time.time() - now
    print(f"Processing time {delta_t:6.3f} seconds.")

    # now = time.time()
    # panda_df.to_feather(args.output+".feather")
    # delta_t = time.time() - now
    # print(f"Saved to {args.output}.feather in {delta_t:6.3f} seconds.")
    now = time.time()
    panda_df.to_hdf(args.output, 'data')
    delta_t = time.time() - now
    print(f"Saved to {args.output} in {delta_t:6.3f} seconds.")


if __name__ == "__main__":
    sys.exit(main())

