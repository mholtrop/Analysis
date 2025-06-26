#ifndef MC2021_ECAL_ANALYSIS_CLASS_H
#define MC2021_ECAL_ANALYSIS_CLASS_H
///
/// This defines a class to help with the analysis of the ECal data, specifically from MC files.
///
/// Notes on usage:
///
/// In C++ or pure root, it can be used in the expected way:
/// using namespace ROOT;
/// using namespace ROOT::VecOps;
/// EnableImplicitMT();
/// gSystem->Load("lib/libMC2021");
/// auto ch = new TChain("MiniDST");
/// ch->Add("hpsForward_e-_*.root");
/// auto df = new RDataFrame(*ch);
/// auto EAC = Ecal_Analysis_Class()
/// auto dfx = df->Define("mc_part_primary_index",EAC.get_list_of_primary_mc,{"mc_part_z"})
/// auto hh = dfx.Histo1D("mc_part_primary_index")
/// hh->Draw()
///
/// To do the same in Python, we just need to be one step more clever and understand the interplay between Python and C++
/// We want the full class accessible and modifiable from Python, but *also* have the full class be accessible to ROOT.
/// Since the Python to ROOT communication is often hampered (as of March 2023), we need to construct the class in C++ space.
/// Example:
/// import ROOT as R
/// R.gSystem.Load("lib/libMC2021")
/// ch = R.TChain("MiniDST")
/// ch.Add("hpsForward_e-_*.root")
/// df = R.RDataFrame(ch)
/// R.gInterpreter.ProcessLine('''auto EAC = Ecal_Analysis_Class()''')   # This is key. It puts the EAC in C++ space.
/// print(R.EAC.Version())    # The class is now accessible through ROOT, and can be controlled from Python.
/// dfx = df.Define("mc_part_primary_index","EAC.get_list_of_primary_mc(mc_part_z)")  # The 3 arg syntax does not work!
/// hh = dfx.Histo1D("mc_part_primary_index")
/// cc0 = R.TCanvas("cc0","CC0",1200,600)
/// hh.Draw()
/// cc0.Draw()
///
///

#include <string>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;
#include "TObject.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDF/HistoModels.hxx"


using namespace ROOT;
using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

class Ecal_Analysis_Class {

public:

   bool   mc_score_indexes_are_sorted = true;
   vector< pair<int,int> > fiducial_cut_exclude;
   double mc_score_z_position_cut = 1400;
   double mc_score_pz_cut = 0.01;
   double mc_score_close_cut = 3*15.09; // About 3 crystal widths.
   // These values come from a fit to the MC data of ecal_hits.
   // crystal_offsets = [-2.2223839899158793,-3.4115017022831444,-0.9954516016045554,0.9355181571469946]
   // crystal_factors = [0.06600114645748535,0.06600114778752586,0.0665927423975814,0.06659274408814787]

   // From Andrea Celetano's code in Java: ClusterPositionCorrection2021.java:
   /**
   * This uses the corrected cluster energy to correct the position of the
   * cluster. This is to be used with 2021 data
         *
         * To determine these corrections, we simulated e+ e- and gamma at fixed
         * energies over the ECAL acceptance, sampled the true hit position with MC
   * scoring plane, and compared with the measured cluster position. We then
   * considered:
         *
               * dX vs X (dX = measured - true) ---> true=measured-dX dY vs Y (dY = measured -
                                                                                    * true) ---> true=measured-dY
         *
         * We then performed a fit to these with dX = q + m*X dY = q1 + t*Y if x < 0 ; =
   * q2 + t*Y if Y > 0
   *
   * See also A.C. Talk at Nov. 2022 collaboration meeting
   *
   * We then checked the dependency of the q,m, q1,q2, t parameters as a function
   * of the energy
         *
         * Electrons and Positrons: parameter(E) = p0 + p1*pow(E,p2) for all parameters
   * Photons: par(E) = p0 + p1*pow(E,p2) | par = q,m par(E) = (a + b*E + c*E*E)/(d
                                                                                 * + e*E + f*E*E) | par = q1,t,q2
   **/
   // Variables for positron position corrections.
    double  POSITRON_POS_Q_P0{1.35533};
    double  POSITRON_POS_Q_P1{5.72332};
    double  POSITRON_POS_Q_P2{-0.541438};

    double  POSITRON_POS_M_P0{-0.0340964};
    double  POSITRON_POS_M_P1{0.014045};
    double  POSITRON_POS_M_P2{-0.545433};

    double  POSITRON_POS_Q1_P0{-3.21226};
    double  POSITRON_POS_Q1_P1{0.339324};
    double  POSITRON_POS_Q1_P2{-2.72148};

    double  POSITRON_POS_T_P0{-0.0362339};
    double  POSITRON_POS_T_P1{0.00449926};
    double  POSITRON_POS_T_P2{-2.91123};

    double  POSITRON_POS_Q2_P0{2.24442};
    double  POSITRON_POS_Q2_P1{-0.282654};
    double  POSITRON_POS_Q2_P2{-3.20633};

   // Variables for electron position corrections.
    double  ELECTRON_POS_Q_P0{5.05789};
    double  ELECTRON_POS_Q_P1{-7.63708};
    double  ELECTRON_POS_Q_P2{-0.593751};

    double  ELECTRON_POS_M_P0{-0.0318827};
    double  ELECTRON_POS_M_P1{0.0100568};
    double  ELECTRON_POS_M_P2{-0.676475};

    double  ELECTRON_POS_Q1_P0{-2.71442};
    double  ELECTRON_POS_Q1_P1{-0.456846};
    double  ELECTRON_POS_Q1_P2{-0.772825};

    double  ELECTRON_POS_T_P0{-0.0275841};
    double  ELECTRON_POS_T_P1{-0.00844973};
    double  ELECTRON_POS_T_P2{-0.628533};

    double  ELECTRON_POS_Q2_P0{1.72361};
    double  ELECTRON_POS_Q2_P1{0.524511};
    double  ELECTRON_POS_Q2_P2{-0.697755};

   // Variables for photon position corrections.
    double  PHOTON_POS_Q_P0{5.53107};
    double  PHOTON_POS_Q_P1{-2.71633};
    double  PHOTON_POS_Q_P2{-0.157991};

    double  PHOTON_POS_M_P0{-0.0645554};
    double  PHOTON_POS_M_P1{0.0376413};
    double  PHOTON_POS_M_P2{-0.119576};

    double  PHOTON_POS_Q1_P0{-3.99615};
    double  PHOTON_POS_Q1_P1{0.967692};
    double  PHOTON_POS_Q1_P2{-0.317565};

    double  PHOTON_POS_T_P0{-0.0582419};
    double  PHOTON_POS_T_P1{0.0247615};
    double  PHOTON_POS_T_P2{-0.233813};

    double  PHOTON_POS_Q2_P0{3.1551};
    double  PHOTON_POS_Q2_P1{-1.09655};
    double  PHOTON_POS_Q2_P2{-0.283272};


public:
   string Version(){ return "V1.1.0";}

   // Note: I tried templating this, with instantiations to make the templates resolve. This works at the root prompt,
   // but in Python it could not resolve the correct template. Given that the RNode is a "wicked" complicated item, we just overload.
   RNode extend_dataframe(RNode in);
   // This one is useful for C++ code in ROOT prompt, though you could just cast there. Python does not seem to pick
   // it up. Instead for Python, call this method with: dfx = EAC.extend_dataframe(R.RDF.AsRNode(df))
   RNode extend_dateframe(RDataFrame in){ return extend_dataframe( (RNode)(in));}
   RNode dataframe_for_ml(RNode in);
   RNode dataframe_for_ml(RDataFrame in){ return dataframe_for_ml( (RNode) in);}

//   vector<int> get_cluster_pdg(vector<vector<int>> &cluster_hits, vector<int> &parent_pdg, vector<double> &hit_energy);
//   vector<double> get_cluster_pdg_purity(vector<vector<int>> &cluster_hits, vector<int> &parent_pdg, vector<double> &hit_energy);

   vector< vector<int> > get_score_cluster_indexes( vector<double> &mc_score_pz,
         vector<double> &mc_score_x, vector<double> &mc_score_y, vector<double> &mc_score_z,
         vector<double> &ecal_cluster_x, vector<double> &ecal_cluster_y);

   vector< double > get_score_cluster_loc(vector< vector<int> > &indexes, vector<double> &mc_score_x, vector<double> &mc_score_pz);
   vector< double > get_score_cluster_pz(vector< vector<int> > &indexes, vector<double> &mc_score_pz);
   vector< double > get_score_cluster_e(vector< vector<int> > &indexes,
                                      vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz);

   static vector<int> get_list_of_primary_mc(vector<double> &part_z);
   static vector<int> get_list_of_primary_mc(vector<double> &part_z, vector<double> &part_pz);
   static vector<int> get_list_of_primary_mc(vector<int> &mc_part_sim_status);
   static vector<int> get_list_of_all_secondary_mc(vector<double> &part_z);
   static vector<int> get_list_of_all_secondary_mc(vector<int> &mc_part_sim_status);

   vector<int> get_centers_of_scored_secondary_mc(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_x,
                                                    vector<double> &mc_score_y, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   vector<double> get_score_primary_hits_energy(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                              vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz);
   vector<double> get_score_secondary_hits_energy(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                              vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz);
   vector<int> get_score_n_primary_hits(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   vector<int> get_score_n_secondary_hits(vector<int> &mc_part_sim_staus, vector<int> &mc_score_part_idx, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   inline static bool fiducial_cut_test(int ix, int iy){  /// Test if ix and iy are in basic fiducial region.
      return !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
             !(iy <= -6 || iy >= 6)   && /* Cut out the top and bottom row */
             !(iy >= -1 && iy <= 1)   && /* Cut out the first row around the gap */
             !(iy >= -2 && iy <= 2 && ix >= -11 && ix <= 1);
   }

  void fiducial_cut_add_bad_crystal(int ix, int iy){  /// Add bad crystal to fiducial cut list.
      fiducial_cut_exclude.push_back( {ix, iy});
   }
   /// Note:
   /// A bit of a hassle, but, when calling a function with RDataFrame columns, in C++ these are vector<> type,
   /// In Python, because of the PyROOT interface, these functions are called with RVec<> type. C++ will not convert one
   /// to another automatically, so we need to overload and have both functions. The following pair allows for:
   /// df.Define("fid_cut_result",fiducial_cut,{"ecal_cluster_seed_ix","ecal_cluster_seed_iy"); // From C++, uses vector<>
   /// df.Define("fid_cut_result","EAC.fiducial_cut(ecal_cluster_seed_ix,ecal_cluster_seed_iy)") // From Python, uses RVec<>
   ///
   static vector<bool> fiducial_cut(vector<int> &ix, vector<int> &iy); /// Fiducial cut for basic fiducial region.
   static RVec<bool> fiducial_cut(RVec<int> ix, RVec<int> iy); /// Fiducial cut for basic fiducial region, for RVec's
   vector<bool> fiducial_cut_extended(vector<int> &ix, vector<int> &iy);  /// Fiducial cut extended with bad crystals from list.
   RVec<int> fiducial_cut_extended(RVec<int> &ix, RVec<int> &iy);  /// RVec version

   static double ecal_xpos_to_index(double xpos);
   static double ecal_ypos_to_index(double ypos);

    double ecal_xpos_correction_electron(double xpos, double energy);
    double ecal_ypos_correction_electron(double ypos, double energy);
    double ecal_xpos_correction_positron(double xpos, double energy);
    double ecal_ypos_correction_positron(double ypos, double energy);
    double ecal_xpos_correction_photon(double xpos, double energy);
    double ecal_ypos_correction_photon(double ypos, double energy);

ClassDef(Ecal_Analysis_Class, 1)
};


#endif //MC2021_ECAL_ANALYSIS_CLASS_H
