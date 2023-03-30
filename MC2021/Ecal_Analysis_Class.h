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
   vector< pair<int,int> > fiducial_cut_exclude;
   double mc_score_z_position_cut = 1400;
   double mc_score_pz_cut = 0.01;
   double mc_score_close_cut = 3*15.09; // About 3 crystal widths.
   // These values come from a fit to the MC data of ecal_hits.
   // crystal_offsets = [-2.2223839899158793,-3.4115017022831444,-0.9954516016045554,0.9355181571469946]
   // crystal_factors = [0.06600114645748535,0.06600114778752586,0.0665927423975814,0.06659274408814787]

public:
   string Version(){ return "V1.0.4";}

   // Note: I tried templating this, with instantiations to make the templates resolve. This works at the root prompt,
   // but in Python it could not resolve the correct template. Given that the RNode is a "wicked" complicated item, we just overload.
   RNode extend_dataframe(RNode in);
   // This one is useful for C++ code in ROOT prompt, though you could just cast there. Python does not seem to pick
   // it up. Instead for Python, call this method with: dfx = EAC.extend_dataframe(R.RDF.AsRNode(df))
   RNode extend_dateframe(RDataFrame in){ return extend_dataframe( (RNode)(in));}

   vector< vector<int> > get_score_cluster_indexes( vector<double> mc_score_pz,
         vector<double> mc_score_x, vector<double> mc_score_y, vector<double> mc_score_z);

   vector< double > get_score_cluster_loc(vector< vector<int> > indexes, vector<double> mc_score_x, vector<double> mc_score_pz);
   vector< double > get_score_cluster_pz(vector< vector<int> > indexes, vector<double> mc_score_pz);
   vector< double > get_score_cluster_e(vector< vector<int> > indexes,
                                      vector<double> mc_score_px, vector<double> mc_score_py, vector<double> mc_score_pz);

   static vector<int> get_list_of_primary_mc(vector<double> &part_z);
   static vector<int> get_list_of_all_secondary_mc(vector<double> &part_z);
   vector<int> get_centers_of_scored_secondary_mc(vector<double> &part_z, vector<int> &mc_score_part_idx, vector<double> &mc_score_x,
                                                    vector<double> &mc_score_y, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   vector<double> get_score_primary_hits_energy(vector<double> &mc_part_z, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                              vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz);
   vector<double> get_score_secondary_hits_energy(vector<double> &mc_part_z, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                              vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz);
   vector<int> get_score_n_primary_hits(vector<double> &mc_part_z, vector<int> &mc_score_part_idx, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   vector<int> get_score_n_secondary_hits(vector<double> &mc_part_z, vector<int> &mc_score_part_idx, vector<double> &mc_score_z, vector<double> &mc_score_pz) const;
   inline static bool ficucial_cut_test(int ix, int iy){  /// Test if ix and iy are in basic fiducial region.
      return !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
             !(iy <= -6 || iy >= 6)   && /* Cut out the top and bottom row */
             !(iy >= -1 && iy <= 1)   && /* Cut out the first row around the gap */
             !(iy >= -2 && iy <= 2 && ix >= -11 && ix <= 1);
   }

  void fiducial_cut_add_bad_crystal(int ix, int iy){  /// Add bad crystal to fiducial cut list.
      fiducial_cut_exclude.push_back( {ix, iy});
   }

   static vector<bool> fiducial_cut(vector<int> ix, vector<int> iy); /// Fiducial cut for basic fiducial region.
   vector<bool> fiducial_cut_extended(vector<int> ix, vector<int> iy);  /// Fiducial cut extended with bad crystals from list.

   static double ecal_xpos_to_index(double xpos);
   static double ecal_ypos_to_index(double ypos);


ClassDef(Ecal_Analysis_Class, 1)
};


#endif //MC2021_ECAL_ANALYSIS_CLASS_H
