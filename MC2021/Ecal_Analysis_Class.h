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
#include "TObject.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "ROOT/RDF/HistoModels.hxx"
#include "ROOT/RVec.hxx"

using namespace ROOT;
using namespace ROOT::VecOps;
using RNode = ROOT::RDF::RNode;

class Ecal_Analysis_Class {

public:
   std::vector< std::pair<int,int> > fiducial_cut_exclude;
   double mc_score_z_position_cut = 1400;
   double mc_score_pz_cut = 0.01;
   double mc_score_close_cut = 2.;

public:
   std::string Version(){ return "V1.0.2";}

   // Note: I tried templating this, with instantiations to make the templates resolve. This works at the root prompt,
   // but in Python it could not resolve the correct template. Given that the RNode is a "wicked" complicated item, we just overload.
   RNode extend_dataframe(RNode in);
   // This one is useful for C++ code in ROOT prompt, though you could just cast there. Python does not seem to pick
   // it up. Instead for Python, call this method with: dfx = EAC.extend_dataframe(R.RDF.AsRNode(df))
   RNode extend_dateframe(RDataFrame in){ return extend_dataframe( (RNode)(in));}

   RVec< std::vector<int> > get_score_cluster_indexes( RVec<double> mc_score_pz,
         RVec<double> mc_score_x, RVec<double> mc_score_y, RVec<double> mc_score_z);



   static RVec<int> get_list_of_primary_mc(RVec<double> &part_z);
   static RVec<int> get_list_of_all_secondary_mc(RVec<double> &part_z);
   RVec<int> get_centers_of_scored_secondary_mc(RVec<double> &part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_x,
                                                    RVec<double> &mc_score_y, RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const;
   RVec<double> get_score_primary_hits_energy(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z,
                                              RVec<double> &mc_score_px, RVec<double> &mc_score_py, RVec<double> &mc_score_pz);
   RVec<double> get_score_secondary_hits_energy(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z,
                                              RVec<double> &mc_score_px, RVec<double> &mc_score_py, RVec<double> &mc_score_pz);
   RVec<int> get_score_n_primary_hits(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const;
   RVec<int> get_score_n_secondary_hits(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const;
   inline static bool ficucial_cut_test(int ix, int iy){  /// Test if ix and iy are in basic fiducial region.
      return !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
             !(iy <= -6 || iy >= 6)   && /* Cut out the top and bottom row */
             !(iy >= -1 && iy <= 1)   && /* Cut out the first row around the gap */
             !(iy >= -2 && iy <= 2 && ix >= -11 && ix <= 1);
   }

  void fiducial_cut_add_bad_crystal(int ix, int iy){  /// Add bad crystal to fiducial cut list.
      fiducial_cut_exclude.push_back( {ix, iy});
   }

   static RVec<bool> fiducial_cut(RVec<int> ix, RVec<int> iy); /// Fiducial cut for basic fiducial region.
   RVec<bool> fiducial_cut_extended(RVec<int> ix, RVec<int> iy);  /// Fiducial cut extended with bad crystals from list.

   ClassDef(Ecal_Analysis_Class, 1)
};


#endif //MC2021_ECAL_ANALYSIS_CLASS_H