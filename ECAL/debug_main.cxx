//
// Created by Maurik Holtrop on 3/31/23.
//
#include "TChain.h"
#include "MiniDst.h"
#include "Ecal_Analysis_Class.h"

int main(){
   auto ch = new TChain("MiniDST");
   ch->Add("/data/HPS/data/physrun2021/sim_2021/new_e-*.root");
   printf("Number of events loaded: %7.4f \n",ch->GetEntries()/1.e6);
   auto EAC = Ecal_Analysis_Class();
   auto mdst = MiniDst();
   mdst.use_mc_particles=true;  // Tell it to look for the MC Particles in the TTree
   mdst.use_ecal_cluster_uncor= true;
   mdst.use_mc_scoring =true;
   mdst.DefineBranchMap();      // Define the map of all the branches to the contents of the TTree
   mdst.SetBranchAddressesOnTree(ch); // Connect the TChain (which contains the TTree) to the class.
   printf("MiniDST version = %s\n",mdst._version_().c_str());

   int event = 0;
   EAC.mc_score_indexes_are_sorted = true;
   for(unsigned long i=0; i<ch->GetEntries(); ++i) {
      ch->GetEntry(i);
      auto idx = EAC.get_score_cluster_indexes(mdst.mc_score_pz, mdst.mc_score_x, mdst.mc_score_y, mdst.mc_score_z,
                                               mdst.ecal_cluster_x, mdst.ecal_cluster_y);

      if(i%10000 == 0){
         printf("event = %6lu\n",i);
      }
   }
}