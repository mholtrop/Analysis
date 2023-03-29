/// Ecal_Analysis_Class
///
#include "Ecal_Analysis_Class.h"
#include <iostream>

RNode Ecal_Analysis_Class::extend_dataframe(RNode in){
   /// Return a dataframe with additional columns

   // To make use of methods in this class that are *not* static, we need to create some lambda function shims to pass
   // the mechod as a funcion to the dataframe.

   auto lambda_get_score_cluster_indexes = [this](RVec<double> mc_score_pz,
         RVec<double> mc_score_x, RVec<double> mc_score_y, RVec<double> mc_score_z){
      return get_score_cluster_indexes(mc_score_pz, mc_score_x, mc_score_y, mc_score_z);
   };

   auto lambda_get_score_primary_hits_energy = [this](RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z,
                                                  RVec<double> &mc_score_px, RVec<double> &mc_score_py, RVec<double> &mc_score_pz){
      return get_score_primary_hits_energy(mc_part_z, mc_score_part_idx, mc_score_z, mc_score_px, mc_score_py, mc_score_pz);};

   auto lambda_get_score_secondary_hits_energy = [this](RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_z,
                                                        RVec<double> &mc_score_px, RVec<double> &mc_score_py, RVec<double> &mc_score_pz){
      return get_score_secondary_hits_energy(mc_part_z, mc_score_part_idx, mc_score_z, mc_score_px, mc_score_py, mc_score_pz);};

   auto lambda_get_score_n_primary_hits = [this](RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
         RVec<double> &mc_score_z, RVec<double> &mc_score_pz){
      return get_score_n_primary_hits(mc_part_z, mc_score_part_idx, mc_score_z, mc_score_pz);
   };
   auto lambda_get_score_n_secondary_hits = [this](RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
         RVec<double> &mc_score_z, RVec<double> &mc_score_pz){
      return get_score_n_secondary_hits(mc_part_z, mc_score_part_idx, mc_score_z, mc_score_pz);
   };

   auto ret = in.Define("mc_score_cluster_indexes", lambda_get_score_cluster_indexes,{"mc_score_pz", "mc_score_x", "mc_score_y", "mc_score_z"});
   ret = ret.Define("mc_part_primary_index", get_list_of_primary_mc,{"mc_part_z"});
   ret = ret.Define("mc_score_primary_hits_e",lambda_get_score_primary_hits_energy,{"mc_part_z", "mc_score_part_idx", "mc_score_z",
   "mc_score_px", "mc_score_py", "mc_score_pz"});
   ret = ret.Define("mc_score_secondary_hits_e",lambda_get_score_secondary_hits_energy,{"mc_part_z", "mc_score_part_idx", "mc_score_z",
   "mc_score_px", "mc_score_py", "mc_score_pz"});
   ret = ret.Define("n_mc_score_primary_hits",lambda_get_score_n_primary_hits,{"mc_part_z", "mc_score_part_idx", "mc_score_z", "mc_score_pz"});
   ret = ret.Define("n_mc_score_secondary_hits",lambda_get_score_n_secondary_hits,{"mc_part_z", "mc_score_part_idx", "mc_score_z", "mc_score_pz"});
   return ret;
}

RVec< std::vector<int> > Ecal_Analysis_Class::get_score_cluster_indexes(
      RVec<double> mc_score_pz,
      RVec<double> mc_score_x, RVec<double> mc_score_y, RVec<double> mc_score_z){
// Return a list of lists of indexes to the mc_score data that form clusters of hits from tracks that
// passed through the score plane in the positive (pz>0) direction.
// This list of lists can then be used to make averages of mc_score quantities.

   RVec< std::vector<int> > out;
   std::vector<bool> has_been_used(mc_score_z.size(), false);
   bool not_done = true;
   while(not_done){
      double max_pz = mc_score_pz_cut;
      int index_max_pz = -1;
      for(size_t i=0; i< mc_score_z.size(); ++i){
         if( !has_been_used[i] && mc_score_z[i] > mc_score_z_position_cut && mc_score_pz[i] > max_pz ){
            max_pz = mc_score_pz[i];
            index_max_pz = i;
         }
      }

      if(index_max_pz < 0){
         not_done = false;
      }else {
         double x_loc = mc_score_x[index_max_pz];
         double y_loc = mc_score_y[index_max_pz];
         std::vector<int> cluster_list{index_max_pz};

         has_been_used[index_max_pz] = true;
         // Second loop to establish the cluster hits beloning to this max_pz
         for (size_t i = 0; i < mc_score_z.size(); ++i) {
            if (!has_been_used[i] && mc_score_z[i] > mc_score_z_position_cut && mc_score_pz[i] > mc_score_pz_cut &&
            abs(mc_score_x[i] - x_loc ) < mc_score_close_cut && abs(mc_score_y[i] - y_loc ) < mc_score_close_cut) {
               has_been_used[i] = true;
               cluster_list.push_back(i);
            }
         }
         out.push_back(cluster_list);
      }
   }
   return out;
}

RVec<bool> Ecal_Analysis_Class::fiducial_cut(RVec<int> ix, RVec<int> iy){ /// Fiducial cut for basic fiducial region.
   RVec<bool> out;
   for(size_t i=0;i< ix.size();++i){
      if(ficucial_cut_test(ix[i], iy[i]) )
      {
         out.push_back(true);
      }else{
         out.push_back(false);
      }
   }
   return out;
}

RVec<bool> Ecal_Analysis_Class::fiducial_cut_extended(RVec<int> ix, RVec<int> iy){
   RVec<bool> out;
   for(size_t i=0;i< ix.size();++i){
      if(ficucial_cut_test(ix[i], iy[i]))
      {
         bool do_we_include = true;
         for(const auto& cc : fiducial_cut_exclude) {
            if (ix[i] >= cc.first - 1 && ix[i] <= cc.first + 1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               // ix,iy is within the 3x3 exclusion zone for this crystal.
               do_we_include = false;
               break;
            }
            // This is where the decision to skip ix==0 bites. If first==1 then first-2 should be excluded too,
            // same for first==-1, then first+2 should be excluded.
            if (cc.first == 1 && ix[i] == -1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               do_we_include = false;
               break;
            }
            if (cc.first == -1 && ix[i] == 1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               do_we_include = false;
               break;
            }
         }
         out.push_back(do_we_include);
      }else{
         out.push_back(false);
      }
   }
   return out;
}

RVec<int> Ecal_Analysis_Class::get_score_n_primary_hits(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
                                                        RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const {
   RVec<int> out;
   for(size_t i_part=0; i_part < mc_part_z.size(); ++i_part){
      if(abs(mc_part_z[i_part]) < 1e-6 ){
         int count = 0;
         for(size_t i=0; i< mc_score_part_idx.size(); ++i) {
            if ((mc_score_z[i] > mc_score_z_position_cut) && (mc_score_pz[i] > mc_score_pz_cut) && mc_score_part_idx[i] == i_part) {
               ++count;
            }
         }
         out.push_back(count);
      }
   }
   return out;
}

RVec<int> Ecal_Analysis_Class::get_score_n_secondary_hits(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
                                                          RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const {
   RVec<int> out;
   for(size_t i_part=0; i_part < mc_part_z.size(); ++i_part){
      if(abs(mc_part_z[i_part]) >= 1e-6 ){
         int count = 0;
         double zsum = 0;
         for(size_t i=0; i< mc_score_part_idx.size(); ++i) {
            if ((mc_score_z[i] > mc_score_z_position_cut) && (mc_score_pz[i] > mc_score_pz_cut) &&
            mc_score_part_idx[i] == i_part) {
               zsum += mc_score_pz[i];
               ++count;
            }
         }
         if( zsum > 0.001)
            out.push_back(count);
      }
   }
   return out;
}

RVec<int> Ecal_Analysis_Class::get_list_of_primary_mc(RVec<double> &part_z) {
   /// Get a list of indexes to MCParticles (mc_part_*) of the primary particles, i.e. from the generator.
   /// We do not need to check for proximity, since hits caused by primaries should always be close anyway.
   RVec<int> out;
   for(size_t i=0; i< part_z.size(); ++i)
      if(abs(part_z[i]) < 1e-6) out.push_back(i);
   return out;
}

RVec<int> Ecal_Analysis_Class::get_list_of_all_secondary_mc(RVec<double> &part_z) {
   /// Get a list of all secondary MCParticles (mc_part_*), i.e. not primary ones.
   RVec<int> out;
   for(size_t i=0; i< part_z.size(); ++i)
      if(abs(part_z[i]) >= 1e-6) out.push_back(i);
   return out;
}

RVec<int> Ecal_Analysis_Class::get_centers_of_scored_secondary_mc(RVec<double> &part_z, RVec<int> &mc_score_part_idx, RVec<double> &mc_score_x,
                                                 RVec<double> &mc_score_y, RVec<double> &mc_score_z, RVec<double> &mc_score_pz) const{
   /// Returns a list of the indexes of the "seed" hits on the score plane, i.e. a list of the indexes of the highest
   /// momentum score hits caused by those secondary MCParticles (mc_part_*) that cross the ECal
   /// scoring plane in positive z direction, and are the highest pz within mc_score_close_cut distance in x and y.
   RVec<int> out;
   std::vector<bool> already_used(mc_score_z.size(), false);
   bool not_done = true;
   while(not_done) {
      double max_val = 0;
      int idx_current_max = -1;
      for (size_t i = 0; i < mc_score_z.size(); ++i){
         int i_part = mc_score_part_idx[i];
         if( !already_used[i] && abs(part_z[i_part]) >= 1e-6 && mc_score_z[i] > mc_score_z_position_cut &&
             mc_score_pz[i] > mc_score_pz_cut && mc_score_pz[i] > max_val ){
            not_done = true;
            idx_current_max = i;
            max_val = mc_score_pz[i];
         }
      }
      already_used[idx_current_max] = true;
      double x_center = mc_score_x[idx_current_max];
      double y_center = mc_score_y[idx_current_max];

      for (size_t ii = 0; ii < mc_score_z.size(); ++ii) {
         int i_part = mc_score_part_idx[ii];
         if (!already_used[ii] && abs(part_z[i_part]) >= 1e-6 && mc_score_z[ii] > mc_score_z_position_cut &&
             mc_score_pz[ii] > mc_score_pz_cut &&
             abs(mc_score_x[ii] - x_center) < mc_score_close_cut &&  abs(mc_score_y[ii] - y_center) < mc_score_close_cut){
            already_used[ii] = true;
         }
      }
      out.push_back(idx_current_max);
   }
   return out;
}


RVec<double> Ecal_Analysis_Class::get_score_primary_hits_energy(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
                                                                RVec<double> &mc_score_z, RVec<double> &mc_score_px,
                                                                RVec<double> &mc_score_py, RVec<double> &mc_score_pz) {
   RVec<double> out;
   for(size_t i_part=0; i_part < mc_part_z.size(); ++i_part){
      if(abs(mc_part_z[i_part]) < 1e-6 ) {
         double sum = 0;
         for (size_t i = 0; i < mc_score_part_idx.size(); ++i) {
            if ((mc_score_z[i] > mc_score_z_position_cut) && (mc_score_pz[i] > mc_score_pz_cut) &&
                mc_score_part_idx[i] == i_part) {
               sum += TMath::Sqrt(mc_score_px[i] * mc_score_px[i] + mc_score_py[i] * mc_score_py[i] + mc_score_pz[i] * mc_score_pz[i]);
            }
         }
         if( sum > 0.001)
            out.push_back(sum);
      }
   }
   return out;
}

RVec<double> Ecal_Analysis_Class::get_score_secondary_hits_energy(RVec<double> &mc_part_z, RVec<int> &mc_score_part_idx,
                                                                  RVec<double> &mc_score_z, RVec<double> &mc_score_px,
                                                                  RVec<double> &mc_score_py, RVec<double> &mc_score_pz) {
   RVec<double> out;
   for(size_t i_part=0; i_part < mc_part_z.size(); ++i_part){
      if(abs(mc_part_z[i_part]) >= 1e-6) {
         double sum = 0;
         for (size_t i = 0; i < mc_score_part_idx.size(); ++i) {
            if ((mc_score_z[i] > mc_score_z_position_cut) && (mc_score_pz[i] > mc_score_pz_cut) &&
                mc_score_part_idx[i] == i_part) {
               sum += TMath::Sqrt(mc_score_px[i] * mc_score_px[i] + mc_score_py[i] * mc_score_py[i] + mc_score_pz[i] * mc_score_pz[i]);
            }
         }
         if(sum > 0.01)
            out.push_back(sum);
      }
   }
   return out;
}
