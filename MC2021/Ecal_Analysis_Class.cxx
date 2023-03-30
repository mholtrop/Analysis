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
// Return a list of lists of indexes to mc_score array hits.
// So index=get_score_indexes[cluster_number][hit_in_cluster_number]
// Conditions for a hit are that the track passed through the score plane at z=1443 in the positive direction with z_momentum > mc_score_pz_cut.
// Hits are grouped in clusters around the most energetic (in pz) hits if they are within mc_score_close_cut in x and y.

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

RVec< double > Ecal_Analysis_Class::get_score_cluster_loc(RVec< std::vector<int> > indexes,
                                                          RVec<double> mc_score_x, RVec<double> mc_score_pz){
   // Returns the cluster location. Pass mc_score_x to get x or mc_score_y to get y.
   // The cluster location is computed as the pz weighted average of the hit locations.
   RVec<double> out;
   for(size_t i=0; i<indexes.size(); ++i){
      double ave = 0;
      double sum = 0;
      for(size_t j=0; j< indexes[i].size(); ++j){
         ave += mc_score_x[indexes[i][j]]*mc_score_pz[indexes[i][j]];
         sum += mc_score_pz[indexes[i][j]];
      }
      out.push_back(ave/sum);
   }
   return out;
}

RVec< double > Ecal_Analysis_Class::get_score_cluster_pz(RVec< std::vector<int> > indexes, RVec<double> mc_score_pz){
   // Returns the summed pz (or py or px if passed those.)
   RVec<double> out;
   for(size_t i=0; i<indexes.size(); ++i){
      double sum = 0;
      for(size_t j=0; j< indexes[i].size(); ++j){
         sum += mc_score_pz[indexes[i][j]];
      }
      out.push_back(sum);
   }
   return out;
}

RVec< double > Ecal_Analysis_Class::get_score_cluster_e(RVec< std::vector<int> > indexes,
                                   RVec<double> mc_score_px, RVec<double> mc_score_py, RVec<double> mc_score_pz){
   // Return the energy as Sum(sqrt(px**2 + py**2 + pz**2))
   RVec<double> out;
   for(size_t i=0; i<indexes.size(); ++i){
      double sum = 0;
      for(size_t j=0; j< indexes[i].size(); ++j){
         int idx = indexes[i][j];
         sum += sqrt(mc_score_px[idx]*mc_score_px[idx]+mc_score_py[idx]*mc_score_py[idx]+mc_score_pz[idx]*mc_score_pz[idx]);
      }
      out.push_back(sum);
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

double Ecal_Analysis_Class::ecal_xpos_to_index(double xpos) {
   // Return the (double!!!) value of the index  given the x position. This is used to plot on an ix, iy histogram.
   int offset = 0;
   const std::vector<double> crystal_pos_x{-298.73819, -282.96774, -267.28214, -251.67653, -236.14636, -220.68710,
                                           -205.29440,
                                           -189.96397, -174.69170, -159.47348, -144.30539, -129.18352, -114.10406,
                                           -99.06329,
                                           -84.05750, -69.08309, -54.13647, -39.21410, -24.31249, -9.42816, 5.44233,
                                           20.30240,
                                           35.15547, 50.20495, 65.05803, 79.91810, 94.78859, 109.67291, 124.57452,
                                           139.49689,
                                           154.44351, 169.41792, 184.42371, 199.46448, 214.54393, 229.66580, 244.83391,
                                           260.05212, 275.32440, 290.65482, 306.04755, 321.50677, 337.03696, 352.64255,
                                           368.32819, 384.09863};

   auto const idx = std::lower_bound(crystal_pos_x.begin(), crystal_pos_x.end(), xpos);
   if (idx == crystal_pos_x.end()) { return -1.; }

   if (xpos > (crystal_pos_x[22] + crystal_pos_x[23]) / 2.) {
      offset = -22;
   } else {
      offset = -23;
   }

   int i_idx = std::distance(crystal_pos_x.begin(), idx);
// Interpolate between idx and idx+1.
   if (i_idx >= crystal_pos_x.size()) {
      i_idx = crystal_pos_x.size()-1;
   }
   return double(i_idx + offset) + (xpos - crystal_pos_x[i_idx]) / (crystal_pos_x[i_idx] - crystal_pos_x[i_idx-1]);
}

double Ecal_Analysis_Class::ecal_ypos_to_index(double ypos) {
   if(ypos>=0) {
      return -0.9954516016045554 + 0.0665927423975814 * ypos;
   }else {
      return 0.9355181571469946 + 0.06659274408814787 * ypos;
   }
}