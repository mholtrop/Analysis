/// Ecal_Analysis_Class
///
#include "Ecal_Analysis_Class.h"
#include <iostream>

RNode Ecal_Analysis_Class::dataframe_for_ml(RNode in) {
   /// Return a dataframe that is "flattened" for use in ML systems.
   /// This will only return some of the ECAL and Score plane data.
}

RNode Ecal_Analysis_Class::extend_dataframe(RNode in){
   /// Return a dataframe with additional columns

   // To make use of methods in this class that are *not* static, we need to create some lambda function shims to pass
   // the mechod as a funcion to the dataframe.

   auto lambda_get_score_cluster_indexes = [this](vector<double> &mc_score_pz,
         vector<double> &mc_score_x, vector<double> &mc_score_y, vector<double> &mc_score_z,
         vector<double> &ecal_cluster_x, vector<double> &ecal_cluster_y ){
      return get_score_cluster_indexes(mc_score_pz, mc_score_x, mc_score_y, mc_score_z, ecal_cluster_x, ecal_cluster_y);
   };

   auto lambda_get_score_cluster_loc = [this](vector< vector<int> > &indexes,
                                              vector<double> &mc_score_x, vector<double> &mc_score_pz){
      return get_score_cluster_loc(indexes, mc_score_x, mc_score_pz);
   };

   auto lambda_get_score_cluster_pz = [this](vector< vector<int> > &indexes, vector<double> &mc_score_pz){
      return get_score_cluster_pz(indexes, mc_score_pz);
   };

   auto lambda_get_score_cluster_e = [this](vector< vector<int> > &indexes, vector<double> &mc_score_px,
         vector<double> &mc_score_py, vector<double> &mc_score_pz){
      return get_score_cluster_e(indexes, mc_score_px, mc_score_py, mc_score_pz);
   };

   auto lambda_get_score_primary_hits_energy = [this](vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                                  vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz){
      return get_score_primary_hits_energy(mc_part_sim_status, mc_score_part_idx, mc_score_z, mc_score_px, mc_score_py, mc_score_pz);};

   auto lambda_get_score_secondary_hits_energy = [this](vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_z,
                                                        vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz){
      return get_score_secondary_hits_energy(mc_part_sim_status, mc_score_part_idx, mc_score_z, mc_score_px, mc_score_py, mc_score_pz);};

   auto lambda_get_score_n_primary_hits = [this](vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
         vector<double> &mc_score_z, vector<double> &mc_score_pz){
      return get_score_n_primary_hits(mc_part_sim_status, mc_score_part_idx, mc_score_z, mc_score_pz);
   };
   auto lambda_get_score_n_secondary_hits = [this](vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
         vector<double> &mc_score_z, vector<double> &mc_score_pz){
      return get_score_n_secondary_hits(mc_part_sim_status, mc_score_part_idx, mc_score_z, mc_score_pz);
   };

   auto ret = in.Define("mc_score_cluster_indexes", lambda_get_score_cluster_indexes,
                        {"mc_score_pz", "mc_score_x", "mc_score_y", "mc_score_z", "ecal_cluster_x", "ecal_cluster_y"});
   ret = ret.Define("mc_score_cluster_x",lambda_get_score_cluster_loc,{"mc_score_cluster_indexes","mc_score_x","mc_score_pz"});
   ret = ret.Define("mc_score_cluster_y",lambda_get_score_cluster_loc,{"mc_score_cluster_indexes","mc_score_y","mc_score_pz"});
   ret = ret.Define("mc_score_cluster_pz",lambda_get_score_cluster_pz,{"mc_score_cluster_indexes","mc_score_pz"});
   ret = ret.Define("mc_score_cluster_e",lambda_get_score_cluster_e,{"mc_score_cluster_indexes","mc_score_px","mc_score_py","mc_score_pz"});
   // Note the static cast to disambiguate the overloaded operator. Isn't C++ fun? :-)
   ret = ret.Define("mc_part_primary_index", static_cast< vector<int>(*)(vector<int> &) >(get_list_of_primary_mc),{"mc_part_sim_status"});
   ret = ret.Define("mc_score_primary_hits_e",lambda_get_score_primary_hits_energy,{"mc_part_sim_status", "mc_score_part_idx", "mc_score_z",
   "mc_score_px", "mc_score_py", "mc_score_pz"});
   ret = ret.Define("mc_score_secondary_hits_e",lambda_get_score_secondary_hits_energy,{"mc_part_sim_status", "mc_score_part_idx", "mc_score_z",
   "mc_score_px", "mc_score_py", "mc_score_pz"});
   ret = ret.Define("n_mc_score_primary_hits",lambda_get_score_n_primary_hits,{"mc_part_sim_status", "mc_score_part_idx", "mc_score_z", "mc_score_pz"});
   ret = ret.Define("n_mc_score_secondary_hits",lambda_get_score_n_secondary_hits,{"mc_part_sim_status", "mc_score_part_idx", "mc_score_z", "mc_score_pz"});
   return ret;
}

vector< vector<int> > Ecal_Analysis_Class::get_score_cluster_indexes(
      vector<double> &mc_score_pz,
      vector<double> &mc_score_x, vector<double> &mc_score_y, vector<double> &mc_score_z,
      vector<double> &ecal_cluster_x, vector<double> &ecal_cluster_y){
// Return a list of lists of indexes to mc_score array hits.
// So index[cluster_number][hit_in_cluster_number] returns the index to that hit in that cluster.
// Conditions for a hit are that the track passed through the score plane at z=1443 in the positive direction with z_momentum > mc_score_pz_cut.
// Hits are grouped in clusters around the most energetic (in pz) hits if they are within mc_score_close_cut in x and y.
// The *clusters* are sorted to be in the same order as those in the ecal_cluster_x and ecal_cluster_y arrays, approcimately matching by location.

   vector< vector<int> > out;        // TODO: is there a more efficient storage model? out total size = total number of hits.
   vector<bool> has_been_used(mc_score_z.size(), false);
   vector<double> ave_x;  // Average x position of cluster in out, for sorting
   vector<double> ave_y;  // Average y position of cluster in out, for sorting
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
         vector<int> cluster_list{index_max_pz};
         double sum_pz = mc_score_pz[index_max_pz];
         double ave_x_tmp = x_loc*sum_pz;
         double ave_y_tmp = y_loc*sum_pz;
         has_been_used[index_max_pz] = true;
         // Second loop to establish the cluster hits belonging to this max_pz
         for (size_t i = 0; i < mc_score_z.size(); ++i) {
            if (!has_been_used[i] && mc_score_z[i] > mc_score_z_position_cut && mc_score_pz[i] > mc_score_pz_cut &&
            abs(mc_score_x[i] - x_loc ) < mc_score_close_cut && abs(mc_score_y[i] - y_loc ) < mc_score_close_cut) {
               has_been_used[i] = true;
               cluster_list.push_back(i);
               ave_x_tmp += mc_score_x[i]*mc_score_pz[i];
               ave_y_tmp += mc_score_y[i]*mc_score_pz[i];
               sum_pz += mc_score_pz[i];
            }
         }
         out.push_back(cluster_list);
         ave_x.push_back(ave_x_tmp / sum_pz);
         ave_y.push_back(ave_y_tmp / sum_pz);
      }
   }

   if(mc_score_indexes_are_sorted) {
      // Here we perform the sorting pass.
      // We need to make sure that we match the closest clusters, which can be incorrect if simply matching the first
      // ECal cluster to the closest score plane cluster, since that score plane cluster may be even closer to another ECal
      // cluster. We thus do the extra work of building a matrix of distances, so we can make sure we have the best matching.

      vector<vector<int> > out_sorted;
      out_sorted.reserve(out.size());
      int w = ecal_cluster_x.size();
      int d = ave_x.size(); // == out.size()
      vector<double> dist_mat(w * d);
      for (size_t i = 0; i < w; ++i) {
         for (size_t j = 0; j < d; ++j) {
            double diffx = ave_x[j] - ecal_cluster_x[i];
            double diffy = ave_y[j] - ecal_cluster_y[i];
            double dist = sqrt(diffx * diffx + diffy * diffy);
            dist_mat[i * d + j] = dist;
         }
      }

      vector<bool> already_used(d, false);
      for (size_t i = 0; i < w; ++i) {
         double min_dist = 10000000000000.;
         int min_j = -1;
         for (size_t j = 0; j < d; ++j) {
            if (!already_used[j] && dist_mat[i * d + j] < min_dist) {
               bool is_closest = true;
               for (size_t ii = 0; ii < w; ++ii) {  // We need to check if there is another closer match for this j
                  if (dist_mat[ii * d + j] < dist_mat[i * d + j]) {  // Another is closer, so do not use this one.
                     is_closest = false;
                  }
               }
               if (is_closest) {
                  min_dist = dist_mat[i * d + j];
                  min_j = j;
               }
            }
         }
         // At this point, min_j should contain the closest valid match.
         if( min_j >= 0) {
            already_used[min_j] = true;  // Mark it as used.
            out_sorted.emplace_back(out[min_j]); // This, unfortunately, copies the vector.
         }else{
            // printf("Strange, min_j = %d for len(out) = %lu at i= %2zu\n",min_j, out.size(), i);
         }
      }
      for (size_t j = 0; j < d; ++j) {   // sweep the remaining "j"'s
         if (!already_used[j]) out_sorted.emplace_back(out[j]);
      }
      return out_sorted;
   }else{
      return out;
   }
}

vector< double > Ecal_Analysis_Class::get_score_cluster_loc(vector< vector<int> > &indexes,
                                                          vector<double> &mc_score_x, vector<double> &mc_score_pz){
   // Returns the cluster location. Pass mc_score_x to get x or mc_score_y to get y.
   // The cluster location is computed as the pz weighted average of the hit locations.
   vector<double> out;
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

vector< double > Ecal_Analysis_Class::get_score_cluster_pz(vector< vector<int> > &indexes, vector<double> &mc_score_pz){
   // Returns the summed pz (or py or px if passed those.)
   vector<double> out;
   for(size_t i=0; i<indexes.size(); ++i){
      double sum = 0;
      for(size_t j=0; j< indexes[i].size(); ++j){
         sum += mc_score_pz[indexes[i][j]];
      }
      out.push_back(sum);
   }
   return out;
}

vector< double > Ecal_Analysis_Class::get_score_cluster_e(vector< vector<int> > &indexes,
                                   vector<double> &mc_score_px, vector<double> &mc_score_py, vector<double> &mc_score_pz){
   // Return the energy as Sum(sqrt(px**2 + py**2 + pz**2))
   vector<double> out;
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



vector<bool> Ecal_Analysis_Class::fiducial_cut(vector<int> &ix, vector<int> &iy){ /// Fiducial cut for basic fiducial region.
   vector<bool> out;
   for(size_t i=0;i< ix.size();++i){
      if(fiducial_cut_test(ix[i], iy[i]) )
         out.push_back(true);
      else
         out.push_back(false);
   }
   return out;
}

RVec<bool> Ecal_Analysis_Class::fiducial_cut(RVec<int> ix, RVec<int> iy){ /// Fiducial cut for basic fiducial region.
   RVec<int> out;
   for(size_t i=0;i< ix.size();++i){
      if(fiducial_cut_test(ix[i], iy[i]) )
         out.push_back(true);
      else
         out.push_back(false);
   }
   return out;
}



vector<bool> Ecal_Analysis_Class::fiducial_cut_extended(vector<int> &ix, vector<int> &iy){
   vector<bool> out;
   printf("Vector version called.\n");
   for(size_t i=0;i< ix.size();++i){
      if(fiducial_cut_test(ix[i], iy[i]))
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

RVec<int> Ecal_Analysis_Class::fiducial_cut_extended(RVec<int> &ix, RVec<int> &iy){
   RVec<int> out;
   for(size_t i=0;i< ix.size();++i){
      if(fiducial_cut_test(ix[i], iy[i]))
      {
         int do_we_include = 1;
         for(const auto& cc : fiducial_cut_exclude) {
            if (ix[i] >= cc.first - 1 && ix[i] <= cc.first + 1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               // ix,iy is within the 3x3 exclusion zone for this crystal.
               do_we_include = 0;
               break;
            }
            // This is where the decision to skip ix==0 bites. If first==1 then first-2 should be excluded too,
            // same for first==-1, then first+2 should be excluded.
            if (cc.first == 1 && ix[i] == -1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               do_we_include = 0;
               break;
            }
            if (cc.first == -1 && ix[i] == 1 && iy[i] >= cc.second - 1 && iy[i] <= cc.second + 1) {
               do_we_include = 0;
               break;
            }
         }
         out.push_back(do_we_include);
      }else{
         out.push_back(0);
      }
   }
   return out;
}



vector<int> Ecal_Analysis_Class::get_score_n_primary_hits(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
                                                        vector<double> &mc_score_z, vector<double> &mc_score_pz) const {
   vector<int> out;
   for(size_t i_part=0; i_part < mc_part_sim_status.size(); ++i_part){
      if( uint(mc_part_sim_status[i_part]) & 0x01){
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

vector<int> Ecal_Analysis_Class::get_score_n_secondary_hits(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
                                                          vector<double> &mc_score_z, vector<double> &mc_score_pz) const {
   vector<int> out;
   for(size_t i_part=0; i_part < mc_part_sim_status.size(); ++i_part){
      if( (uint(mc_part_sim_status[i_part]) & 0x1 ) == 0 ){
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

vector<int> Ecal_Analysis_Class::get_list_of_primary_mc(vector<double> &part_z) {
/// Get a list of indexes to MCParticles (mc_part_*) of the primary particles, i.e. from the generator.
/// This version is based on the starting point of the particle being in the target.
   vector<int> out;
   for(size_t i=0; i< part_z.size(); ++i)
      if(abs(part_z[i]) < 1e-6) out.push_back(int(i));
   return out;
}

vector<int> Ecal_Analysis_Class::get_list_of_primary_mc(vector<int> &mc_part_sim_status) {
/// Get a list of indexes to MCParticles (mc_part_*) of the primary particles, i.e. from the generator.
/// This version is based on the MC status word.
   vector<int> out;
   for(size_t i=0; i< mc_part_sim_status.size(); ++i){
      if( uint(mc_part_sim_status[i]) & 0x01 ){
         out.push_back(int(i));
      }
   }
   return out;
}


vector<int> Ecal_Analysis_Class::get_list_of_all_secondary_mc(vector<double> &part_z) {
   /// Get a list of all secondary MCParticles (mc_part_*), i.e. not primary ones.
   /// Decision is based on the starting points of the tracks.
   vector<int> out;
   for(size_t i=0; i< part_z.size(); ++i)
      if(abs(part_z[i]) >= 1e-6) out.push_back( int(i));
   return out;
}

vector<int> Ecal_Analysis_Class::get_list_of_all_secondary_mc(vector<int> &mc_part_sim_status) {
   /// Get a list of all secondary MCParticles (mc_part_*), i.e. not primary ones.
   /// Decision is based on the status word of the MC particle.
   vector<int> out;
   for(size_t i=0; i< mc_part_sim_status.size(); ++i)
      if( (uint(mc_part_sim_status[i]) & 0x1) == 0 ) out.push_back( int(i));
   return out;
}


vector<int> Ecal_Analysis_Class::get_centers_of_scored_secondary_mc(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx, vector<double> &mc_score_x,
                                                 vector<double> &mc_score_y, vector<double> &mc_score_z, vector<double> &mc_score_pz) const{
   /// Returns a list of the indexes of the "seed" hits on the score plane, i.e. a list of the indexes of the highest
   /// momentum score hits caused by those secondary MCParticles (mc_part_*) that cross the ECal
   /// scoring plane in positive z direction, and are the highest pz within mc_score_close_cut distance in x and y.
   vector<int> out;
   vector<bool> already_used(mc_score_z.size(), false);
   bool not_done = true;
   while(not_done) {
      double max_val = 0;
      int idx_current_max = -1;
      for (size_t i = 0; i < mc_score_z.size(); ++i){
         int i_part = mc_score_part_idx[i];
         if( !already_used[i] && (uint(mc_part_sim_status[i_part]) & 0x1)==0 && mc_score_z[i] > mc_score_z_position_cut &&
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
         if (!already_used[ii] && (uint(mc_part_sim_status[i_part]) & 0x1)==0 && mc_score_z[ii] > mc_score_z_position_cut &&
             mc_score_pz[ii] > mc_score_pz_cut &&
             abs(mc_score_x[ii] - x_center) < mc_score_close_cut &&  abs(mc_score_y[ii] - y_center) < mc_score_close_cut){
            already_used[ii] = true;
         }
      }
      out.push_back(idx_current_max);
   }
   return out;
}


vector<double> Ecal_Analysis_Class::get_score_primary_hits_energy(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
                                                                vector<double> &mc_score_z, vector<double> &mc_score_px,
                                                                vector<double> &mc_score_py, vector<double> &mc_score_pz) {
   vector<double> out;
   for(size_t i_part=0; i_part < mc_part_sim_status.size(); ++i_part){
      if(uint(mc_part_sim_status[i_part]) & 0x1 ) {
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

vector<double> Ecal_Analysis_Class::get_score_secondary_hits_energy(vector<int> &mc_part_sim_status, vector<int> &mc_score_part_idx,
                                                                  vector<double> &mc_score_z, vector<double> &mc_score_px,
                                                                  vector<double> &mc_score_py, vector<double> &mc_score_pz) {
   vector<double> out;
   for(size_t i_part=0; i_part < mc_part_sim_status.size(); ++i_part){
      if((uint(mc_part_sim_status[i_part]) & 0x1)==0) {
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
   const vector<double> crystal_pos_x{-298.73819, -282.96774, -267.28214, -251.67653, -236.14636, -220.68710,
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

   auto const idx = lower_bound(crystal_pos_x.begin(), crystal_pos_x.end(), xpos);
   if (idx == crystal_pos_x.end()) { return -1.; }

   if (xpos > (crystal_pos_x[22] + crystal_pos_x[23]) / 2.) {
      offset = -22;
   } else {
      offset = -23;
   }

   int i_idx = distance(crystal_pos_x.begin(), idx);
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