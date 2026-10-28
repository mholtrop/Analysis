"""! @package ecal
Class with helpers to analyze the HPS ECal

- fiducial_cut_idx C++
- df_extend_ecal_cluster_fiducial
"""

import ROOT as R

class ecal:

    __version__ = "1.0.1"
    def __init__(self):
        """Initialize the ecal class"""
        # Define some useful C++ functions.
        # Test if the ix, iy are in the fiducial region.
        #
        # ToDo: Template the fiducial_cut function if calling with vector<int> is also needed.
        R.gInterpreter.Declare("""
           inline static bool is_in_fiducial_region(const int ix, const int iy){  
             return !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
             !(iy <= -5 || iy >= 5)   && /* Cut out the top and bottom row */
             !(iy >= -1 && iy <= 1)   && /* Cut out the first row around the gap */
             !(iy >= -2 && iy <= 2 && ix >= -11 && ix <= 1);
            }
            """)
        R.gInterpreter.Declare(""" 
            ROOT::VecOps::RVec<int> fiducial_cut_idx(const ROOT::VecOps::RVec<int> &ix, const ROOT::VecOps::RVec<int> &iy){
                ROOT::VecOps::RVec<int> out;
                for(size_t i=0;i< ix.size();++i){
                   if(is_in_fiducial_region(ix[i], iy[i])) out.push_back(i);
                   }
               return out;
            }
            ROOT::VecOps::RVec<bool> fiducial_cut_bool(const ROOT::VecOps::RVec<int> &ix, const ROOT::VecOps::RVec<int> &iy){
                ROOT::VecOps::RVec<bool> out;
                for(size_t i=0;i< ix.size();++i){
                   if(is_in_fiducial_region(ix[i], iy[i]) ){
                      out.push_back(true);
                   }else{
                      out.push_back(false);
                   }
                }
               return out;
            }
        """)


    def df_extend_ecal_cluster_fiducial(self, df_in):
        """Extend the RDataframe with a column that has the indexes of the Ecal cluster seeds
         that are in the fiducial region."""
        assert("RDataFrame" in str(type(df_in)) or "RInterface" in str(type(df_in)))
        assert(df_in.HasColumn("ecal_cluster_seed_ix") and df_in.HasColumn("ecal_cluster_seed_iy"))
        if(not df_in.HasColumn("ecal_cluster_fiducial_idx")):
            df_out = df_in.Define("ecal_cluster_fiducial_idx","return fiducial_cut_idx(ecal_cluster_seed_ix,ecal_cluster_seed_iy);")
        else:
            df_out = df_in
        return df_out


if __name__ == '__main__':
    e = ecal()
    # TODO: Write some testing code.