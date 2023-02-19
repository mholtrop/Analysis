import ROOT as R

R.EnableImplicitMT()
import os
recompile = True
try:
    if os.path.getmtime('Particles_C.so') - os.path.getmtime('../Python/Particles.C') > 0:
        recompile = False
        print("Recompile is not needed")
    else:
        print("Recompiling: ")
except:
    print("Recompile needed, file not found.")
if recompile:
    R.gROOT.LoadMacro("Particles.C++")
else:
    R.gROOT.LoadMacro("Particles_C.so")
R.Particles()
# Note: runs 14624 through 14673 are with Ebeam = 1.92 GeV. The rest is at 3.742 GeV
Ebeam = 3.742
ch = R.TChain("MiniDST")
#ch.Add("/data/HPS/data/physrun2021/pass0/minidst/hps_014722_000*.root")
ch.Add("/data/HPS/data/physrun2021/pass0/minidst/hps_0147*.root")
#ch.Add("/data/HPS/data/physrun2021/fee/*.root")
print(f"Loaded {ch.GetEntries():,d} events ({ch.GetEntries():7.2e})")
df = R.RDataFrame(ch)

print("Available data names in Tuple:")
ColumnNames = df.GetColumnNames()
ll = 0
pr_colnames = [x for x in ColumnNames if str(x).startswith('ecal')]

R.gInterpreter.Declare("""
bool is_in_fiducial_region(int ix, int iy){
    return(
           !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
           !(iy <= -5  || iy >= 5)  && /* Cut out the top and bottom row */
           !(iy >= -1  && iy <= 1)  && /* Cut out the first row around the gap */
           !(iy >= -2  && iy <= 2   && ix >= -11 && ix <= -1) && /* Cut around the photon hole */
           !(iy >=  4  && ix >= -19 && ix <= -17) && /* Cut around crystal (-18, 5) */
          // !(iy >=  1  && iy <= 3   && ix <= -21 ) &&   /* Cut around crystal (-23, 2) */
           !(iy >= 2   && iy <= 4 && ix >= -14 && ix <= -12 ) && /* Cut around crystal ( -13, 3) */
           !(iy >=  4               && ix >= 2 && ix <= 4)  /* Cut around crystal (3, 5) */
    );
}
""")

R.gInterpreter.Declare("""
RVec<bool> fid_cut(RVec<int> ix, RVec<int> iy){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if(
           is_in_fiducial_region(ix[i], iy[i]) &&
           ( ix[i] < 0 )  /* Only electron side */
        ){
            out.push_back(true);
        }else{
            out.push_back(false);
        }
    }
    return out;
}
""")


dfx = df.Define("n_clus", "ecal_cluster_energy.size()")\
        .Define("cluster_is_fiducial", "fid_cut(ecal_cluster_seed_ix,ecal_cluster_seed_iy)")\
        .Define("ecs_ix_f", "RVec<int> out;for(size_t i=0; i<cluster_is_fiducial.size();++i){ if("
                            "cluster_is_fiducial[i]) out.push_back(ecal_cluster_seed_ix[i]);}; return out;")\
        .Define("ecs_iy_f", "RVec<int> out;for(size_t i=0; i<cluster_is_fiducial.size();++i){ if("
                            "cluster_is_fiducial[i]) out.push_back(ecal_cluster_seed_iy[i]);}; return out;")\
        .Define("ecal_tot_energy", "double etot=0; for(size_t i=0; i<ecal_hit_energy.size();++i){ etot+= "
                                   "ecal_hit_energy[i]; } return etot;")\
        .Define("ecal_tot_clus_e", "double etot=0; for(size_t i=0; i<ecal_cluster_energy.size();++i){ etot+= "
                                   "ecal_cluster_energy[i]; } return etot;")\
        .Define("ecal_tot_energy_fid", """
        double etot=0;
        for(size_t i=0; i<ecal_hit_energy.size();++i){
            if(is_in_fiducial_region(ecal_hit_index_x[i],ecal_hit_index_y[i])) etot+= ecal_hit_energy[i];
            }
        return etot;""")
# Calculate the ecal cluster energy for clusters with a fiducial cut (limiting the region in the ECal), and a cut on the
# number of hits in the cluster, plus a cut on the seed energy.

dfx = dfx.Define("ec_e_f_np", """
    RVec<RVec<double>> out;
    for(int n=0;n<15;n++){
        RVec<double> ntmp;
        for(size_t i=0; i<cluster_is_fiducial.size();++i){
            if( cluster_is_fiducial[i] &&
                ecal_cluster_nhits[i] == n &&
                ecal_cluster_seed_energy[i]>= 2. ){
                    ntmp.push_back(ecal_cluster_energy[i]);
            }
        }
       out.push_back(ntmp);
    }
    return out;""")

# Same, but without the cut on the number of hits.
dfx = dfx.Define("ec_e_f", """
        RVec<double> out;
        for(size_t i=0; i<cluster_is_fiducial.size();++i){
            if(cluster_is_fiducial[i]){out.push_back(ecal_cluster_energy[i]);}
        }
        return out;""")\
    .Define("ec_e_f_e", """
        RVec<double> out;
        for(size_t i=0; i<cluster_is_fiducial.size();++i){
            if(cluster_is_fiducial[i] && ecal_cluster_seed_energy[i]>= 2.){out.push_back(ecal_cluster_energy[i]);}
        }
        return out;""")

pr_colnames.append("n_clus")
pr_colnames.append("cluster_is_fiducial")
pr_colnames.append("ecs_ix_f")
pr_colnames.append("ecs_iy_f")
pr_colnames.append("ecal_tot_energy")
pr_colnames.append("ecal_tot_clus_e")
pr_colnames.append("ecal_tot_energy_fid")
pr_colnames.append("ec_e_f_np")
pr_colnames.append("ec_e_f")
pr_colnames.append("ec_e_f_e")

for nn in pr_colnames:
    if ll < len(nn):
        ll = len(nn)
for n in range(len(pr_colnames)):
    if n%4 == 0:
        print("")
    print(f"{str(pr_colnames[n]):{ll}s}", end="")

dfx = dfx.Filter("ec_e_f_e.size()>0")
dfx.Snapshot("MiniDST", "Filtered_147xxx.root", pr_colnames)

print("Done!")
