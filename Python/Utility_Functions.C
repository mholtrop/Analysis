//
// Created by Maurik Holtrop on 9/29/21.
//
#include <tuple>
#include "TStyle.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"
#include "ROOT/RDF/RInterface.hxx"
#include "Math/LorentzVector.h"
#include "Math/Vector4D.h"
#include <locale.h>
#include "TROOT.h"
#include "TChain.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/RDF/InterfaceUtils.hxx"
#include "TMath.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TRatioPlot.h"
#include "TVector3.h"
#include "TLorentzVector.h"

using namespace std;
using namespace ROOT;
using namespace ROOT::RDF;
using namespace ROOT::VecOps;

//
// This code is needed to generate dictionaries for the RVec<vector<type>> vector of vectors.
// These RVec<> are needed because they interface with Python and Numpy, where just plain vector<> does not.
// The vector of vectors are not pre-defined, where the vectors of primitives are.
//
#include <vector>
#ifdef __CLING__
#pragma link C++ nestedtypedefs;
#pragma link C++ class ROOT::VecOps::RVec<vector<int>>+;
#pragma link C++ class ROOT::VecOps::RVec<vector<int>>::*+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>>::*+;
#pragma link C++ class ROOT::VecOps::RVec<vector<double>>+;
#pragma link C++ class ROOT::VecOps::RVec<vector<double>>::*+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<double>>::*+;
#pragma link C++ class ROOT::VecOps::RVec<vector<bool>>+;
#pragma link C++ class ROOT::VecOps::RVec<vector<bool>>::*+;
#endif

int    Debug = 0;

string Utility_Functions(void){
    return("Utility Functions V1.0.4 \n");
}

bool is_in_fiducial_region(int ix, int iy){
    return(
          !(ix <= -23 || ix >= 23) && /* Cut out the left and right side */
               !(iy <= -5  || iy >= 5)  && /* Cut out the top and bottom row */
               !(iy >= -1  && iy <= 1)  && /* Cut out the first row around the gap */
               !(iy >= -2  && iy <= 2   && ix >= -11 && ix <= -1)  /* Cut around the photon hole */
        );
}

bool is_in_fiducial_region_extended(int ix, int iy, vector< pair<int,int> > exclude){
    if( !is_in_fiducial_region(ix,iy) ){
        return false;
    }
    for(const auto& cc : exclude) {
        if( ix >= cc.first-1 && ix <= cc.first+1 && iy >= cc.second-1 && iy <= cc.second +1){
        // ix,iy is within the 3x3 exclusion zone for this crystal.
            return false;
        }
        // This is where the decision to skip ix==0 bites. If first==1 then first-2 should be excluded too,
        // same for first==-1, then first+2 should be excluded.
        if( cc.first == 1 && ix == -1 && iy >= cc.second-1 && iy <= cc.second +1) return false;
        if( cc.first == -1 && ix == 1 && iy >= cc.second-1 && iy <= cc.second +1) return false;
    }
    return true;

}

RVec<int> find_index_primary_mc_part(RVec<double> mc_part_z){
    RVec<int> out;
    for(size_t i=0;i< mc_part_z.size(); ++i){
        if( mc_part_z[i]<= 0.001) out.push_back(i);
    }
    return out;
}

RVec<double> find_end_z_of_primary_mc_part(RVec<double> mc_part_z, RVec<double> mc_part_end_z){
    RVec<int> idxs = find_index_primary_mc_part(mc_part_z);
    RVec<double> out;
    for(int i: idxs){
        out.push_back(mc_part_end_z[i]);
    }
    return out;
}

RVec<double> find_average_end_of_daughters_of_primary_mc_part(RVec<double> mc_part_z, RVec<double> mc_part_end_z, RVec< std::vector<int> > mc_part_daughters){
    /// Find the average z endpoint of all the daughters of the primary particles. If the primary has no daughters then return its endpoint.
    RVec<int> idxs = find_index_primary_mc_part(mc_part_z);
    RVec<double> out;
    for(int i: idxs){
        if( mc_part_daughters[i].size() > 0){
            double ave = 0;
            for(int ii: mc_part_daughters[i]){
                ave += mc_part_end_z[ii];
            }
            ave = ave/mc_part_daughters[i].size();
            out.push_back(ave);
        }else{
            out.push_back(mc_part_end_z[i]);
        }
    }
    return out;
}

RVec<bool> fiducial_cut(RVec<int> ix, RVec<int> iy){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if( is_in_fiducial_region(ix[i], iy[i])){
            out.push_back(true);
         }else{
            out.push_back(false);
         }
    }
    return out;
}

RVec<bool> fiducial_cut_extended(RVec<int> ix, RVec<int> iy, vector< pair<int,int> > exclude){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if(is_in_fiducial_region_extended(ix[i], iy[i], exclude)){
            out.push_back(true);
         }else{
            out.push_back(false);
         }
    }
    return out;
}

int count_true(RVec<bool> inarr){
    int cc =0;
    for(bool v: inarr){
        if(v) cc++;
    }
    return cc;
}

RVec<double> track_kf_x(RVec<double> x, RVec<int> track_type){
    RVec<double> out;
    for(int i=0; i< (int)x.size(); ++i){
        if( track_type[i] == 1)
            out.push_back(x[i]);
    }
    return out;
}

RVec<double> track_gbl_x(RVec<double> x, RVec<int> track_type){
    RVec<double> out;
    for(int i=0; i< (int)x.size(); ++i){
        if( track_type[i] > 31)
            out.push_back(x[i]);
    }
    return out;
}

RVec<double> part_type_x(RVec<double> x, RVec<int> part_type, int type_val){
    RVec<double> out;
    for(int i=0; i< (int)x.size(); ++i){
        if( part_type[i] == type_val)
            out.push_back(x[i]);
    }
    return out;
}

RVec<double> track_mom(RVec<double> omega, RVec<double> tanlam){
    RVec<double> out;
    for(int i=0; i< (int)omega.size(); ++i){
        double mom = 2.99792458E-4*(0.8416/omega[i])*sqrt(1 + tanlam[i]*tanlam[i]);  // 0.537
        out.push_back(mom);
    }
    return out;
}

RVec<double> track_p_signed(RVec<double> px, RVec<double> py, RVec<double> pz, RVec<double> omega){
    RVec<double> out;
    for(int i=0; i< (int)px.size(); ++i){
        double p = TMath::Sign(1.,omega[i])*sqrt(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i]);
        out.push_back(p);
    }
    return out;
}

double sum_vector(RVec<double> inarr){
    double out=0;
    for(double val: inarr){
        out += val;
    }
    return out;
}

double get_max(RVec<double> inarr){
    double out=0;
    for(double val: inarr){
        if(val>out){
            out = val;
        }
    }
    return out;
}

int get_max_index(RVec<double> inarr){
    double out=0;
    int i_out = 0;
    for(int i=0; i< (int)inarr.size(); ++i){
        if(inarr[i]>out){
            out = inarr[i];
            i_out = i;
        }
    }
    return i_out;
}

RVec<double> get_part_tht(RVec<double> px, RVec<double> pz){
    RVec<double> out;
    for(int i=0; i<(int)px.size(); ++i){
        double tht = TMath::ATan2(px[i],pz[i]);
        out.push_back(tht);
    }
    return out;
}

RVec<double> get_vec_abs(RVec<double> px, RVec<double> py, RVec<double> pz){
    RVec<double> out;
    for(int i=0; i<(int)px.size(); ++i){
        double p = TMath::Sqrt(px[i]*px[i] + py[i]*py[i] + pz[i] *pz[i]);
        out.push_back(p);
    }
    return out;
}


RVec<double> get_vec_theta(RVec<double> px, RVec<double> py, RVec<double> pz){
    RVec<double> out;
    for(int i=0; i<(int)px.size(); ++i){
        TVector3 mom(px[i], py[i], pz[i]);
        out.push_back(mom.Theta());
    }
    return out;
}

RVec<double> get_vec_phi(RVec<double> px, RVec<double> py, RVec<double> pz){
    RVec<double> out;
    for(int i=0; i<(int)px.size(); ++i){
        TVector3 mom(px[i], py[i], pz[i]);
        out.push_back(mom.Phi());
    }
    return out;
}

RVec<bool> fiducial_cut_X(RVec<int> ix, RVec<int> iy){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if( ix[i]>=-22 && ix[i]<=22 &&
          ( ( iy[i]>=2 && iy[i]<=4  ) ||
            (iy[i]>=-4 && iy[i]<=-2 )))
        {
            out.push_back(true);
        }else{
            out.push_back(false);
        }
    }
    return out;
}

Double_t fit_gaus_tail(Double_t *x, Double_t *par){
// A function for fitting a Gaussian with an exponential tail on the high side.
    Double_t mu = par[1];
    Double_t sigma = par[2];
    Double_t lamb = par[3];
    Double_t arg = 0;

    if(sigma< 0) return 0;
    Double_t ex = lamb/2*(2*(mu - x[0]) + lamb*sigma*sigma);
    Double_t er = (mu - x[0] + lamb*sigma*sigma)/(sqrt(2)*sigma);
    Double_t f = par[0]*lamb/2*exp(ex)*erfc(er);
    return f;
}

Double_t fit_gaus_tailn(Double_t *x, Double_t *par){
// A function for fitting a Gaussian with an exponential tail on the low side.
    Double_t mu = par[1];
    Double_t sigma = par[2];
    Double_t lamb = par[3];
    Double_t arg = 0;

    if(sigma< 0) return 0;
    Double_t ex = lamb/2*(2*(x[0] - mu) + lamb*sigma*sigma);
    Double_t er = (x[0]-mu + lamb*sigma*sigma)/(sqrt(2)*sigma);
    Double_t f = par[0]*lamb/2*exp(ex)*erfc(er);
    return f;
}

//TF1 *fit_function1 = new TF1("fit_function1",&fit_gaus_tailn,1.,5.,4);
//TF1 *fit_function2 = new TF1("fit_function2",&fit_gaus_tailn,1.,5.,4);
