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

int    Debug = 0;

string Particles(void){
    return("Particles analysis. V1.0.1 \n");
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

RVec<bool> fiducial_cut(RVec<int> ix, RVec<int> iy){
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

RVec<bool> fiducial_cut_top(RVec<int> ix, RVec<int> iy){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if( ix[i]>=-22 && ix[i]<=22 &&
            iy[i]>=2 && iy[i]<=4
          )
        {
            out.push_back(true);
        }else{
            out.push_back(false);
        }
    }
    return out;
}

RVec<bool> fiducial_cut_bot(RVec<int> ix, RVec<int> iy){
    RVec<bool> out;
    for(size_t i=0;i< ix.size();++i){
        if( ix[i]>=-22 && ix[i]<=22 &&
            iy[i]>=-4 && iy[i]<=-2
          )
        {
            out.push_back(true);
        }else{
            out.push_back(false);
        }
    }
    return out;
}