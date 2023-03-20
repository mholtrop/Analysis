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
    return("Utility Functions V1.0.5 \n");
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

RVec<double> score_plane_hit_energy(RVec<int> type, RVec<double> px, RVec<double> py, RVec<double> pz,
                                                    RVec<double> x, RVec<double> y, RVec<double> z){
    // Find the summed energies of particles that are closer than 20mm crossing the ECal scoring plane.
    RVec<double> out;
    vector<int> already_used(type.size(), false);
    bool done = false;
    while( !done){
        double max_pz = 0;
        double e_sum=0;
        int i_max=0;
        // Find any next max pz
        for(size_t ii=0; ii<type.size(); ++ii){
            if( !already_used[ii] && z[ii]>= 1440 && pz[ii]>0.05){
                if(pz[ii] > max_pz){
                    i_max = ii;
                    max_pz = pz[ii];
                }
            }else{
                already_used[ii] = true; // don't want to see them again.
            }
        }
        already_used[i_max] = true;
        e_sum = sqrt(px[i_max]*px[i_max]+py[i_max]*py[i_max]+pz[i_max]*pz[i_max]);
        // Find crossing with pz>0.1
        for(size_t j=0; j<type.size(); ++j){
            if( !already_used[j] && z[j]>= 1440 && pz[j]>0.01 && abs(x[i_max] - x[j])<=20 && abs(y[i_max]- y[j]) <= 20){
                e_sum += sqrt(px[j]*px[j]+py[j]*py[j]+pz[j]*pz[j]);
                already_used[j] = true;
            }
        }
        out.push_back(e_sum);

        int count_used = 0;
        for(size_t i=0; i< type.size(); ++i){
            if(already_used[i]) count_used++;
        }
        if( count_used == (int)type.size()) done = true;
    }
    return out;
}

RVec<double> score_plane_hit_x(RVec<int> type, RVec<double> px, RVec<double> py, RVec<double> pz,
                                                    RVec<double> x, RVec<double> y, RVec<double> z){
    // Find the summed energies of particles that are closer than 20mm crossing the ECal scoring plane.
    RVec<double> out;
    vector<int> already_used(type.size(), false);
    bool done = false;
    while( !done){
        double max_pz = 0;
        double e_sum=0;
        double x_ave=0;
        int i_max=0;
        // Find any next max pz
        for(size_t ii=0; ii<type.size(); ++ii){
            if( !already_used[ii] && z[ii]>= 1440 && pz[ii]>0.05){
                if(pz[ii] > max_pz){
                    i_max = ii;
                    max_pz = pz[ii];
                }
            }else{
                already_used[ii] = true; // don't want to see them again.
            }
        }
        already_used[i_max] = true;
        e_sum = sqrt(px[i_max]*px[i_max]+py[i_max]*py[i_max]+pz[i_max]*pz[i_max]);
        x_ave = x[i_max]*e_sum;
        // Find crossing with pz>0.1
        for(size_t j=0; j<type.size(); ++j){
            if( !already_used[j] && z[j]>= 1440 && pz[j]>0.01 && abs(x[i_max] - x[j])<=20 && abs(y[i_max]- y[j]) <= 20){
                double en = sqrt(px[j]*px[j]+py[j]*py[j]+pz[j]*pz[j]);
                e_sum += en;
                x_ave += x[j]*en;
                already_used[j] = true;
            }
        }

        x_ave = x_ave/e_sum;
        out.push_back(x_ave);

        int count_used = 0;
        for(size_t i=0; i< type.size(); ++i){
            if(already_used[i]) count_used++;
        }
        if( count_used == (int)type.size()) done = true;
    }
    return out;
}

RVec<double> score_plane_hit_y(RVec<int> type, RVec<double> px, RVec<double> py, RVec<double> pz,
                                                    RVec<double> x, RVec<double> y, RVec<double> z){
    // Find the summed energies of particles that are closer than 20mm crossing the ECal scoring plane.
    RVec<double> out;
    vector<int> already_used(type.size(), false);
    bool done = false;
    while( !done){
        double max_pz = 0;
        double e_sum=0;
        double y_ave=0;
        int i_max=0;
        // Find any next max pz
        for(size_t ii=0; ii<type.size(); ++ii){
            if( !already_used[ii] && z[ii]>= 1440 && pz[ii]>0.05){
                if(pz[ii] > max_pz){
                    i_max = ii;
                    max_pz = pz[ii];
                }
            }else{
                already_used[ii] = true; // don't want to see them again.
            }
        }
        already_used[i_max] = true;
        e_sum = sqrt(px[i_max]*px[i_max]+py[i_max]*py[i_max]+pz[i_max]*pz[i_max]);
        y_ave = y[i_max]*e_sum;
        // Find crossing with pz>0.1
        for(size_t j=0; j<type.size(); ++j){
            if( !already_used[j] && z[j]>= 1440 && pz[j]>0.01 && abs(x[i_max] - x[j])<=20 && abs(y[i_max]- y[j]) <= 20){
                double en = sqrt(px[j]*px[j]+py[j]*py[j]+pz[j]*pz[j]);
                e_sum += en;
                y_ave += y[j]*en;
                already_used[j] = true;
            }
        }

        y_ave = y_ave/e_sum;
        out.push_back(y_ave);

        int count_used = 0;
        for(size_t i=0; i< type.size(); ++i){
            if(already_used[i]) count_used++;
        }
        if( count_used == (int)type.size()) done = true;
    }
    return out;
}



RVec<double> score_plane_hit_energy2(RVec<int> type, RVec<double> px, RVec<double> py, RVec<double> pz,
                                                    RVec<double> x, RVec<double> y, RVec<double> z){
    // Find the summed energies of particles that are closer than 20mm crossing the ECal scoring plane.
    RVec<double> out;
    vector<int> already_used(type.size(), false);
    for(size_t i=0; i<type.size(); ++i){
        double e_sum=0;
        // Find crossing with pz>0.1
        if( !already_used[i] && z[i]>= 1440 && pz[i]>0.05){
            e_sum = sqrt(px[i]*px[i]+py[i]*py[i]+pz[i]*pz[i]);
            already_used[i] = true;
            for(size_t j=0; j<type.size(); ++j){
                if(z[j]>= 1440 && pz[j]>0.01)
                    printf("[%2zu, %2zu]  pz: %7.3f  dx: %7.4f  dy: %7.4f \n" ,i, j, pz[j],abs(x[i] - x[j]), abs(y[i]- y[j]) );
                if( !already_used[j] && z[j]>= 1440 && pz[j]>0.01 && abs(x[i] - x[j])<=20 && abs(y[i]- y[j]) <= 20){
                    e_sum += sqrt(px[j]*px[j]+py[j]*py[j]+pz[j]*pz[j]);
                    cout << "added. Sum = " << e_sum << "\n";
                    already_used[j] = true;
                }
            }
            out.push_back(e_sum);
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
//
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

#include "TMath.h"
/// From:
///  https://root.cern.ch/doc/master/langaus_8C_source.html
/// -------
/// Convoluted Landau and Gaussian Fitting Function
///         (using ROOT's Landau and Gauss functions)
///
///  Based on a Fortran code by R.Fruehwirth (fruhwirth@hephy.oeaw.ac.at)
///
double langaufun(double *x, double *par) {

   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation),
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.

      // Numeric constants
      double invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      double mpshift  = -0.22278298;       // Landau maximum location

      // Control constants
      double np = 100.0;      // number of convolution steps
      double sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

      // Variables
      double xx;
      double mpc;
      double fland;
      double sum = 0.0;
      double xlow,xupp;
      double step;
      double i;


      // MP shift correction
      mpc = par[1] - mpshift * par[0];

      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];

      step = (xupp-xlow) / np;

      // Convolution integral of Landau and Gaussian by sum
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);

         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }

      return (par[2] * step * sum * invsq2pi / par[3]);
}

