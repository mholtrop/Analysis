// svt_hitfilter.h
//
// Per-hit features for the SVT hit filter (namespace svt::hf), used from RDataFrame by Analysis/svt_hitfilter.py.
//
// THE JAVA DRIVER (SVT/Hit_Filtering/hps-java/.../SvtHitFilterFeatures.java) IMPLEMENTS EXACTLY THE SAME
// DEFINITIONS. Any change here must be made there too (and the parity test re-run).
//
// Inputs are the 1D strip clusters ("StripClusterer_SiTrackerHitStrip1D") of one event, as stored in the MiniDST:
//   layer 1..14, module (even = top, odd = bottom; in L9-L14 0/1 = electron side, 2/3 = positron side), time [ns],
//   global position x, y, z [mm], edep (TrackerHit::getEDep / getdEdx), number of strips in the cluster.
// Stereo layers: even layers in the top half, odd layers in the bottom half.
//
// Features (in this order, see FEATURES in svt_hitfilter.py):
//   0 layer        1 top          2 stereo        3 side (1 = positron-side sensor in L9-L14, else 0)
//   4 t            hit time - t0 [ns]
//   5 q            edep * 1e6
//   6 nstrip       strips in the cluster
//   7 absy         |y| [mm]
//   8 tl           y / (z - zT), sign flipped in the bottom half (> 0 points away from the beam plane)
//   9 nsame        other hits on the same sensor (layer, module) within |dt| < 10 ns
//  10 pair_dt      partner sensor (other sensor of the same module, same half and side):
//                  min |dt| over partner hits with |dy| < 4 mm (100 if none)
//  11 pair_dy      min |dy| over partner hits with |dt| < 8 ns (50 if none)
//  12 has_anchor   1 if the hit has an axial "anchor": itself if axial, else the partner axial hit with
//                  |dy| < 4 mm and the smallest |dt| (must be < 8 ns)
//  13 nroad_tight  number of other modules with an axial hit (same half) within |dt| < 8 ns of the anchor and
//                  |tl - tl_anchor| < 0.002
//  14 nroad_loose  same with |tl - tl_anchor| < 0.006
//  15..21 dtl_m1..dtl_m7  per module (1..7): min |tl - tl_anchor| over axial hits of that module (same half,
//                  |dt| < 8 ns of the anchor); 1 if none; 0 for the anchor's own module; 1 if no anchor
//  22 noot_half    out-of-time hits (|t| > 25 ns) in the same half: a background-occupancy measure that does
//                  not depend on the signal (an all-hit count would leak the signal into PU training)
//  23 road_adt     |t - t_road|: t_road = mean time of the road hits of the anchor, i.e. for each other module the
//                  axial hit with the smallest |dtl| (if < 0.006), plus the anchor itself when the hit is stereo.
//                  Modules are summed in order 1..7, the anchor last. 100 if there are no road hits.
//  24 road_at      |t_road| (100 if no road hits)
//  25 nroad_t      number of hits in t_road
//  26 ncomp        other hits on the same sensor within |dy| < 2 mm and |dt| < 10 ns ("competitors")
//
#pragma once
#include <ROOT/RVec.hxx>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

namespace svt {
namespace hf {

constexpr int kNFeat = 27;
constexpr double kSameDt = 10.0, kPairDy = 4.0, kPairDt = 8.0, kRoadDt = 8.0, kTight = 0.002, kLoose = 0.006;
constexpr double kOotT = 25.0;
constexpr double kNoPairDt = 100.0, kNoPairDy = 50.0, kNoDtl = 1.0, kNoRoadT = 100.0, kCompDy = 2.0;

inline bool is_stereo(int layer, bool top) { return top ? (layer % 2 == 0) : (layer % 2 == 1); }

/// Features of the hits with use[i] true, flattened row-major (n_used x kNFeat), in original hit order.
inline ROOT::RVec<float> features(const ROOT::RVec<bool>& use, const ROOT::RVec<int>& layer, const ROOT::RVec<int>& module,
                                  const ROOT::RVec<double>& time, const ROOT::RVec<double>& y, const ROOT::RVec<double>& z,
                                  const ROOT::RVec<double>& edep, const ROOT::RVec<std::vector<int>>& strips,
                                  double t0, double zT) {
   std::vector<int> idx;
   for (size_t i = 0; i < use.size(); ++i) if (use[i]) idx.push_back((int)i);
   const int n = (int)idx.size();
   std::vector<int> L(n), M(n), half(n), side(n), mod(n);
   std::vector<char> st(n);
   std::vector<double> t(n), yy(n), tl(n);
   int ntop = 0, nbot = 0;
   for (int a = 0; a < n; ++a) {
      const int i = idx[a];
      L[a] = layer[i]; M[a] = module[i]; half[a] = (M[a] % 2 == 0) ? 1 : 0; side[a] = (M[a] >= 2) ? 1 : 0;
      mod[a] = (L[a] - 1) / 2;            // 0..6
      st[a] = is_stereo(L[a], half[a]);
      t[a] = time[i] - t0; yy[a] = y[i];
      tl[a] = y[i] / (z[i] - zT) * (half[a] ? 1.0 : -1.0);
   }
   for (int a = 0; a < n; ++a) if (std::abs(t[a]) > kOotT) { if (half[a]) ++ntop; else ++nbot; }
   ROOT::RVec<float> out(n * kNFeat);
   // anchor for every hit (index into 0..n-1, or -1)
   std::vector<int> anchor(n, -1);
   std::vector<double> pdt(n, kNoPairDt), pdy(n, kNoPairDy);
   std::vector<int> nsame(n, 0), ncomp(n, 0);
   for (int a = 0; a < n; ++a) {
      double best_dt = 1e9; int best = -1;
      for (int b = 0; b < n; ++b) {
         if (b == a || half[b] != half[a]) continue;
         const double dt = std::abs(t[b] - t[a]);
         if (L[b] == L[a] && M[b] == M[a]) {
            if (dt < kSameDt) { ++nsame[a]; if (std::abs(yy[b] - yy[a]) < kCompDy) ++ncomp[a]; }
            continue;
         }
         if (mod[b] != mod[a] || side[b] != side[a]) continue;    // partner sensor: same module pair, other layer
         const double dy = std::abs(yy[b] - yy[a]);
         if (dy < kPairDy) { pdt[a] = std::min(pdt[a], dt); if (st[a] && !st[b] && dt < kPairDt && dt < best_dt) { best_dt = dt; best = b; } }
         if (dt < kPairDt) pdy[a] = std::min(pdy[a], dy);
      }
      anchor[a] = st[a] ? best : a;
   }
   // road features per axial hit, computed once
   std::vector<std::array<double, 7>> dtl(n);
   std::vector<std::array<int, 7>> best(n);
   for (int a = 0; a < n; ++a) {
      dtl[a].fill(kNoDtl); best[a].fill(-1);
      if (st[a]) continue;
      dtl[a][mod[a]] = 0.0;
      for (int b = 0; b < n; ++b) {
         if (b == a || st[b] || half[b] != half[a] || mod[b] == mod[a]) continue;
         if (std::abs(t[b] - t[a]) >= kRoadDt) continue;
         const double d = std::abs(tl[b] - tl[a]);
         if (d < dtl[a][mod[b]]) { dtl[a][mod[b]] = d; best[a][mod[b]] = b; }
      }
   }
   for (int a = 0; a < n; ++a) {
      const int i = idx[a];
      float* f = &out[a * kNFeat];
      f[0] = L[a]; f[1] = half[a]; f[2] = st[a]; f[3] = side[a];
      f[4] = t[a]; f[5] = edep[i] * 1e6; f[6] = strips[i].size(); f[7] = std::abs(yy[a]); f[8] = tl[a];
      f[9] = nsame[a]; f[10] = pdt[a]; f[11] = pdy[a];
      const int an = anchor[a];
      f[12] = an >= 0;
      int nt = 0, nl = 0;
      for (int m = 0; m < 7; ++m) {
         const double d = an >= 0 ? dtl[an][m] : kNoDtl;
         f[15 + m] = d;
         if (an >= 0 && m != mod[an]) { if (d < kTight) ++nt; if (d < kLoose) ++nl; }
      }
      f[13] = nt; f[14] = nl;
      f[22] = half[a] ? ntop : nbot;
      double sum = 0.0; int nr = 0;
      if (an >= 0) {
         for (int m = 0; m < 7; ++m)
            if (m != mod[an] && best[an][m] >= 0 && dtl[an][m] < kLoose) { sum += t[best[an][m]]; ++nr; }
         if (an != a) { sum += t[an]; ++nr; }
      }
      if (nr > 0) { const double tr = sum / nr; f[23] = std::abs(t[a] - tr); f[24] = std::abs(tr); }
      else { f[23] = kNoRoadT; f[24] = kNoRoadT; }
      f[25] = nr; f[26] = ncomp[a];
   }
   return out;
}

/// Per used hit: 1 if it carries MC truth (any MC particle), else 0.
inline ROOT::RVec<int> is_mc(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& mc) {
   ROOT::RVec<int> o; for (size_t i = 0; i < use.size(); ++i) if (use[i]) o.push_back(mc[i].empty() ? 0 : 1); return o;
}

/// Per used hit: the first primary particle (one of `parts`) in its MC list, else -1.
inline ROOT::RVec<int> primary_of(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& mc, const ROOT::RVec<int>& parts) {
   ROOT::RVec<int> o;
   for (size_t i = 0; i < use.size(); ++i) {
      if (!use[i]) continue;
      int p = -1;
      for (int j : mc[i]) { for (int q : parts) if (q == j) { p = j; break; } if (p >= 0) break; }
      o.push_back(p);
   }
   return o;
}

/// Per used hit: 0 not on a track, 1 on a track that is not truth-matched, 2 correct hit on a matched track,
/// 3 wrong hit on a matched track (does not contain the track's particle). For data pass matched = all false and
/// an empty mc list: then the codes are 0 / 1 (and `good` can flag golden tracks: code 2 = on a good track).
inline ROOT::RVec<int> track_code(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& trk_hits,
                                  const ROOT::RVec<int>& trk_particle, const ROOT::RVec<bool>& matched,
                                  const ROOT::RVec<std::vector<int>>& mc) {
   ROOT::RVec<int> code(use.size(), 0);
   for (size_t k = 0; k < trk_hits.size(); ++k)
      for (int h : trk_hits[k]) {
         if (h < 0 || h >= (int)use.size()) continue;
         int c = 1;
         if (matched[k]) {
            bool has = false;
            if (!mc.empty()) for (int j : mc[h]) if (j == trk_particle[k]) { has = true; break; }
            c = has ? 2 : 3;
         }
         code[h] = std::max(code[h], c);
      }
   ROOT::RVec<int> o; for (size_t i = 0; i < use.size(); ++i) if (use[i]) o.push_back(code[i]); return o;
}

/// Data: per used hit, 1 if it is on a track flagged good (e.g. golden tag tracks), else 0.
inline ROOT::RVec<int> on_good_track(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& trk_hits,
                                     const ROOT::RVec<bool>& good) {
   ROOT::RVec<int> on(use.size(), 0);
   for (size_t k = 0; k < trk_hits.size(); ++k) if (good[k]) for (int h : trk_hits[k]) if (h >= 0 && h < (int)use.size()) on[h] = 1;
   ROOT::RVec<int> o; for (size_t i = 0; i < use.size(); ++i) if (use[i]) o.push_back(on[i]); return o;
}

/// Data: per used hit, index of the good track it is on (first one), else -1 (for track-level findability).
inline ROOT::RVec<int> good_track_index(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& trk_hits,
                                        const ROOT::RVec<bool>& good) {
   ROOT::RVec<int> on(use.size(), -1);
   for (size_t k = 0; k < trk_hits.size(); ++k) if (good[k]) for (int h : trk_hits[k]) if (h >= 0 && h < (int)use.size() && on[h] < 0) on[h] = (int)k;
   ROOT::RVec<int> o; for (size_t i = 0; i < use.size(); ++i) if (use[i]) o.push_back(on[i]); return o;
}

}  // namespace hf
}  // namespace svt

namespace svt {
namespace hf {

constexpr int kNTrkCols = 7;
/// Stage 2 (hits on tracks): one row per (track, hit) pair, flattened (n_pairs x kNTrkCols):
///   0 index of the hit among the used hits (to join with the per-hit features), 1 track index,
///   2 hit time - track time [ns], 3 track n hits, 4 track chi2/ndf, 5 number of tracks sharing the hit,
///   6 label: 1 unmatched track, 2 correct hit on a matched track, 3 wrong hit on a matched track.
inline ROOT::RVec<float> track_hit_rows(const ROOT::RVec<bool>& use, const ROOT::RVec<std::vector<int>>& trk_hits,
                                        const ROOT::RVec<double>& trk_time, const ROOT::RVec<int>& trk_nhits,
                                        const ROOT::RVec<double>& trk_chi2, const ROOT::RVec<int>& trk_particle,
                                        const ROOT::RVec<bool>& matched, const ROOT::RVec<double>& hit_time,
                                        const ROOT::RVec<std::vector<int>>& mc) {
   std::vector<int> used_idx(use.size(), -1), nshare(use.size(), 0);
   int k = 0;
   for (size_t i = 0; i < use.size(); ++i) if (use[i]) used_idx[i] = k++;
   for (auto& v : trk_hits) for (int h : v) if (h >= 0 && h < (int)use.size()) ++nshare[h];
   ROOT::RVec<float> out;
   for (size_t t = 0; t < trk_hits.size(); ++t)
      for (int h : trk_hits[t]) {
         if (h < 0 || h >= (int)use.size() || used_idx[h] < 0) continue;
         int lab = 1;
         if (matched[t]) {
            bool has = false;
            if (!mc.empty()) for (int j : mc[h]) if (j == trk_particle[t]) { has = true; break; }
            lab = has ? 2 : 3;
         }
         const float row[kNTrkCols] = {(float)used_idx[h], (float)t, (float)(hit_time[h] - trk_time[t]), (float)trk_nhits[t],
                                       (float)(trk_chi2[t] / std::max(1, trk_nhits[t] - 5)), (float)nshare[h], (float)lab};
         out.insert(out.end(), row, row + kNTrkCols);
      }
   return out;
}

}  // namespace hf
}  // namespace svt

namespace svt {
namespace hf {
/// Number of strips of each cluster.
inline ROOT::RVec<int> n_strips(const ROOT::RVec<std::vector<int>>& s) {
   ROOT::RVec<int> o(s.size()); for (size_t i = 0; i < s.size(); ++i) o[i] = (int)s[i].size(); return o;
}
}  // namespace hf
}  // namespace svt
