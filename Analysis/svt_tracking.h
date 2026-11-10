// svt_tracking.h
//
// C++ helpers for the SVT hit / tracking-efficiency study, used from RDataFrame.
// Loaded by Analysis/svt_tracking.py through ROOT's interpreter; not meant to be compiled alone.
//
// Conventions for the 2021 MiniDST (MiniDst/LcioReader v1.6):
//  - svt_hit_* holds the 1D strip clusters twice: type 0 = "StripClusterer_SiTrackerHitStrip1D",
//    type 1 = the second collection, which in these files is an identical copy. KF tracks
//    (track_svt_hits) point to type-1 hits in the MC and pass5_v10 data, but to type-0 hits in pass5_v9 data,
//    so the hit type is a parameter (`htype`), detected per sample in svt_tracking.py.
//  - svt_hit_layer runs 1..14 (one sensor per layer and per half; layers 9-14 have hole+slot sensors).
//  - svt_hit_module: even = top half, odd = bottom half.
//  - svt_hit_mc_part_idx[i] lists the MC-particle *indices* that deposited charge in hit i.
//    Hits from the overlaid pulser data have an empty list.
//  - track_hit_pattern (KF): per layer 1 = hit on track, -1 = sensor crossed without a hit
//    (a "hole"), 0 = no sensor crossed.
//
//  - Sensor orientation alternates and is mirrored between the halves (svt_daq_map of the 2021 conditions, and
//    the comment in KalmanParams.java): top L1 axial, L2 stereo, L3 axial, ...; bottom L1 stereo, L2 axial, ...
//    So the stereo layers are the even layers in the top half and the odd layers in the bottom half. With B along y,
//    the axial strips measure y (non-bend plane); the bend-plane (momentum) information comes from the stereo strips.
//  - 2021 dead sensors (conditions table svt_bad_channels, "Dead Channels for 2021 Run", runs 14000-15000, also
//    seen in run-14229 data): the whole L9 bottom electron-side stereo sensor (module 1, 640/640 strips); 128 strips
//    of L13 top electron-side axial; 54 strips of L14 top positron-side stereo. In layers 9-14, svt_hit_module
//    0/1 = electron side (top/bottom), 2/3 = positron side.
//
// Layers are encoded as bit masks: bit (L-1) is set when layer L is present.
//
#pragma once
#include <ROOT/RVec.hxx>
#include <vector>
#include <cmath>

namespace svt {

using RVecI  = ROOT::RVec<int>;
using RVecD  = ROOT::RVec<double>;
using RVecB  = ROOT::RVec<bool>;
using VecVI  = ROOT::RVec<std::vector<int>>;

constexpr int kTrackHitType = 1;   ///< default; functions take the actual type as an argument
constexpr int kNLayers      = 14;
constexpr int kStereoTop    = 0x2AAA;  ///< layers 2,4,...,14: stereo sensors of the top half
constexpr int kStereoBot    = 0x1555;  ///< layers 1,3,...,13: stereo sensors of the bottom half
constexpr int kFrontLayers  = 0x00FF;  ///< layers 1-8 (modules L1-L4)
constexpr int kBackLayers   = 0x3F00;  ///< layers 9-14 (modules L5-L7, hole and slot sensors)

inline int popcount(int m) { return __builtin_popcount(static_cast<unsigned>(m)); }
inline int first_layer(int m) { return m ? __builtin_ctz(static_cast<unsigned>(m)) + 1 : 0; }
inline int last_layer(int m)  { return m ? 32 - __builtin_clz(static_cast<unsigned>(m)) : 0; }

inline RVecI popcount(const RVecI& m)    { RVecI o(m.size()); for (size_t i = 0; i < m.size(); ++i) o[i] = popcount(m[i]);    return o; }
inline RVecI first_layer(const RVecI& m) { RVecI o(m.size()); for (size_t i = 0; i < m.size(); ++i) o[i] = first_layer(m[i]); return o; }
inline RVecI last_layer(const RVecI& m)  { RVecI o(m.size()); for (size_t i = 0; i < m.size(); ++i) o[i] = last_layer(m[i]);  return o; }

/// Element-wise a & ~b on layer masks (the RDataFrame expression parser does not accept '~').
inline RVecI and_not(const RVecI& a, const RVecI& b) {
   RVecI o(a.size()); for (size_t i = 0; i < a.size(); ++i) o[i] = a[i] & ~b[i]; return o;
}

/// Element-wise: the stereo layers of each layer mask, for a track/particle in the top (true) or bottom half.
inline RVecI stereo_layers(const RVecI& m, const RVecB& top) {
   RVecI o(m.size()); for (size_t i = 0; i < m.size(); ++i) o[i] = m[i] & (top[i] ? kStereoTop : kStereoBot); return o;
}

/// Element-wise: the axial layers of each layer mask (complement of stereo_layers).
inline RVecI axial_layers(const RVecI& m, const RVecB& top) {
   RVecI o(m.size()); for (size_t i = 0; i < m.size(); ++i) o[i] = m[i] & (top[i] ? kStereoBot : kStereoTop); return o;
}

/// Flatten a list of layer masks into the list of layer numbers they contain (for per-layer histograms).
inline RVecI mask_to_layers(const RVecI& masks) {
   RVecI out;
   for (int m : masks)
      for (int l = 1; l <= kNLayers; ++l)
         if (m & (1 << (l - 1))) out.push_back(l);
   return out;
}

/// out[i] = v[idx[i]], or `fallback` where idx[i] < 0.
template <typename T>
inline ROOT::RVec<T> take_or(const ROOT::RVec<T>& v, const RVecI& idx, T fallback) {
   ROOT::RVec<T> out(idx.size(), fallback);
   for (size_t i = 0; i < idx.size(); ++i) if (idx[i] >= 0) out[i] = v[idx[i]];
   return out;
}

/// out[i] = true if a[i] is one of the values in b.
inline RVecB is_in(const RVecI& a, const RVecI& b) {
   RVecB out(a.size(), false);
   for (size_t i = 0; i < a.size(); ++i) for (int x : b) if (a[i] == x) { out[i] = true; break; }
   return out;
}

/// Indices of the generated (primary) charged trident particles: daughters of the PDG 622/623 "mother" label.
inline RVecI primary_charged(const RVecI& pdg, const VecVI& parents, const RVecD& charge) {
   RVecI out;
   for (size_t i = 0; i < pdg.size(); ++i) {
      if (charge[i] == 0 || parents[i].empty()) continue;
      int p = parents[i][0];
      if (p >= 0 && p < (int)pdg.size() && (pdg[p] == 622 || pdg[p] == 623)) out.push_back((int)i);
   }
   return out;
}

/// True if a hit's MC list contains particle `ip`.
inline bool hit_has(const std::vector<int>& mc, int ip) {
   for (int j : mc) if (j == ip) return true;
   return false;
}

/// For each particle in `parts`: bit mask of layers where it produced at least one reconstructed (type-1) hit.
inline RVecI truth_layer_mask(const RVecI& parts, const RVecI& hit_type, const RVecI& hit_layer, const VecVI& hit_mc,
                              int htype = kTrackHitType) {
   RVecI out(parts.size(), 0);
   for (size_t h = 0; h < hit_type.size(); ++h) {
      if (hit_type[h] != htype || hit_mc[h].empty()) continue;
      for (size_t k = 0; k < parts.size(); ++k)
         if (hit_has(hit_mc[h], parts[k])) out[k] |= 1 << (hit_layer[h] - 1);
   }
   return out;
}

/// For each particle in `parts`: true ECal energy (GeV) deposited by its shower (ECal hits whose ultimate parent it is).
inline RVecD truth_ecal_energy(const RVecI& parts, const RVecI& ecal_parent, const RVecD& ecal_e) {
   RVecD out(parts.size(), 0.);
   for (size_t h = 0; h < ecal_parent.size(); ++h)
      for (size_t k = 0; k < parts.size(); ++k)
         if (ecal_parent[h] == parts[k]) out[k] += ecal_e[h];
   return out;
}

/// Truth content of each reconstructed track.
struct TrackTruth {
   RVecI particle;   ///< MC index contributing to most hits on the track (-1 if none: all hits from background).
   RVecI n_shared;   ///< number of hits on the track that contain `particle`.
   RVecI hit_mask;   ///< layers with a hit on the track.
   RVecI good_mask;  ///< layers whose hit on the track contains `particle`.
};

inline TrackTruth track_truth(const VecVI& trk_hits, const RVecI& hit_layer, const VecVI& hit_mc) {
   TrackTruth t;
   const size_t n = trk_hits.size();
   t.particle.resize(n, -1); t.n_shared.resize(n, 0); t.hit_mask.resize(n, 0); t.good_mask.resize(n, 0);
   std::vector<std::pair<int, int>> counts;   // (mc index, n hits); a track has <= 14 hits, so a flat list is fastest.
   for (size_t i = 0; i < n; ++i) {
      counts.clear();
      for (int h : trk_hits[i]) {
         if (h < 0) continue;
         t.hit_mask[i] |= 1 << (hit_layer[h] - 1);
         const auto& mc = hit_mc[h];
         for (size_t a = 0; a < mc.size(); ++a) {
            bool dup = false;                  // count each particle only once per hit
            for (size_t b = 0; b < a; ++b) if (mc[b] == mc[a]) { dup = true; break; }
            if (dup) continue;
            bool found = false;
            for (auto& c : counts) if (c.first == mc[a]) { ++c.second; found = true; break; }
            if (!found) counts.emplace_back(mc[a], 1);
         }
      }
      for (auto& c : counts)
         if (c.second > t.n_shared[i]) { t.n_shared[i] = c.second; t.particle[i] = c.first; }
      if (t.particle[i] >= 0)
         for (int h : trk_hits[i])
            if (h >= 0 && hit_has(hit_mc[h], t.particle[i])) t.good_mask[i] |= 1 << (hit_layer[h] - 1);
   }
   return t;
}

/// For each particle in `parts`: index of the best reconstructed track whose majority particle it is and whose
/// purity (n_shared / n_hits) is at least `min_purity`; -1 if no such track. Best = most shared hits.
inline RVecI match_track(const RVecI& parts, const RVecI& trk_particle, const RVecI& trk_shared,
                         const RVecI& trk_nhits, double min_purity) {
   RVecI out(parts.size(), -1);
   for (size_t k = 0; k < parts.size(); ++k) {
      int best = -1;
      for (size_t i = 0; i < trk_particle.size(); ++i) {
         if (trk_particle[i] != parts[k] || trk_shared[i] < min_purity * trk_nhits[i]) continue;
         if (best < 0 || trk_shared[i] > trk_shared[best]) best = (int)i;
      }
      out[k] = best;
   }
   return out;
}

/// Number of tracks matched (as in match_track) to each particle; > 1 means duplicate tracks.
inline RVecI n_matched_tracks(const RVecI& parts, const RVecI& trk_particle, const RVecI& trk_shared,
                              const RVecI& trk_nhits, double min_purity) {
   RVecI out(parts.size(), 0);
   for (size_t k = 0; k < parts.size(); ++k)
      for (size_t i = 0; i < trk_particle.size(); ++i)
         if (trk_particle[i] == parts[k] && trk_shared[i] >= min_purity * trk_nhits[i]) ++out[k];
   return out;
}

/// Number of entries equal to `value` in each per-track hit pattern (e.g. value = -1 counts holes).
inline RVecI count_in_pattern(const VecVI& pattern, int value) {
   RVecI out(pattern.size(), 0);
   for (size_t i = 0; i < pattern.size(); ++i)
      for (int v : pattern[i]) if (v == value) ++out[i];
   return out;
}

/// Per type-1 hit: 1 if it contains any MC particle (signal), 0 if it came from the overlaid data (background).
/// Returned only for type-1 hits, in their original order.
inline RVecI hit_is_mc(const RVecI& hit_type, const VecVI& hit_mc, int htype = kTrackHitType) {
   RVecI out;
   for (size_t h = 0; h < hit_type.size(); ++h)
      if (hit_type[h] == htype) out.push_back(hit_mc[h].empty() ? 0 : 1);
   return out;
}

/// Per type-1 hit: 1 if it is used by at least one track.
inline RVecI hit_on_track(const RVecI& hit_type, const VecVI& trk_hits, int htype = kTrackHitType) {
   RVecI used(hit_type.size(), 0);
   for (auto& v : trk_hits) for (int h : v) if (h >= 0) used[h] = 1;
   RVecI out;
   for (size_t h = 0; h < hit_type.size(); ++h)
      if (hit_type[h] == htype) out.push_back(used[h]);
   return out;
}

/// Time of every hit on a track minus the track time, split by whether the hit belongs to the track's truth
/// particle (good = true) or not (good = false). Only tracks with a truth particle are used.
inline RVecD hit_minus_track_time(const VecVI& trk_hits, const RVecD& trk_time, const RVecI& trk_particle,
                                  const RVecD& hit_time, const VecVI& hit_mc, bool good) {
   RVecD out;
   for (size_t i = 0; i < trk_hits.size(); ++i) {
      if (trk_particle[i] < 0) continue;
      for (int h : trk_hits[i])
         if (h >= 0 && hit_has(hit_mc[h], trk_particle[i]) == good) out.push_back(hit_time[h] - trk_time[i]);
   }
   return out;
}

/// Per track: index of an ECal cluster attached to it by the reconstructed particles (part_track, part_ecal_cluster),
/// or -1 if no reconstructed particle with this track has a cluster.
inline RVecI track_cluster(int n_tracks, const RVecI& part_track, const RVecI& part_cluster) {
   RVecI out(n_tracks, -1);
   for (size_t i = 0; i < part_track.size(); ++i) {
      const int t = part_track[i];
      if (t >= 0 && t < n_tracks && out[t] < 0 && part_cluster[i] >= 0) out[t] = part_cluster[i];
   }
   return out;
}

/// For each particle: true if `clus[k]` is a cluster whose truth (ecal_cluster_mc_id, the index of the MC particle
/// that started the shower) is the particle itself.
inline RVecB cluster_is_true(const RVecI& parts, const RVecI& clus, const RVecI& clus_mc) {
   RVecB out(parts.size(), false);
   for (size_t k = 0; k < parts.size(); ++k)
      out[k] = clus[k] >= 0 && clus[k] < (int)clus_mc.size() && clus_mc[clus[k]] == parts[k];
   return out;
}

}  // namespace svt
