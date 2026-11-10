"""! @package svt_tracking
Tools to relate SVT hits to Kalman-filter tracking efficiency in HPS MiniDST files (MC with truth).

Typical use::

    from Analysis import svt_tracking as st
    st.setup(n_threads=32)
    df = st.load("overlay")                      # RDataFrame node with all derived columns
    eff = st.book_ratio(df, "tp_p", "tp_found", bins=(20, 0, 3))
    hist = st.book_hist(df, "n_svt_hits", bins=(60, 0, 600))
    st.run(eff, hist)                            # one event loop for everything booked
    st.plot_ratio(ax, eff, label="overlay")

Derived columns made by load() (prefix tells the granularity):

  Event:      n_svt_hits, n_svt_hits_mc, n_svt_hits_bkg, n_tracks
  tp_*        one entry per primary (generated) charged trident particle
              tp_idx, tp_pdg, tp_p, tp_tanl, tp_top, tp_layers (mask), tp_n_layers, tp_last_layer,
              tp_n_svt_hits (event occupancy), tp_ecal_e, tp_track, tp_found, tp_n_tracks, tp_trk_nhits, tp_missing (mask),
              tp_pos (positron), tp_trk_dp_rel (of the matched track), tp_gap (|tanl| < 0.015, beam gap), tp_acc (ECal acceptance: true shower energy > 0.1 GeV),
              efficiency tiers tp_clean, tp_good_p, tp_in_time, tp_ecal_match, and tp_tier = number of TIERS passed
              tp_stereo (mask of truth layers that are stereo sensors in the particle's half), tp_n_stereo, tp_n_axial,
              tp_trk_layers, tp_trk_stereo (masks of the matched track, 0 if not found)
  trk_*       one entry per KF track
              trk_particle, trk_n_shared, trk_nhits, trk_purity, trk_layers, trk_good, trk_wrong (masks),
              trk_first_layer, trk_last_layer, trk_n_holes, trk_time, trk_time_rel, trk_p, trk_truth_p, trk_dp_rel, trk_is_primary,
              trk_top (tanl > 0), trk_pos (truth is a positron), trk_n_wrong (hits not from trk_particle), trk_cluster (ECal cluster via part_*, -1 if none),
              trk_matched (purity >= min_purity and truth is a primary trident particle),
              trk_stereo (mask of stereo layers on the track), trk_n_stereo, trk_n_axial
  hit_*       one entry per (type-1) SVT strip hit
              hit_layer, hit_time, hit_time_rel (= hit_time - Sample.t0), hit_top, hit_is_mc, hit_on_track
  dt_good, dt_wrong   hit time - track time for hits on tracks that do / do not belong to the track's particle

Layer masks have bit (L-1) set for layer L; see svt_tracking.h for the data conventions. Useful masks in expressions:
svt::kFrontLayers (L1-L8), svt::kBackLayers (L9-L14), svt::kStereoTop / svt::kStereoBot; e.g. the number of stereo
hits in L9-L14 on the matched track is "svt::popcount(tp_trk_stereo & svt::kBackLayers)".

Stereo sensors are the even layers in the top half and the odd layers in the bottom half. In the 2021 detector the
whole L9 bottom electron-side stereo sensor is dead (conditions DB and data), see KF_REQUIREMENTS / DEAD_SENSORS.
"""
import glob
import itertools
import os
from dataclasses import dataclass

import numpy as np
import ROOT as R
from scipy.stats import beta

__all__ = ["SAMPLES", "DATA", "load_hits", "Sample", "KF_REQUIREMENTS", "DEAD_SENSORS", "TIERS", "TIER_DOC", "CATEGORIES", "CHARGE_COLORS", "cat_style", "tier_efficiency", "detect_hit_type", "setup", "load", "book_hist", "book_hist2d", "book_ratio", "book_ratio_of", "book_ratio2d",
           "run", "plot_hist", "plot_ratio", "plot_eff_map", "style", "mpl_style", "quantile", "central_width", "window_fraction",
           "clopper_pearson", "LAYERS", "COLORS", "layer_bins", "int_bins"]

_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svt_tracking.h")
_MC_DIR = "/net/data/endeavour2/HPS/physrun2021/home_made_mc"

LAYERS = np.arange(1, 15)

# Fixed, validated categorical colours: identity follows the sample, never the plotting order.
COLORS = {"pure": "#2a78d6", "overlay": "#eb6834", "third": "#1baf7a", "ink": "#3d3d3a", "muted": "#8a8984"}

# Particle categories: charge x detector half. Colour = charge (validated pair), marker fill = half (filled = top,
# open = bottom), so the half never relies on colour alone. `tp` / `trk` are the selections for truth particles and
# for truth-matched tracks (track half from the reconstructed tan(lambda)).
CHARGE_COLORS = {"e+": "#e34948", "e-": "#2a78d6"}
CATEGORIES = {
    "e+ top":    dict(tp="tp_pos && tp_top",   trk="trk_pos && trk_top"),
    "e+ bottom": dict(tp="tp_pos && !tp_top",  trk="trk_pos && !trk_top"),
    "e- top":    dict(tp="!tp_pos && tp_top",  trk="!trk_pos && trk_top"),
    "e- bottom": dict(tp="!tp_pos && !tp_top", trk="!trk_pos && !trk_top"),
}


def cat_style(cat, line=False):
    """Matplotlib keyword arguments for a category of CATEGORIES (for errorbar/plot)."""
    charge, half = cat.split()
    kw = dict(color=CHARGE_COLORS[charge], marker="o" if half == "top" else "s",
              mfc=CHARGE_COLORS[charge] if half == "top" else "white", mew=1.5)
    if line:
        kw["ls"] = "-" if half == "top" else "--"
    return kw


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    pattern: str
    description: str
    t0: float = 0.0   # ns; median time of truth-matched tracks. The readout time offset differs between samples.
    hit_type: int = None   # svt_hit_type referenced by the tracks; None = detect from the first file.

    def files(self, n_files=None):
        f = sorted(glob.glob(self.pattern))
        if not f:
            raise FileNotFoundError(f"No files match {self.pattern}")
        return f[:n_files] if n_files else f


SAMPLES = {
    "pure": Sample("pure", "tritrig, pure MC",
                   f"{_MC_DIR}/tritrig_pure_run14272/minidst/tritrig_3pt74_minidst_*.root",
                   "3.74 GeV tritrig, run 14272 conditions, KF tracking, no beam background.", t0=9.9),
    "overlay": Sample("overlay", "tritrig + pulser overlay",
                      f"{_MC_DIR}/tritrig_pulser_beamspot_run14272/minidst/tritrig_pulser_3pt74_minidst_*.root",
                      "Same physics with real 2021 pulser-trigger data overlaid (beam background).", t0=-0.1),
}

_is_setup = False

# Kalman pattern-recognition requirements of the step-1 samples (steering
# PhysicsRun2021MCRecon_KF_WithSpacing_pass5_physics_HitSmear_HitKill_14272.lcsim and KalmanParams.java, identical at
# hps-java commits 473c732da (pure) and 4d47d0383 (overlay)). Two pattern-recognition iterations are run.
KF_REQUIREMENTS = {
    "min hits (iteration 1 / 2)": "9 / 8 (top and bottom)",
    "min stereo hits (iteration 1 / 2)": "4 / 3",
    "min axial hits": "2",
    "seed": "5 hits from one of 20 fixed module patterns (16 in iteration 1), e.g. '000BBS0' = both sensors of "
            "modules 4 and 5 + the stereo sensor of module 6",
    "max hit-time spread on a track": "40 ns (maxTimeRange)",
}

# Dead SVT channels in the 2021 conditions (svt_bad_channels collection 4544, runs 14000-15000) that matter here.
DEAD_SENSORS = {
    "L9 bottom, electron side, stereo (svt_hit_module 1)": "640 / 640 strips: whole sensor, also no hits in data",
    "L13 top, electron side, axial": "128 / 640 strips (one APV25)",
    "L14 top, positron side, stereo": "54 / 640 strips",
}

# Efficiency tiers (cumulative). tp_tier = number of tiers a truth particle passes (0 = not found ... 5 = ECal match),
# so "tier k reached" is tp_tier >= k+1. The denominator (acceptance) is chosen by the user, e.g. tp_acc or
# tp_n_layers >= 10.
TIERS = ["found", "clean", "good p", "in time", "ECal match"]
TIER_DOC = {
    "found": "a track with this particle on >= min_purity of its hits (and on the most hits)",
    "clean": "found, and the track has at most `max_wrong` hits from other particles",
    "good p": "clean, and |p_reco - p_true| / p_true < dp_cut",
    "in time": "good p, and |track time - t0| < t_window",
    "ECal match": "in time, and the track's reconstructed particle has the ECal cluster made by this particle",
}


# Real data (no MC truth): use load_hits(), not load().
DATA = {
    "run14229": Sample("run14229", "data, run 14229 (pass5_v9)",
                       "/net/data/endeavour2/HPS/physrun2021/hps_014229_noskim_mdst/hps_014229_job*_minidst.root",
                       "3.74 GeV, unskimmed, pass5_v9 recon.", t0=1.0, hit_type=0),
}


def load_hits(sample, n_files=None):
    """RDataFrame node with hit- and track-level columns that need no MC truth, for data or MC.

    Columns: hit_sel, n_svt_hits, hit_layer, hit_module, hit_top, hit_time, hit_time_rel, n_tracks, trk_nhits,
    trk_p, trk_top, trk_time_rel."""
    setup()
    s = (DATA.get(sample) or SAMPLES[sample]) if isinstance(sample, str) else sample
    files = s.files(n_files)
    ht = s.hit_type if s.hit_type is not None else detect_hit_type(files)
    defs = [("hit_sel", f"svt_hit_type == {ht}"),
            ("n_svt_hits", "(int)Sum(hit_sel)"),
            ("hit_layer", "svt_hit_layer[hit_sel]"),
            ("hit_module", "svt_hit_module[hit_sel]"),
            ("hit_top", "hit_module % 2 == 0"),
            ("hit_time", "svt_hit_time[hit_sel]"),
            ("hit_time_rel", f"hit_time - ({s.t0})"),
            ("n_tracks", "(int)track_n_hits.size()"),
            ("trk_nhits", "ROOT::RVec<int>(track_n_hits)"),
            ("trk_p", "sqrt(track_px*track_px + track_py*track_py + track_pz*track_pz)"),
            ("trk_top", "ROOT::RVec<double>(track_tan_lambda) > 0"),
            ("trk_time_rel", f"ROOT::RVec<double>(track_time) - ({s.t0})")]
    node = R.RDF.AsRNode(R.RDataFrame("MiniDST", files))
    for name, expr in defs:
        node = node.Define(name, expr)
    return node


def detect_hit_type(files, max_events=2000):
    """Return the svt_hit_type that track_svt_hits points to (0 or 1), from the first events of the first file."""
    f = R.TFile.Open(files[0])
    t = f.Get("MiniDST")
    for i in range(min(max_events, t.GetEntries())):
        t.GetEntry(i)
        types = t.svt_hit_type
        for hits in t.track_svt_hits:
            for h in hits:
                if 0 <= h < len(types):
                    ht = int(types[h])
                    f.Close()
                    return ht
    f.Close()
    return 1


def setup(n_threads=0):
    """Enable ROOT multithreading (0 = all cores) and declare the C++ helpers. Safe to call more than once."""
    global _is_setup
    if _is_setup:
        return
    if n_threads is not None:
        R.EnableImplicitMT(n_threads)
    if not R.gInterpreter.Declare(f'#include "{_HEADER}"'):
        raise RuntimeError(f"Could not declare {_HEADER}")
    _is_setup = True


def load(sample, n_files=None, min_purity=0.5, max_wrong=1, dp_cut=0.15, t_window=6.0):
    """Return an RDataFrame node for `sample` ("pure", "overlay" or a Sample) with all derived columns defined.

    A track is matched to a truth particle when that particle contributes to the most hits on the track and to
    at least a fraction `min_purity` of them. A primary trident particle is *found* when such a track exists.
    The other efficiency tiers (column tp_tier, see TIERS / TIER_DOC) use max_wrong, dp_cut and t_window.
    """
    setup()
    s = SAMPLES[sample] if isinstance(sample, str) else sample
    files = s.files(n_files)
    ht = s.hit_type if s.hit_type is not None else detect_hit_type(files)
    df = R.RDataFrame("MiniDST", files)
    mp = f"{float(min_purity)}"
    defs = [
        # --- event level -------------------------------------------------------------------------------------
        ("hit_sel", f"svt_hit_type == {ht}"),
        ("n_svt_hits", "(int)Sum(hit_sel)"),
        ("hit_is_mc", f"svt::hit_is_mc(svt_hit_type, svt_hit_mc_part_idx, {ht})"),
        ("n_svt_hits_mc", "(int)Sum(hit_is_mc)"),
        ("n_svt_hits_bkg", "n_svt_hits - n_svt_hits_mc"),
        ("n_tracks", "(int)track_n_hits.size()"),
        # --- per hit -----------------------------------------------------------------------------------------
        ("hit_layer", "svt_hit_layer[hit_sel]"),
        ("hit_time", "svt_hit_time[hit_sel]"),
        ("hit_time_rel", f"hit_time - ({s.t0})"),
        ("hit_top", "svt_hit_module[hit_sel] % 2 == 0"),
        ("hit_on_track", f"svt::hit_on_track(svt_hit_type, track_svt_hits, {ht})"),
        # --- per track ---------------------------------------------------------------------------------------
        ("trk_tt", "svt::track_truth(track_svt_hits, svt_hit_layer, svt_hit_mc_part_idx)"),
        ("trk_particle", "trk_tt.particle"),
        ("trk_n_shared", "trk_tt.n_shared"),
        ("trk_nhits", "ROOT::RVec<int>(track_n_hits)"),
        ("trk_purity", "ROOT::RVec<double>(trk_n_shared) / ROOT::RVec<double>(trk_nhits)"),
        ("trk_layers", "trk_tt.hit_mask"),
        ("trk_good", "trk_tt.good_mask"),
        ("trk_wrong", "svt::and_not(trk_layers, trk_good)"),
        ("trk_first_layer", "svt::first_layer(trk_layers)"),
        ("trk_last_layer", "svt::last_layer(trk_layers)"),
        ("trk_n_holes", "svt::count_in_pattern(track_hit_pattern, -1)"),
        ("trk_p", "sqrt(track_px*track_px + track_py*track_py + track_pz*track_pz)"),
        ("trk_time", "ROOT::RVec<double>(track_time)"),
        ("trk_time_rel", f"trk_time - ({s.t0})"),
        ("trk_top", "ROOT::RVec<double>(track_tan_lambda) > 0"),
        ("trk_n_wrong", "trk_nhits - trk_n_shared"),
        ("trk_pos", "svt::take_or(mc_part_pdg, trk_particle, 0) == -11"),
        ("trk_cluster", "svt::track_cluster((int)track_n_hits.size(), part_track, part_ecal_cluster)"),
        ("trk_stereo", "svt::stereo_layers(trk_layers, trk_top)"),
        ("trk_n_stereo", "svt::popcount(trk_stereo)"),
        ("trk_n_axial", "trk_nhits - trk_n_stereo"),
        # --- per primary truth particle ----------------------------------------------------------------------
        ("tp_idx", "svt::primary_charged(mc_part_pdg, mc_part_parents, mc_part_charge)"),
        ("tp_pdg", "Take(mc_part_pdg, tp_idx)"),
        ("tp_p", "sqrt(Take(mc_part_px, tp_idx)*Take(mc_part_px, tp_idx) + Take(mc_part_py, tp_idx)*"
                 "Take(mc_part_py, tp_idx) + Take(mc_part_pz, tp_idx)*Take(mc_part_pz, tp_idx))"),
        ("tp_tanl", "Take(mc_part_py, tp_idx) / sqrt(Take(mc_part_px, tp_idx)*Take(mc_part_px, tp_idx) + "
                    "Take(mc_part_pz, tp_idx)*Take(mc_part_pz, tp_idx))"),
        ("tp_top", "tp_tanl > 0"),
        ("tp_pos", "tp_pdg == -11"),
        ("tp_gap", "abs(tp_tanl) < 0.015"),
        ("tp_layers", f"svt::truth_layer_mask(tp_idx, svt_hit_type, svt_hit_layer, svt_hit_mc_part_idx, {ht})"),
        ("tp_n_layers", "svt::popcount(tp_layers)"),
        ("tp_last_layer", "svt::last_layer(tp_layers)"),
        ("tp_n_svt_hits", "ROOT::RVec<int>(tp_idx.size(), n_svt_hits)"),
        ("tp_ecal_e", "svt::truth_ecal_energy(tp_idx, ecal_hit_mc_parent_id, ecal_hit_energy)"),
        ("tp_track", f"svt::match_track(tp_idx, trk_particle, trk_n_shared, trk_nhits, {mp})"),
        ("tp_found", "tp_track >= 0"),
        ("tp_n_tracks", f"svt::n_matched_tracks(tp_idx, trk_particle, trk_n_shared, trk_nhits, {mp})"),
        ("tp_trk_nhits", "svt::take_or(trk_nhits, tp_track, 0)"),
        ("tp_missing", "svt::and_not(tp_layers, svt::take_or(trk_good, tp_track, 0))"),
        ("tp_stereo", "svt::stereo_layers(tp_layers, tp_top)"),
        ("tp_n_stereo", "svt::popcount(tp_stereo)"),
        ("tp_n_axial", "tp_n_layers - tp_n_stereo"),
        ("tp_trk_layers", "svt::take_or(trk_layers, tp_track, 0)"),
        ("tp_trk_stereo", "svt::take_or(trk_stereo, tp_track, 0)"),
        # --- track <-> truth -------------------------------------------------------------------------------
        ("trk_is_primary", "svt::is_in(trk_particle, tp_idx)"),
        ("trk_matched", f"trk_is_primary && trk_purity >= {mp}"),
        ("trk_truth_p", "sqrt(svt::take_or(mc_part_px, trk_particle, 0.)*svt::take_or(mc_part_px, trk_particle, 0.)"
                        " + svt::take_or(mc_part_py, trk_particle, 0.)*svt::take_or(mc_part_py, trk_particle, 0.)"
                        " + svt::take_or(mc_part_pz, trk_particle, 0.)*svt::take_or(mc_part_pz, trk_particle, 0.))"),
        ("trk_dp_rel", "(trk_p - trk_truth_p) / trk_truth_p"),
        # --- efficiency tiers for the truth particles (cumulative, see TIERS) --------------------------------
        ("tp_acc", "tp_ecal_e > 0.1"),
        ("tp_trk_dp_rel", "svt::take_or(trk_dp_rel, tp_track, -9.)"),
        ("tp_clean", f"tp_found && svt::take_or(trk_n_wrong, tp_track, 99) <= {int(max_wrong)}"),
        ("tp_good_p", f"tp_clean && abs(svt::take_or(trk_dp_rel, tp_track, 9.)) < {float(dp_cut)}"),
        ("tp_in_time", f"tp_good_p && abs(svt::take_or(trk_time_rel, tp_track, 999.)) < {float(t_window)}"),
        ("tp_ecal_match", "tp_in_time && svt::cluster_is_true(tp_idx, svt::take_or(trk_cluster, tp_track, -1), ecal_cluster_mc_id)"),
        ("tp_tier", "ROOT::RVec<int>(tp_found) + ROOT::RVec<int>(tp_clean) + ROOT::RVec<int>(tp_good_p) + "
                    "ROOT::RVec<int>(tp_in_time) + ROOT::RVec<int>(tp_ecal_match)"),
        ("dt_good", "svt::hit_minus_track_time(track_svt_hits, track_time, trk_particle, svt_hit_time, "
                    "svt_hit_mc_part_idx, true)"),
        ("dt_wrong", "svt::hit_minus_track_time(track_svt_hits, track_time, trk_particle, svt_hit_time, "
                     "svt_hit_mc_part_idx, false)"),
    ]
    node = R.RDF.AsRNode(df)
    for name, expr in defs:
        node = node.Define(name, expr)
    return node


# ------------------------------------------------------------------------------------------------------------------
# Lazy booking: everything is booked first and filled in a single event loop per sample with run().
# ------------------------------------------------------------------------------------------------------------------
_uid = itertools.count()


def _edges(bins):
    if isinstance(bins, tuple) and len(bins) == 3:
        n, lo, hi = bins
        return np.linspace(lo, hi, int(n) + 1)
    return np.asarray(bins, dtype=float)


def int_bins(lo, hi):
    """Bin edges centred on the integers lo..hi."""
    return np.arange(lo - 0.5, hi + 1.0, 1.0)


def layer_bins():
    return int_bins(1, 14)


def _select(df, expr, where):
    """Define a temporary column `expr` restricted to the entries where `where` is true (both per-entry RVecs)."""
    name = f"_v{next(_uid)}"
    return df.Define(name, f"({expr})[{where}]" if where else expr), name


class _Booked:
    """Base for lazily booked results. Subclasses keep a list of RResultPtr handles in self._handles."""
    _handles = ()

    def handles(self):
        return list(self._handles)


class Hist(_Booked):
    """A booked 1D histogram; `.counts`, `.edges` and `.centers` are numpy arrays after run()."""

    def __init__(self, df, expr, bins, where=None, weight=None):
        self.edges = _edges(bins)
        df, col = _select(df, expr, where)
        model = R.RDF.TH1DModel(col, "", len(self.edges) - 1, self.edges)
        if weight:
            df, wcol = _select(df, weight, where)
            self._h = df.Histo1D(model, col, wcol)
        else:
            self._h = df.Histo1D(model, col)
        self._handles = [self._h]

    @property
    def counts(self):
        h = self._h.GetValue()
        return np.array([h.GetBinContent(i + 1) for i in range(len(self.edges) - 1)])

    @property
    def centers(self):
        return 0.5 * (self.edges[1:] + self.edges[:-1])

    @property
    def underflow_overflow(self):
        h = self._h.GetValue()
        return h.GetBinContent(0), h.GetBinContent(len(self.edges))


class Hist2D(_Booked):
    """A booked 2D histogram; `.counts` has shape (nx, ny) after run()."""

    def __init__(self, df, xexpr, yexpr, xbins, ybins, where=None):
        self.xedges, self.yedges = _edges(xbins), _edges(ybins)
        df, xc = _select(df, xexpr, where)
        df, yc = _select(df, yexpr, where)
        model = R.RDF.TH2DModel(xc + yc, "", len(self.xedges) - 1, self.xedges, len(self.yedges) - 1, self.yedges)
        self._h = df.Histo2D(model, xc, yc)
        self._handles = [self._h]

    @property
    def counts(self):
        h = self._h.GetValue()
        nx, ny = len(self.xedges) - 1, len(self.yedges) - 1
        return np.array([[h.GetBinContent(i + 1, j + 1) for j in range(ny)] for i in range(nx)])


def clopper_pearson(k, n, cl=0.6827):
    """Efficiency k/n with central Clopper-Pearson interval; returns (eff, err_low, err_high), NaN where n = 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        eff = np.where(n > 0, k / n, np.nan)
        a = (1 - cl) / 2
        lo = np.where(k > 0, beta.ppf(a, k, n - k + 1), 0.0)
        hi = np.where(k < n, beta.ppf(1 - a, k + 1, n - k), 1.0)
    lo, hi = np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)
    return eff, eff - lo, hi - eff


class Ratio(_Booked):
    """Fraction of entries passing `passed`, in bins of `expr`, for entries where `where` holds.

    After run(): .passed, .total (counts), .eff, .err (2 x n asymmetric Clopper-Pearson errors), .centers."""

    def __init__(self, sel, all_):
        self.sel, self.all = sel, all_
        self.edges = self.all.edges
        self._handles = self.all.handles() + self.sel.handles()

    @property
    def centers(self):
        return self.all.centers

    @property
    def total(self):
        return self.all.counts

    @property
    def passed(self):
        return self.sel.counts

    @property
    def eff(self):
        return clopper_pearson(self.passed, self.total)[0]

    @property
    def err(self):
        _, lo, hi = clopper_pearson(self.passed, self.total)
        return np.vstack([lo, hi])

    def integral(self):
        """Overall fraction (with Clopper-Pearson errors) summed over all bins, including under/overflow."""
        k = self.passed.sum() + sum(self.sel.underflow_overflow)
        n = self.total.sum() + sum(self.all.underflow_overflow)
        e, lo, hi = clopper_pearson(k, n)
        return float(e), float(lo), float(hi)


class Ratio2D(_Booked):
    """2D version of Ratio; `.eff` and `.total` have shape (nx, ny)."""

    def __init__(self, df, xexpr, yexpr, passed, xbins, ybins, where=None):
        cond = f"({where}) && ({passed})" if where else passed
        self.all = Hist2D(df, xexpr, yexpr, xbins, ybins, where)
        self.sel = Hist2D(df, xexpr, yexpr, xbins, ybins, cond)
        self.xedges, self.yedges = self.all.xedges, self.all.yedges
        self._handles = self.all.handles() + self.sel.handles()

    @property
    def total(self):
        return self.all.counts

    @property
    def eff(self):
        return clopper_pearson(self.sel.counts, self.all.counts)[0]


def book_hist(df, expr, bins, where=None, weight=None):
    return Hist(df, expr, bins, where, weight)


def book_hist2d(df, xexpr, yexpr, xbins, ybins, where=None):
    return Hist2D(df, xexpr, yexpr, xbins, ybins, where)


def book_ratio(df, expr, passed, bins, where=None):
    """Fraction of the entries of `expr` (restricted to `where`) that also satisfy `passed`, in bins of `expr`."""
    cond = f"({where}) && ({passed})" if where else passed
    return Ratio(Hist(df, expr, bins, cond), Hist(df, expr, bins, where))


def book_ratio_of(df, num_expr, den_expr, bins):
    """Ratio of the histograms of two different expressions, e.g. per-layer lists from svt::mask_to_layers().
    The entries of num_expr must be a subset of those of den_expr for the binomial errors to be meaningful."""
    return Ratio(Hist(df, num_expr, bins), Hist(df, den_expr, bins))


def book_ratio2d(df, xexpr, yexpr, passed, xbins, ybins, where=None):
    return Ratio2D(df, xexpr, yexpr, passed, xbins, ybins, where)


def run(*booked):
    """Fill everything booked (any nesting of lists/dicts of booked objects) with one event loop per data frame."""
    flat = []

    def collect(x):
        if isinstance(x, _Booked):
            flat.extend(x.handles())
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                collect(v)

    collect(booked)
    R.RDF.RunGraphs(flat)


# ------------------------------------------------------------------------------------------------------------------
# Small analysis helpers on filled histograms
# ------------------------------------------------------------------------------------------------------------------
def quantile(counts, edges, q):
    """Quantile q (0..1) of a binned distribution, linearly interpolated inside the bin."""
    c = np.concatenate([[0.0], np.cumsum(counts)])
    if c[-1] <= 0:
        return np.nan
    return float(np.interp(q * c[-1], c, edges))


def central_width(hist2d, min_entries=50):
    """For each x bin of a Hist2D: (median, half-width of the central 68% interval) of the y distribution.
    A robust resolution estimate that is insensitive to non-Gaussian tails. NaN where < min_entries."""
    med, width = [], []
    for row in hist2d.counts:
        if row.sum() < min_entries:
            med.append(np.nan); width.append(np.nan)
            continue
        lo, m, hi = (quantile(row, hist2d.yedges, q) for q in (0.1587, 0.5, 0.8413))
        med.append(m); width.append(0.5 * (hi - lo))
    return np.array(med), np.array(width)


def window_fraction(hist, half_widths, center=0.0):
    """Fraction of all entries (incl. under/overflow) of a Hist inside |x - center| < w, for each w.
    Bin edges should fall on the window boundaries for an exact answer; otherwise it is interpolated."""
    c = np.concatenate([[0.0], np.cumsum(hist.counts)])
    total = c[-1] + sum(hist.underflow_overflow)
    w = np.asarray(half_widths, float)
    inside = np.interp(center + w, hist.edges, c) - np.interp(center - w, hist.edges, c)
    return inside / total


# ------------------------------------------------------------------------------------------------------------------
# Plotting (matplotlib)
# ------------------------------------------------------------------------------------------------------------------
def tier_efficiency(hist):
    """Cumulative tier efficiencies from a booked Hist of tp_tier (bins int_bins(0, len(TIERS))).
    Returns (eff, err) with eff[k] = fraction reaching TIERS[k], err = (2, n) Clopper-Pearson errors, and N."""
    c = hist.counts
    n = c.sum()
    k = np.array([c[i + 1:].sum() for i in range(len(TIERS))])
    eff, err_lo, err_hi = clopper_pearson(k, np.full(len(k), n))
    return eff, np.vstack([err_lo, err_hi]), n


def mpl_style():
    """Set matplotlib defaults for this study: readable sizes, thin recessive axes."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110, "figure.constrained_layout.use": True,
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.edgecolor": COLORS["muted"], "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["ink"], "ytick.color": COLORS["ink"], "text.color": COLORS["ink"],
        "lines.linewidth": 2, "image.cmap": "Blues",
    })


def plot_hist(ax, hist, label=None, color=None, density=False, **kw):
    """Step plot of a booked Hist. density=True normalises to unit area."""
    y = hist.counts
    if density and y.sum() > 0:
        y = y / (y.sum() * np.diff(hist.edges))
    ax.stairs(y, hist.edges, label=label, color=color, linewidth=2, **kw)
    return ax


def plot_ratio(ax, ratio, label=None, color=None, min_total=1, **kw):
    """Points with asymmetric Clopper-Pearson error bars for a booked Ratio; bins with < min_total entries hidden."""
    ok = ratio.total >= min_total
    x = ratio.centers[ok]
    xerr = 0.5 * np.diff(ratio.edges)[ok]
    ax.errorbar(x, ratio.eff[ok], yerr=ratio.err[:, ok], xerr=xerr, fmt="o", ms=4, lw=1.5, capsize=0,
                color=color, label=label, **kw)
    return ax


def plot_eff_map(ax, ratio2d, min_total=20, fontsize=6.5, label="efficiency"):
    """Colour map of a booked Ratio2D with the value written in each cell; cells with < min_total entries blank.
    Returns the QuadMesh (for a colorbar)."""
    eff = np.where(ratio2d.total >= min_total, ratio2d.eff, np.nan)
    im = ax.pcolormesh(ratio2d.xedges, ratio2d.yedges, eff.T, vmin=0, vmax=1, cmap="Blues")
    xc = 0.5 * (ratio2d.xedges[1:] + ratio2d.xedges[:-1])
    yc = 0.5 * (ratio2d.yedges[1:] + ratio2d.yedges[:-1])
    for i, x in enumerate(xc):
        for j, y in enumerate(yc):
            if np.isfinite(eff[i, j]):
                ax.text(x, y, f"{eff[i, j]:.2f}", ha="center", va="center", fontsize=fontsize,
                        color="white" if eff[i, j] > 0.6 else COLORS["ink"])
    return im


def style(ax, xlabel=None, ylabel=None, title=None, legend=True, grid=True):
    """Common axis styling: recessive grid, labels, legend without frame."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize="medium")
    if grid:
        ax.grid(True, color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)
    return ax
