"""! @package svt_hitfilter
Machine-learned SVT hit filter for HPS 2021: features, datasets, KF findability emulator, model export.

The filter scores every 1D strip cluster ("StripClusterer_SiTrackerHitStrip1D") before Kalman pattern recognition;
hits below a threshold are removed. Features are computed per event by svt::hf::features() in svt_hitfilter.h; the
Java driver (SVT/Hit_Filtering/hps-java) re-implements the same features and evaluates the exported model.

Typical use::

    from Analysis import svt_tracking as st, svt_hitfilter as hf
    hf.setup()
    mc = hf.collect(hf.mc_frame("overlay", files))            # dict of numpy arrays, one row per hit
    data = hf.collect(hf.data_frame("run14229", files, "pulser"))
    model.fit(mc["X"], mc["is_mc"]) ...
    hf.export_model([model], "model.txt", stage_names=["mc"])

Training on data without truth (positive-unlabelled learning): hits from physics-trigger events are a mixture of
signal and background; hits from pulser (random) trigger events are background only. A classifier trained to
separate the two estimates r(x) = p_phys(x) / p_pulser(x), which is monotonic in the signal/background likelihood
ratio; pu_signal_probability() converts it to P(signal | x) in a physics event.
"""
import glob
import os

import numpy as np
import ROOT as R

from . import svt_tracking as st

__all__ = ["FEATURES", "FEATURE_DOC", "MONOTONE", "T0", "TARGET_Z", "PULSER_BIT", "setup", "mc_frame", "data_frame",
           "collect", "findable_lut", "findable", "layer_masks", "export_model", "load_model", "predict_exported",
           "pu_signal_probability", "golden_track_expr"]

_HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svt_hitfilter.h")

FEATURES = ["layer", "top", "stereo", "side", "t", "q", "nstrip", "absy", "tl", "nsame", "pair_dt", "pair_dy",
            "has_anchor", "nroad_tight", "nroad_loose"] + [f"dtl_m{m}" for m in range(1, 8)] + ["noot_half",
            "road_adt", "road_at", "nroad_t", "ncomp"]
FEATURE_DOC = {
    "layer": "SVT layer 1-14", "top": "1 = top half", "stereo": "1 = stereo sensor",
    "side": "1 = positron-side sensor (L9-L14)", "t": "hit time - t0 [ns]", "q": "cluster charge (edep x 1e6)",
    "nstrip": "strips in the cluster", "absy": "|y| [mm]", "tl": "y/(z - zT), > 0 away from the beam plane",
    "nsame": "other in-time (10 ns) hits on the same sensor",
    "pair_dt": "partner sensor: min |dt| of hits within 4 mm in y", "pair_dy": "partner sensor: min |dy| of hits within 8 ns",
    "has_anchor": "an axial hit (itself or its in-time partner) defines the road",
    "nroad_tight": "other modules with an in-time axial hit on the same target-pointing line (|d tanl| < 0.002)",
    "nroad_loose": "same, |d tanl| < 0.006", "noot_half": "out-of-time (|t| > 25 ns) hits in this half: background occupancy",
    **{f"dtl_m{m}": f"min |d tanl| to an in-time axial hit in module {m} (1 = none)" for m in range(1, 8)},
    "road_adt": "|t - mean time of the road hits| [ns] (100 = no road)", "road_at": "|mean time of the road hits| [ns]",
    "nroad_t": "road hits in the mean time", "ncomp": "in-time hits on the same sensor within 2 mm (competitors)",
}
# Monotonic constraints (+1 increasing, -1 decreasing signal-likeness): physics priors that make the model behave
# sensibly where data and MC differ. 0 = unconstrained.
MONOTONE = {"has_anchor": 1, "nroad_tight": 1, "nroad_loose": 1, "pair_dt": -1, "pair_dy": -1, "nroad_t": 1,
            "road_adt": -1, "road_at": -1,
            **{f"dtl_m{m}": -1 for m in range(1, 8)}}

# Signal time per sample [ns] (peak of hits on matched tracks) and target z [mm] used by the road features.
T0 = {"pure": 9.9, "overlay": -0.1, "run14229": 0.4}   # run14229: median time of hits on golden tracks
TARGET_Z = -1.1
PULSER_BIT = 15      # TSData2019: bit 15 = pulser (random) trigger

_is_setup = False


def setup(n_threads=0):
    global _is_setup
    st.setup(n_threads)
    if not _is_setup:
        if not R.gInterpreter.Declare(f'#include "{_HEADER}"'):
            raise RuntimeError(f"Could not declare {_HEADER}")
        _is_setup = True


def _feat_expr(t0):
    return (f"svt::hf::features(hf_use, svt_hit_layer, svt_hit_module, svt_hit_time, svt_hit_y, svt_hit_z, "
            f"svt_hit_edep, svt_hit_strip, {float(t0)}, {TARGET_Z})")


def mc_frame(sample, files, drop_mc=False, t0=None, parity=None):
    """MC (with truth) frame for an explicit list of files of `sample` ("pure"/"overlay").

    Adds per-hit columns hf_feat (flattened features), hf_is_mc, hf_prim (primary particle or -1) and hf_code
    (track code, see svt::hf::track_code) to the columns of svt_tracking.load().
    drop_mc=True removes every hit with MC truth before the features are computed: the overlay event then becomes the
    pulser event it was made from, which emulates pulser triggers in MC. parity = 0 / 1 keeps only even / odd
    entries, to split the same files into disjoint "physics" and "pulser" event sets with the same background."""
    setup()
    s = _FileSample(st.SAMPLES[sample] if isinstance(sample, str) else sample, files)
    df = st.load(s)
    if parity is not None:
        df = df.Filter(f"abs(event_number) % 2 == {int(parity)}")   # rdfentry_ is not reproducible with MT
    ht = s.hit_type if s.hit_type is not None else st.detect_hit_type(s.files())
    use = f"svt_hit_type == {ht}"
    if drop_mc:
        use += " && svt::hf::is_mc(ROOT::RVec<bool>(svt_hit_type.size(), true), svt_hit_mc_part_idx) == 0"
    t0 = T0.get(s.key, s.t0) if t0 is None else t0
    return (df.Define("hf_use", f"ROOT::RVec<bool>({use})")
              .Define("hf_feat", _feat_expr(t0))
              .Define("hf_is_mc", "svt::hf::is_mc(hf_use, svt_hit_mc_part_idx)")
              .Define("hf_prim", "svt::hf::primary_of(hf_use, svt_hit_mc_part_idx, tp_idx)")
              .Define("hf_code", "svt::hf::track_code(hf_use, track_svt_hits, trk_particle, trk_matched, svt_hit_mc_part_idx)")
              .Define("hf_trk_golden", f"ROOT::RVec<bool>({golden_track_expr(t0)})")
              .Define("hf_on_golden", "svt::hf::on_good_track(hf_use, track_svt_hits, hf_trk_golden)")
              .Define("hf_golden_trk", "svt::hf::good_track_index(hf_use, track_svt_hits, hf_trk_golden)")
              .Define("hf_trk_rows", "svt::hf::track_hit_rows(hf_use, track_svt_hits, ROOT::RVec<double>(track_time), "
                      "ROOT::RVec<int>(track_n_hits), ROOT::RVec<double>(track_chi2), trk_particle, trk_matched, svt_hit_time, "
                      "svt_hit_mc_part_idx)"))


class _FileSample(st.Sample):
    """A Sample restricted to an explicit list of files."""

    def __init__(self, s, files):
        for k in ("key", "label", "description", "t0", "hit_type"):
            object.__setattr__(self, k, getattr(s, k))
        object.__setattr__(self, "pattern", "<file list>")
        object.__setattr__(self, "_files", list(files))

    def files(self, n_files=None):
        return self._files[:n_files] if n_files else self._files


def golden_track_expr(t0, min_hits=12, max_chi2ndf=3.0, min_p=1.0, dt=3.0):
    """Per-track boolean expression for 'golden' data tracks used as the signal tag: in time, many hits, good fit,
    p > min_p and linked to an ECal cluster."""
    return (f"abs(ROOT::RVec<double>(track_time) - ({t0})) < {dt} && track_n_hits >= {min_hits} && "
            f"ROOT::RVec<double>(track_chi2) / (ROOT::RVec<double>(track_n_hits) - 5) < {max_chi2ndf} && "
            f"sqrt(track_px*track_px + track_py*track_py + track_pz*track_pz) > {min_p} && "
            f"svt::track_cluster((int)track_n_hits.size(), part_track, part_ecal_cluster) >= 0")


def data_frame(sample, files, trigger="physics", t0=None):
    """Data frame (no truth) restricted to physics triggers (not pulser) or pulser triggers.
    Per-hit columns: hf_feat, hf_on_track, hf_on_golden, hf_golden_trk; per event: hf_ntrk."""
    setup()
    s = st.DATA[sample] if isinstance(sample, str) else sample
    t0 = T0.get(s.key, s.t0) if t0 is None else t0
    ht = s.hit_type if s.hit_type is not None else st.detect_hit_type(files)
    df = R.RDF.AsRNode(R.RDataFrame("MiniDST", files))
    bit = f"((trigger >> {PULSER_BIT}) & 1u) != 0"
    df = df.Filter(bit if trigger == "pulser" else f"!({bit})", trigger)
    gold = golden_track_expr(t0)
    return (df.Define("hf_use", f"ROOT::RVec<bool>(svt_hit_type == {ht})")
              .Define("hf_feat", _feat_expr(t0))
              .Define("hf_trk_golden", f"ROOT::RVec<bool>({gold})")
              .Define("hf_all_trk", "ROOT::RVec<bool>(track_n_hits.size(), true)")
              .Define("hf_on_track", "svt::hf::on_good_track(hf_use, track_svt_hits, hf_all_trk)")
              .Define("hf_on_golden", "svt::hf::on_good_track(hf_use, track_svt_hits, hf_trk_golden)")
              .Define("hf_golden_trk", "svt::hf::good_track_index(hf_use, track_svt_hits, hf_trk_golden)"))


TRK_COLS = ["hit", "track", "dt", "trk_nhits", "trk_chi2ndf", "nshare", "label"]
TRK_COLS_N = len(TRK_COLS)


def collect(df, per_hit=("hf_is_mc", "hf_prim", "hf_code"), per_particle=None, track_rows=False):
    """Run the frame and return numpy arrays: X (n_hits x n_features), the per_hit columns, `event` (event index per
    hit) and, if per_particle is given, a dict 'tp' of flattened per-particle columns with their own `event`."""
    cols = ["hf_feat", "rdfentry_", "event_number"] + list(per_hit) + list(per_particle or []) + (["hf_trk_rows"] if track_rows else [])
    a = df.AsNumpy(cols)
    # With implicit multithreading neither the order of the events nor the values of rdfentry_ are reproducible
    # (rdfentry_ is unique per event, so it is still used to group hits). Sort events by content (event number,
    # number of hits, first feature row) so that training and any subsampling are deterministic.
    nf_ = len(FEATURES)
    first = np.array([np.asarray(v[:nf_], np.float64).sum() if len(v) else 0.0 for v in a["hf_feat"]])
    nhit = np.array([len(v) for v in a["hf_feat"]])
    order = np.lexsort((first, nhit, a["event_number"]))
    a = {k: v[order] for k, v in a.items()}
    feats = a["hf_feat"]
    nf = len(FEATURES)
    nh = np.array([len(v) // nf for v in feats])
    out = {"X": np.concatenate([np.asarray(v, dtype=np.float32) for v in feats]).reshape(-1, nf) if len(feats) else
           np.zeros((0, nf), np.float32),
           "event": np.repeat(a["rdfentry_"].astype(np.int64), nh), "n_events": len(feats),
           "event_number": np.repeat(a["event_number"].astype(np.int64), nh)}
    for c in per_hit:
        out[c[3:] if c.startswith("hf_") else c] = np.concatenate([np.asarray(v) for v in a[c]])
    if "hf_trk_rows" in a:
        rows = a["hf_trk_rows"]
        nr = np.array([len(v) // TRK_COLS_N for v in rows])
        R_ = np.concatenate([np.asarray(v, np.float32) for v in rows]).reshape(-1, TRK_COLS_N)
        offs = np.repeat(np.concatenate([[0], np.cumsum(nh)[:-1]]), nr)      # first hit row of each event
        out["trk"] = {c: R_[:, k] for k, c in enumerate(TRK_COLS)}
        out["trk"]["hit_row"] = (offs + R_[:, 0]).astype(np.int64)          # row of the hit in X
        out["trk"]["event"] = np.repeat(a["rdfentry_"].astype(np.int64), nr)
    if per_particle:
        n_p = np.array([len(v) for v in a[per_particle[0]]])
        tp = {"event": np.repeat(a["rdfentry_"].astype(np.int64), n_p)}
        for c in per_particle:
            tp[c] = np.concatenate([np.asarray(v) for v in a[c]])
        out["tp"] = tp
    return out


# ----------------------------------------------------------------------------------------------------------------
# KF findability emulator (requirements of the pass5 steering, see svt_tracking.KF_REQUIREMENTS)
# ----------------------------------------------------------------------------------------------------------------
KF_STRATEGIES = ["000BBS0", "00BBS00", "00ASBS0", "00ABSS0", "0A0SBS0", "00B0BS0", "00SBSA0", "00SBB00", "00SBAS0",
                 "0SA0BS0", "0000SBB", "000SABS", "0BBS000", "0SBB000", "0000BBS", "000SSBA", "000BSSA", "ABSS000",
                 "SBB0000", "SABS000"]
KF_N_ITER1 = 16


def layer_masks(top):
    """(axial, stereo) layer masks of one half."""
    st_mask = sum(1 << (L - 1) for L in range(1, 15) if (L % 2 == 0) == bool(top))
    return (0x3FFF & ~st_mask), st_mask


def _strategy_masks(top):
    out = []
    for s in KF_STRATEGIES:
        need = 0
        for m, c in enumerate(s):
            ax_layer = 2 * m + 1 if top else 2 * m + 2
            st_layer = 2 * m + 2 if top else 2 * m + 1
            if c in "AB":
                need |= 1 << (ax_layer - 1)
            if c in "SB":
                need |= 1 << (st_layer - 1)
        out.append(need)
    return np.array(out)


def findable_lut(top):
    """Boolean lookup table over 14-bit layer masks: would the KF (as configured) be able to find a track with
    hits in exactly these layers? Emulates the hit-count, stereo/axial and seed-pattern requirements of both
    iterations; it does not model fit quality, time spread or hit competition."""
    ax, stm = layer_masks(top)
    m = np.arange(1 << 14)
    pc = np.vectorize(lambda x: bin(x).count("1"))
    n, ns = pc(m), pc(m & stm)
    na = n - ns
    strat = _strategy_masks(top)
    seed = (m[:, None] & strat[None, :]) == strat[None, :]
    it1 = (n >= 9) & (ns >= 4) & (na >= 2) & seed[:, :KF_N_ITER1].any(axis=1)
    it2 = (n >= 8) & (ns >= 3) & (na >= 2) & seed.any(axis=1)
    return it1 | it2


_LUT = {}


def findable(masks, top):
    """Vectorised findability of layer masks (array) for particles/tracks in the given halves (bool array)."""
    if not _LUT:
        _LUT[True], _LUT[False] = findable_lut(True), findable_lut(False)
    masks = np.asarray(masks, np.int64) & 0x3FFF
    top = np.asarray(top, bool)
    return np.where(top, _LUT[True][masks], _LUT[False][masks])


def group_masks(group_keys, layers, keep=None):
    """OR of (1 << (layer-1)) over hits sharing a group key (e.g. event*1000 + particle); returns (keys, masks)."""
    keys, inv = np.unique(group_keys, return_inverse=True)
    bits = (1 << (layers.astype(np.int64) - 1))
    if keep is not None:
        bits = np.where(keep, bits, 0)
    masks = np.zeros(len(keys), np.int64)
    np.bitwise_or.at(masks, inv, bits)
    return keys, masks


# ----------------------------------------------------------------------------------------------------------------
# Positive-unlabelled calibration
# ----------------------------------------------------------------------------------------------------------------
def pu_signal_probability(score, frac_phys_train, signal_fraction):
    """Convert a classifier score s = P(physics-event hit | x), trained with a fraction frac_phys_train of physics
    hits, into P(signal | x) for a hit in a physics event whose signal fraction is signal_fraction:
    r = s/(1-s) * (1-pi)/pi = p_phys/p_bkg, P(signal | x) = 1 - (1 - f)/r (clipped to [0, 1])."""
    s = np.clip(np.asarray(score, float), 1e-9, 1 - 1e-9)
    r = s / (1 - s) * (1 - frac_phys_train) / frac_phys_train
    return np.clip(1 - (1 - signal_fraction) / r, 0, 1)


# ----------------------------------------------------------------------------------------------------------------
# Model export (scikit-learn HistGradientBoostingClassifier -> plain text read by the Java driver)
# ----------------------------------------------------------------------------------------------------------------
def export_model(models, path, stage_names, threshold, meta=None):
    """Write a chain of binary HistGradientBoostingClassifier models to a text file.

    Stage k sees FEATURES + the probabilities of stages 0..k-1 (as extra features "p_<name>"). The final score is the
    probability of the last stage; hits with score < threshold are removed. Format (line based):
      hps-svt-hitfilter 1
      meta <key> <value>                (any number; t0, targetZ, ...)
      features <n> <names...>
      threshold <value>
      stage <name> <n_inputs> <baseline> <n_trees>
      tree <n_nodes>
      <is_leaf> <feature> <threshold> <left> <right> <missing_go_left> <value>    (one line per node)
    Node values and thresholds are written with 17 significant digits."""
    lines = ["hps-svt-hitfilter 1"]
    for k, v in (meta or {}).items():
        lines.append(f"meta {k} {v}")
    lines.append(f"features {len(FEATURES)} " + " ".join(FEATURES))
    lines.append(f"threshold {threshold:.17g}")
    for k, (mdl, name) in enumerate(zip(models, stage_names)):
        if mdl.n_trees_per_iteration_ != 1:
            raise ValueError("only binary classifiers are supported")
        n_in = len(FEATURES) + k
        if mdl.n_features_in_ != n_in:
            raise ValueError(f"stage {name}: expected {n_in} inputs, model has {mdl.n_features_in_}")
        base = float(np.ravel(mdl._baseline_prediction)[0])
        trees = [p[0] for p in mdl._predictors]
        lines.append(f"stage {name} {n_in} {base:.17g} {len(trees)}")
        for tr in trees:
            nd = tr.nodes
            if np.any(nd["is_categorical"]):
                raise ValueError("categorical splits are not supported")
            lines.append(f"tree {len(nd)}")
            for r in nd:
                lines.append(f"{int(r['is_leaf'])} {int(r['feature_idx'])} {float(r['num_threshold']):.17g} "
                             f"{int(r['left'])} {int(r['right'])} {int(r['missing_go_to_left'])} {float(r['value']):.17g}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def load_model(path):
    """Read an exported model into a dict (for predict_exported and inspection)."""
    model = {"meta": {}, "stages": []}
    with open(path) as f:
        lines = [l.split() for l in f if l.strip()]
    i = 0
    assert lines[0][0] == "hps-svt-hitfilter"
    i = 1
    while i < len(lines):
        w = lines[i]
        if w[0] == "meta":
            model["meta"][w[1]] = " ".join(w[2:]); i += 1
        elif w[0] == "features":
            model["features"] = w[2:]; i += 1
        elif w[0] == "threshold":
            model["threshold"] = float(w[1]); i += 1
        elif w[0] == "stage":
            stage = {"name": w[1], "n_inputs": int(w[2]), "baseline": float(w[3]), "trees": []}
            n_trees = int(w[4]); i += 1
            for _ in range(n_trees):
                nn = int(lines[i][1]); i += 1
                arr = np.array([[float(x) for x in l] for l in lines[i:i + nn]])
                stage["trees"].append(arr); i += nn
            model["stages"].append(stage)
        else:
            raise ValueError(f"bad line {w}")
    return model


def _predict_stage(stage, X):
    raw = np.full(len(X), stage["baseline"])
    for tr in stage["trees"]:
        node = np.zeros(len(X), np.int64)
        active = tr[node, 0] == 0
        while active.any():
            idx = np.nonzero(active)[0]
            nd = tr[node[idx]]
            x = X[idx, nd[:, 1].astype(np.int64)]
            go_left = np.where(np.isnan(x), nd[:, 5] == 1, x <= nd[:, 2])
            node[idx] = np.where(go_left, nd[:, 3], nd[:, 4]).astype(np.int64)
            active = tr[node, 0] == 0
        raw += tr[node, 6]
    return 1.0 / (1.0 + np.exp(-raw))


def predict_exported(model, X):
    """Score hits with an exported model (same arithmetic as the Java driver). Returns the last-stage probability."""
    X = np.asarray(X, np.float64)
    p = None
    for stage in model["stages"]:
        p = _predict_stage(stage, X)
        X = np.column_stack([X, p])
    return p


# ----------------------------------------------------------------------------------------------------------------
# Track-level evaluation (MC truth)
# ----------------------------------------------------------------------------------------------------------------
def _tp_masks(d, keep=None):
    """Layer mask of the hits of each truth particle in d['tp'] (only hits with keep True, if given)."""
    sel = d["prim"] >= 0
    keys = d["event"][sel] * 1000 + d["prim"][sel]
    k, m = group_masks(keys, d["X"][sel, 0], None if keep is None else keep[sel])
    key_tp = d["tp"]["event"] * 1000 + d["tp"]["tp_idx"]
    pos = np.clip(np.searchsorted(k, key_tp), 0, max(len(k) - 1, 0))
    has = (k[pos] == key_tp) if len(k) else np.zeros(len(key_tp), bool)
    return np.where(has, m[pos], 0)


def track_metrics(d, keep):
    """Effect of keeping only the hits with keep == True, for a collected MC test sample d (needs per_particle).

    Reference particles: found by the KF and findable (emulator) with all their hits. Returns the fraction of them
    that are no longer findable (all, and in the ECal acceptance), and hit-level fractions."""
    tp = d["tp"]
    top = tp["tp_top"].astype(bool)
    if "_base" not in d:
        d["_base"] = findable(_tp_masks(d), top) & tp["tp_found"].astype(bool)
    base, acc = d["_base"], tp["tp_acc"].astype(bool)
    F = findable(_tp_masks(d, keep), top)
    code, y = d["code"], d["is_mc"]
    return {"lost": (base & ~F).sum() / base.sum(), "lost_acc": (base & acc & ~F).sum() / (base & acc).sum(),
            "good_kept": keep[code == 2].mean(), "wrong_removed": 1 - keep[code == 3].mean(),
            "bkg_kept": keep[y == 0].mean(), "sig_kept": keep[y == 1].mean(),
            "fake_removed": 1 - keep[code == 1].mean()}


def scan(d, score, quantiles=None):
    """track_metrics for a series of thresholds on `score` (thresholds = quantiles of the signal-hit scores)."""
    import pandas as pd
    if quantiles is None:
        quantiles = np.r_[0.0, np.geomspace(2e-5, 0.08, 24)]
    thr = np.quantile(score[d["is_mc"] == 1], quantiles)
    rows = []
    for q, t in zip(quantiles, thr):
        r = track_metrics(d, score >= t)
        r.update(threshold=t, sig_loss_target=q)
        rows.append(r)
    return pd.DataFrame(rows)


def scan_time(d, widths):
    """track_metrics for a simple hit-time window |t| < w."""
    import pandas as pd
    t = np.abs(d["X"][:, FEATURES.index("t")])
    return pd.DataFrame([dict(track_metrics(d, t < w), threshold=w) for w in widths])


# ----------------------------------------------------------------------------------------------------------------
# Data evaluation (no truth): golden tracks as tag
# ----------------------------------------------------------------------------------------------------------------
def golden_metrics(d, keep):
    """For a collected physics-trigger data sample: fraction of golden-track hits kept, and fraction of golden tracks
    (findable with all their hits) that stay findable with the kept hits."""
    sel = d["golden_trk"] >= 0
    keys = d["event"][sel] * 100 + d["golden_trk"][sel]
    k, m0 = group_masks(keys, d["X"][sel, 0])
    _, m1 = group_masks(keys, d["X"][sel, 0], keep[sel])
    _, inv = np.unique(keys, return_inverse=True)
    top = np.zeros(len(k), bool)
    top[inv] = d["X"][sel, 1] > 0.5
    f0 = findable(m0, top)
    f1 = findable(m1, top)
    return {"golden_hits_kept": keep[sel].mean(), "golden_tracks_lost": (f0 & ~f1).sum() / max(f0.sum(), 1),
            "n_golden_tracks": int(f0.sum())}


def reweight(clf_X_target, clf_X_source, X_apply, features=None, clip=(0.1, 10.0), seed=1, max_n=2_000_000):
    """Density-ratio weights w(x) = p_target(x) / p_source(x) from a classifier trained to separate target (e.g. data)
    from source (e.g. MC) control samples, evaluated on X_apply. `features` = column indices used (default all).
    Weights are normalised to mean 1 on X_apply and clipped."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    cols = list(range(clf_X_target.shape[1])) if features is None else list(features)
    n = min(len(clf_X_target), len(clf_X_source), max_n)
    a = clf_X_target[rng.choice(len(clf_X_target), n, replace=False)][:, cols]
    b = clf_X_source[rng.choice(len(clf_X_source), n, replace=False)][:, cols]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=500,
                                         early_stopping=True, random_state=seed)
    clf.fit(np.vstack([a, b]), np.r_[np.ones(n), np.zeros(n)])
    p = np.clip(clf.predict_proba(X_apply[:, cols])[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p)
    w = np.clip(w / w.mean(), *clip)
    return w / w.mean(), clf
