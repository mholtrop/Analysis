# Handoff: SVT hits, KF tracking efficiency, and hit filtering (HPS 2021)

*Written 2026-11-05 at the end of step 1. Load this document into a new session to continue the study without any
other context. All paths are absolute or relative to the git repository root `/net/home/maurik/Analysis`.*

---

## 0. TL;DR for the next agent

- **Goal of the overall study (user: Maurik Holtrop, UNH, HPS collaboration):** replicate Zhaozhong Shi's 2025
  tracking-efficiency studies for the 2021 HPS run, relate tracking efficiency to the hits in the SVT, and then
  **develop and evaluate SVT hit filtering** (removing background hits before/while tracking) to improve track
  purity, momentum resolution and efficiency.
- **Step 1 is done:** notebook `SVT/Tracking_Efficiency_SVT_Hits.ipynb` (executed, outputs saved), using the reusable
  Python package module `Analysis/svt_tracking.py` + C++ helpers `Analysis/svt_tracking.h`.
- **Nothing has been committed to git yet.** New/modified files: `Analysis/svt_tracking.py`, `Analysis/svt_tracking.h`,
  `Analysis/__init__.py` (modified), `README.md` (modified), `SVT/Tracking_Efficiency_SVT_Hits.ipynb`, this file.
  Only commit if the user asks.
- **Known required fix before using the package on real data:** the package hard-codes that tracks use
  `svt_hit_type == 1` hits. That is true for the MC samples but **in real data (run 14229) tracks reference
  `svt_hit_type == 0`** (pass5_v9 data), while the **pass5_v10 data reference type 1** — so the type must be
  auto-detected per sample, not hard-coded. See §5.2 and §11, task A.
- **More data is available than step 1 used** (§4.2): MiniDSTs with **SVT raw hits** (`minidst_plus`) for the same
  overlay events, a 2.1M-event pure rad sample on `/fast`, WAB and A' overlay samples, a **pass6** tritrig-overlay
  sample, and 64M unskimmed data events from 14 runs with SVT hits.
- **Recommended next step:** quantify, per layer, how many *wrong* hits on truth-matched tracks a pre-tracking
  hit-time window would remove, and at what cost in correct hits (no re-reconstruction needed). See §11.

---

## 1. Background: the physics question and prior work

HPS (Heavy Photon Search, JLab Hall B) has a 14-layer silicon vertex tracker (SVT) followed by a PbWO4 ECal. In the
2021 run (3.74 GeV and 1.92 GeV beam) the SVT occupancy is high, mostly from beam background (X-rays, scattered beam)
concentrated in the inner layers. The Kalman-filter (KF) track finder in hps-java picks up background hits, which
lowers track purity, degrades momentum/vertex resolution and creates background/fake tracks.

The prior work is summarised in
`SVT/Hits_Tracking_efficiency/svt_hits_tracking_efficiency.tex` (and `.pdf`), compiled from presentations in
`SVT/Hits_Tracking_efficiency/Zhaozhong_Presentations/` and `SVT/Hits_Tracking_efficiency/Norman Presentations/`.
**Read that .tex file first** if you need the detailed prior numbers. Key points from it:

- **Zhaozhong Shi (2025), MC:** track–cluster matching efficiency ε(N_hits, last layer) rises steeply with hits on
  track (pure tritrig: 0.10 at 5 hits → 0.98 at 13–14; with beam background: ~0 at 5–7 → 0.96 at 14). Cause: the
  track extrapolation to the ECal degrades with fewer hits, not the ECal clustering. He proposed (not optimised) a
  non-rectangular cut in the (N_hits, last layer) plane instead of N_hits ≥ 10.
- **Occupancy and KF seeding time window** (`kPar.mxTdif` in `KalmanPatRecHPS`/`KalmanParams.java`, nominal ≈ 40 ns):
  his efficiency (core of Δp/p, N_hits > 8, χ² < 20, p > 0.4 GeV) fell from ≈0.85 (20 hits/event) to ≈0.40
  (200 hits/event) at 40 ns; 10 ns or 5 ns windows keep it ≈0.8–0.9 and cut the fake rate. Typical 2021 data have
  200–300 SVT hits/event (his count).
- **Per-layer (MC, positrons):** wrong hits concentrated in L1–L2 (8.6 %/6.9 % at 40 ns), missing hits at the ends
  (L9 dead-sensor spike, L13–L14 ≈10 %). Tighter time windows trade purity for missing hits.
- **Data yields:** 10 ns window nearly doubles 2-prong trident yield vs 40 ns; removing L1–L2 entirely is "not that bad".
- **Norman Graf:** hit count and track quality are the same problem (momentum RMS by hit count); strip-clustering
  time window 8 → 20 ns gains two-strip clusters with no fakes; proposed (never done) a calorimeter-tagged
  tag-and-probe hit-efficiency measurement in data.
- **Gaps listed in the summary:** no per-layer hit efficiency in data; track finding/extrapolation/matching confounded;
  MC hit-time distributions don't describe data; 2D (N_hits, last-layer) map not exploited; no run dependence;
  dead sensors lumped per layer; KF vs GBL not revisited.

---

## 2. Computing environment (verified)

- Host: linux (AlmaLinux 9 family), 64 cores, ~124 GB RAM. Shared filesystem `/net/...`.
- Python: `/net/data/endeavour2/Tools/miniconda3/envs/T/bin/python3` (3.12). Available: ROOT 6.41.01
  (`/net/data/endeavour2/Tools/root-master-gcc11.5.0`), numpy 2.3.4, matplotlib 3.10.9, scipy 1.16.3, pandas 2.3.3,
  nbformat, jupyter/nbconvert 7.16.6 (kernels: `python3`, `root`).
- **Not available:** uproot, awkward, hist, mplhep. Use ROOT RDataFrame (the user's existing style).
- Shell has `noclobber` set: `cat > existing_file` fails. Use the Write tool or `>|`.
- Execute notebooks headless:
  `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 <nb>.ipynb`
- hps-java source (for checking conventions): `/net/home/erbaz/hps-java` (e.g.
  `tracking/src/main/java/org/hps/recon/tracking/kalman/KalmanInterface.java`, `KalmanParams.java`).
- MiniDST writer source: `/net/home/maurik/hps-analysis/MiniDst/LcioReader/LcioReader.cxx` (v1.6.0 used by the MC).
- Older helper code in the repo: `Python/root_helpers.py`, `Python/Utility_Functions.C`, `Analysis/ecal.py`
  (ECal fiducial-cut helpers, class `ecal`).

---

## 3. Repository layout (relevant parts)

```
/net/home/maurik/Analysis/                 git repo root (branch main)
├── Analysis/                              Python package "Analysis"
│   ├── __init__.py                        exports ecal (class) and svt_tracking (module)
│   ├── ecal.py                            pre-existing ECal helpers
│   ├── svt_tracking.py                    NEW: study package (Python side)
│   └── svt_tracking.h                     NEW: C++ helpers (namespace svt), declared via ROOT interpreter
├── README.md                              mentions the new module
└── SVT/
    ├── Tracking_Efficiency_SVT_Hits.ipynb NEW: step-1 notebook (executed)
    ├── HANDOFF_Tracking_Efficiency_and_Hit_Filtering.md   this document
    ├── Hits_Tracking_efficiency/          prior-work summary (.tex/.pdf) + presentations
    ├── Hit_Filtering/                     empty directory, intended for the next steps
    ├── FEE.ipynb, FEE2021.ipynb           older user notebooks on FEE data
```

Notebooks import the package with `sys.path.insert(0, "..")` then `from Analysis import svt_tracking as st`.

---

## 4. Data

### 4.1 Samples used in step 1 (MC with full truth)

Both in `/net/data/endeavour2/HPS/physrun2021/home_made_mc/`, each with a detailed `MC_INFO.md` (read it).

| key | directory | content | files | MiniDST events | size |
|---|---|---|---|---|---|
| `pure` | `tritrig_pure_run14272/minidst/tritrig_3pt74_minidst_*.root` | 3.74 GeV tritrig (MadGraph, all QED trident diagrams), detector `HPS_Run2021Pass1_v9_14272`, readout `PhysicsRun2021TrigMultiSingles.lcsim`, **no background** | 474 | 1,009,136 | 4.5 GB |
| `overlay` | `tritrig_pulser_beamspot_run14272/minidst/tritrig_pulser_3pt74_minidst_*.root` | same physics, beam spot σx=0.055, σy=0.045 mm, **real 2021 pulser-trigger data overlaid** (Overlay.lcsim + `PhysicsRun2021TrigSinglesWithPulserDataMerging.lcsim`) | 601 | 993,781 | 12 GB |

Both: recon steering `PhysicsRun2021MCRecon_KF_WithSpacing_pass5_physics_HitSmear_HitKill_14272.lcsim` (KF tracking
with SVT hit smearing + hit killing tuned to run 14272), local SQLite conditions DB, MiniDST made with
`make_mini_dst -a` (all switches; SVT hits with truth on, SVT raw hits OFF, GBL tracks off). Produced by erbaz
(Erbaz Khan) in Oct 2026; slightly different hps-java versions between the two (pure: commit 473c732da;
overlay: 4d47d0383) — keep this in mind when comparing them.

The overlay directory also keeps every intermediate step (`lhe/ conv/ mom/ rot/ sim/ filt/ overlay/ spaced/
readout/ recon/ minidst/`); the pure one keeps `recon/` and `minidst/`. **This means reconstruction can be re-run
from `readout/*.slcio` (overlay) with a modified hit filter in hps-java**, and MiniDSTs remade (e.g. with SVT raw
hits enabled to get amplitudes). Reproduction configs are in each `_config/`.

### 4.2 Other data on disk (full survey 2026-11-05)

Two trees: `E = /net/data/endeavour2/HPS/physrun2021` and `F = /fast/HPS/physrun2021` (fast local disk;
`/fast/HPS/physrun2019` only has 2019 FEE MiniDSTs without SVT hits). Numbers below were measured on the first 3 files of
each sample: hits/event = `svt_hit_type==1` count (type 0 is an identical copy); "trk hit type" = which `svt_hit_type`
the `track_svt_hits` indices point to; "MC frac" = fraction of SVT hits with non-empty `svt_hit_mc_part_idx`.
All MiniDSTs have tree `MiniDST`.

**(a) MC with per-hit truth (`svt_hit_mc_part_idx`) — usable directly with `st.load()`:**

| path | content | files × ev/file | hits/ev | MC frac | trk/ev, ⟨nhit⟩ | use for |
|---|---|---|---|---|---|---|
| `E/home_made_mc/tritrig_pure_run14272/minidst` | step-1 `pure` | 474 × 2.1k | 19 | 1.00 | | baseline |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/minidst` | step-1 `overlay` | 601 × 1.6k | 139 | 0.14 | | main sample |
| **`E/home_made_mc/*/minidst_plus`** (tritrig_pure 474, tritrig_pulser 467 files) | **same events as above + `svt_raw_hit_*`** (layer, module, strip, adc[6 samples], t0, t0_err, amp, amp_err, chi2, fit_no) + ECal/hodo raw | as above | 140 (≈300 raw hits/ev) | 0.14 | 2.7, 10.4 | **raw-hit (pre-clustering) filtering: amplitude, fit χ², t0 error** |
| `E/home_made_mc/wab_pulser_run14272/minidst` | WAB + pulser overlay (MC_INFO.md) | 720 × 260 (≈190k) | 104–140 | 0.35 | 2.3, 10.2 | photon-conversion e± efficiency; second physics sample |
| `E/home_made_mc/ap_pulser_lheuniform_run14272/m{30..200}MeV/minidst` | A' signal (uniform decay length) + pulser overlay, 13 masses | ~70 × 7k per mass | ~140 | | 1.6, 10.5 | **displaced-vertex signal**: efficiency vs decay length/L1 miss |
| `E/home_made_mc/ap_pure_lheuniform_run14272/m{50,100,150,200}MeV/minidst` | same, no overlay | ~170–180 × 2.8k | 21 | 1.00 | | pure reference for A' |
| `F/pass5_v9_mc_rad_readout_recon/minidst` | **rad tridents, pure** (no overlay), same recon as `pure` (MC_INFO.md) | 2000 × 1.06k = 2.1M | 24 | 1.00 | 1.7, 11.8 | 2× more pure statistics; rad kinematics |
| `E/pass6_v10/mc/tritrig_pulser/mdst` | **pass6 (v10-dev, "lowp") recon** of tritrig + pulser (M. Graham SLAC production, hps-java pass6dev steering) | 625 × 1.8k = 1.1M | 137 | 0.14 | **4.7, 8.65** | **compare pass5 vs pass6 tracking** (more, shorter tracks: low-p tracking) |
| `E/pass6_v10/mc/tritrig_3track/mdst` (+ `tuple/` hpstr "cascade" ntuples) | pass6 tritrig, 3-track selection | 125 × 1.8k | 138 | 0.14 | 4.7, 8.7 | |
| `E/pass6_v10/mc/ap_pulser/120MeV/mdst` | pass6 A' 120 MeV | 10 × 15k | 141 | 0.17 | 5.0, 8.5 | |
| `E/pass6_v10/mc/{moller_pulser,new_moller_test/*}/mdst` | pass6 Møller MC | 1–3 files | | | | |
| `E/3pt74_MC/ap_pulser_truth_matched_mdst`, `ap_nooverlay/100MEV_mdst`, `E/3pt74_latest_ML/tritrig_pure/mdst` | older A'/tritrig productions | 200 / 194 / 125 | | | | older software, lower priority |
| `E/MC_test/run_beam/minidst`, `run_beam_geant4/minidst` | beam-on-target background MC test | 62 × 10k | **0.8** | 0 | 0 | **not useful** (essentially empty) |

Caveat found: overlay files can lack the pulser SVT data for many events (e.g. `ap_pulser…m105MeV/minidst_0001.root`:
78 % of events have < 30 hits; the other files checked have ≈140). Before using an overlay sample, check the
per-file mean occupancy and drop deficient files (step 1 saw the same effect in its lowest occupancy bin).

**(b) MC with SVT hits but no per-hit truth** (older MiniDST; tracks only via `track_particle`/ECal truth):
`E/1pt92_MC/{tritrig,tritrig_singles,wab}` (1.92 GeV, 100 files each), `E/3pt74_MC/{tritrig_pulser_newmdst,
wab_pulser,rad_pulser,bh_pulser,ap_pulser,ap_prompt_pulser,ap_pulser_mdst_new}/mdst`,
`E/MC_test/minidst/tritrig_pulser/pass5_v9/…_hitSmearKill` (751 files), `E/moller_mc`. Low priority.

**(c) Real data with SVT hits** (no truth; use for occupancy, hit-time and track-time validation, data/MC):

| path | runs | files × ev/file | hits/ev | trk hit type | notes |
|---|---|---|---|---|---|
| `E/hps_014229_noskim_mdst` | 14229 (3.74 GeV) | 45 × 145k = 6.5M | 136 | **0** | pass5_v9, unskimmed |
| **`E/5percent/pass5_v9_unskimmed`** | **14 runs**: 14185, 14201, 14213, 14229, 14256, 14268, 14282, 14288, 14297, 14307, 14314, 14320, 14328, 14337 (6–50 files each) | 410 × 156k = 64M | 122 (run-dependent) | **0** | **run dependence of occupancy / efficiency proxies** (gap in prior work) |
| `F/pass5_v10_test` | 14185 (24 files) | 24 × 156k | 122 | **1** | pass5_v10 recon of the same run — **v9 vs v10 comparison** on identical events |
| `E/1pt92_unskimmed_data` | 14629 (1.92 GeV) | 51 × 173k | 73 | **0** | lower beam energy, lower occupancy |

**(d) Data/MC without SVT hit branches** (only tracks + `track_svt_hits` indices): `E/FEE`, `E/5percent/{physics,
physics_newMDST,physics_trackZ0Test,hps_14231*}`, `E/3pt74_latest_ML/data`, `E/pass6_v10/data/{3pt74,1pt92}`,
`E/mollers_newMDST`, all of `F/5percent`, `F/new_dst`. The matching `slcio/` directories of the pass6 data are
**v0/Møller skims whose LCIO files contain no SVT hit collections** either (only tracks, ECal, vertices), so SVT hits
cannot be recovered by re-running `make_mini_dst`. Useful only for track-level quantities (n hits, layers, time).

**(e) LCIO for re-reconstruction with a hit filter** (needed for §11 tasks that change tracking):
- `E/home_made_mc/tritrig_pulser_beamspot_run14272/readout/*.slcio` (601) — `SVTRawTrackerHits`, `SVTTrueHitRelations`,
  `MCParticle`, ECal/hodo readout. Also `overlay/`, `spaced/` earlier steps. Same for `wab_pulser` (recon only) and the A' samples.
- `E/pass6_v10/mc/tritrig_pulser_readout/*.slcio` (125 jobs, README.md) — pass6 readout input.
- `F/pass5_v9_mc/rad_readout_slic` (2000 SLIC files) — input of the rad sample.
- Recon LCIO (`…/recon/*.slcio`) keeps `SVTRawTrackerHits`, `SVTFittedRawTrackerHits`, `SVTShapeFitParameters`,
  `StripClusterer_SiTrackerHitStrip1D`, `KalmanFullTracks`, and hps-java's own truth links
  `KalmanFullTracksToMCParticleRelations` / `KalmanFullTracksTruth` (could cross-check our `trk_matched`).
- Tools in `/net/home/maurik/AlmaLinux/bin`: `make_mini_dst`, `dumpevent <file> <n>` (lists collections),
  `lcio_event_counter`, `lcio_split_file`, `lcio_merge_files`.

**Recommended additions to `SAMPLES`** (in `svt_tracking.py`): `overlay_raw` (minidst_plus), `wab_overlay`,
`rad_pure` (/fast), `pass6_overlay`, `ap_overlay_<mass>`, and data samples `data14229`, `data_5pct` (glob over runs),
`data14185_v9` vs `data14185_v10`. Data samples need task A (hit-type auto-detection) first, and code paths that
skip truth columns.

To check a file quickly: open with ROOT, tree name `MiniDST`, and there is a `metadata_json` TObjString with the
MiniDST switches (`use_svt_hits`, `use_svt_hits_truth`, `use_svt_raw_hits`, ...).

---

## 5. MiniDST conventions (all verified in this session)

### 5.1 SVT hits
- `svt_hit_*` vectors hold the 1D strip clusters. LcioReader fills from two collections in order:
  `StripClusterer_SiTrackerHitStrip1D` → `svt_hit_type == 0`, and `RotatedHelicalTrackHits` → `svt_hit_type == 1`.
  **In all files checked (MC and data) the two sets are identical copies** (same count and content per event).
- `svt_hit_layer`: 1..14 (sensor layer; one axial/stereo sensor per layer per half; layers 9–14 have hole and slot
  sensors). Decoded from the raw hit cell ID with `system:6,barrel:3,layer:4,module:12,sensor:1,side:32:-2,strip:12`.
- `svt_hit_module`: **even = top half, odd = bottom half** (0,2 top; 1,3 bottom; verified against hit y).
- `svt_hit_time` in ns; `svt_hit_x/y/z` position in mm (detector frame, y vertical, z along beam).
- `svt_hit_mc_part_idx[i]` (vector<vector<int>>): **indices into the `mc_part_*` arrays** of the MC particles whose
  SimTrackerHits are related to the raw hits of this cluster (a particle can appear more than once per hit).
  `svt_hit_mc_part_edep` parallel energies. **Overlaid pulser-data hits have an empty list** → background label.
- `svt_hit_strip` strips in the cluster; `svt_hit_raw_index/other` empty (raw hits not stored).
- In RDataFrame, nested branches appear as `ROOT::RVec<std::vector<int>>` — C++ helper signatures must use exactly
  that type (not `RVec<RVec<int>>`).

### 5.2 Tracks (KF only in these files; `track_type == 1` for all, `track_n_gbl == 0`)
- `track_svt_hits[i]`: indices into the `svt_hit_*` arrays. **MC: always point to type-1 hits. Real data run 14229:
  point to type-0 hits.** (Measured: MC overlay sample all tracks → type 1; run 14229 data all tracks → type 0.)
  Since both copies are identical, the correct approach is to use whichever type the tracks reference
  (or, per event, the type of `track_svt_hits[0][0]`).
- `track_hit_pattern[i]` (14 ints, from `KalmanInterface.java` ~line 633): **1 = hit on track in that layer,
  −1 = KF site crossed without a hit ("hole"), 0 = no site/sensor crossed.**
- `track_n_hits`, `track_chi2`, `track_ndf` (= n_hits − 5), `track_px/py/pz` (GeV), `track_time` (ns),
  `track_x/y/z_at_ecal`, `track_*_at_target`, `track_particle` (index into `part_*`).
- **Track/hit time offsets differ per sample:** matched tracks peak at **+9.9 ns in `pure`**, at **−0.1 ns in
  `overlay`** (trigger time aligned with the pulser data), and around **+1 ns in data run 14229** (peak bin; data
  RMS is large because of background tracks). Track time resolution of matched tracks: 16–84 % ≈ ±1.7 ns (pure),
  ±2.4 ns (overlay). Signal hit-time 16–84 %: ±3.3 ns (pure), ≈ −3.8/+3.2 ns (overlay).

### 5.3 MC truth
- `mc_part_*` arrays; `mc_part_parents/daughters` are indices. Tritrig events have a parentless **PDG 622 "label"
  particle** (not a physical A′) whose daughters are the generated trident e⁺e⁻e⁻. Primary particles start at
  z = −1.1 mm (target).
- `ecal_hit_mc_parent_id`: **index of the ultimate primary ancestor** (walks up until the parent is PDG 622/623) of the
  particle contributing most energy to the ECal hit. So "particle reached the ECal" = sum of `ecal_hit_energy` over
  hits with `ecal_hit_mc_parent_id == idx`.
- `ecal_cluster_mc_id` etc. also exist (not used).

---

## 6. The package `Analysis/svt_tracking.py` (+ `svt_tracking.h`)

### 6.1 Design
- C++ helpers in `svt_tracking.h` (namespace `svt`), declared once with `R.gInterpreter.Declare('#include ...')`
  by `st.setup()`. All per-event work is C++ inside RDataFrame `Define`s → fast, multithreaded.
- **Lazy booking**: every histogram/ratio is booked first; `st.run(...)` fills everything with `ROOT.RDF.RunGraphs`
  (one event loop per sample, samples in parallel). Full step-1 notebook: ~65 s cold, ~12 s with warm file cache,
  32 threads, both samples (2M events).
- Results are converted to numpy and plotted with matplotlib; efficiency errors are Clopper–Pearson (68.27 %).
- Layers are encoded as **bit masks**: bit (L−1) set ⇔ layer L present. `svt::mask_to_layers()` flattens masks to a
  list of layer numbers for per-layer histograms.

### 6.2 Python API
```python
from Analysis import svt_tracking as st
st.setup(n_threads=32)          # EnableImplicitMT + declare C++ (idempotent)
st.mpl_style()                  # matplotlib rcParams used in the notebook
st.SAMPLES                      # {"pure": Sample, "overlay": Sample}; Sample(key, label, pattern, description, t0)
st.SAMPLES["pure"].files(n)     # sorted file list (first n)
df = st.load("overlay", n_files=None, min_purity=0.5)   # RNode with all derived columns (6.3)

h  = st.book_hist(df, expr, bins, where=None, weight=None)     # Hist: .counts .edges .centers .underflow_overflow
h2 = st.book_hist2d(df, xexpr, yexpr, xbins, ybins, where=None) # Hist2D: .counts (nx, ny) .xedges .yedges
r  = st.book_ratio(df, expr, passed, bins, where=None)          # Ratio: .eff .err(2xn) .passed .total .centers .integral()
r  = st.book_ratio_of(df, num_expr, den_expr, bins)             # Ratio of two different expressions (num ⊂ den)
r2 = st.book_ratio2d(df, xexpr, yexpr, passed, xbins, ybins, where=None)  # Ratio2D: .eff .total
st.run(anything_nested)         # lists/dicts of booked objects → single RunGraphs
```
- `bins` is either `(n, lo, hi)` or an array of edges. `st.int_bins(lo, hi)` gives edges centred on integers;
  `st.layer_bins()` = `int_bins(1, 14)`; `st.LAYERS = 1..14`.
- `expr`, `passed`, `where` are RDataFrame C++ expressions on **per-entry RVec columns of equal length** (e.g. all
  `tp_*`, or all `trk_*`); internally the booked column is `expr[where]`. Event-level scalars can be used as expr
  without `where`.
- `Ratio.integral()` returns (eff, err_lo, err_hi) summed over **all bins including under/overflow** — so to get an
  efficiency for a sub-range, put the restriction in `where`, not in the binning (this bit me once).
- Analysis helpers: `st.clopper_pearson(k, n)`, `st.quantile(counts, edges, q)`,
  `st.central_width(hist2d)` → (median, half-width of central 68 %) per x bin, `st.window_fraction(hist, half_widths,
  center=0)` → fraction of entries (incl. under/overflow) inside ±w.
- Plot helpers: `st.plot_hist(ax, hist, label, color, density)`, `st.plot_ratio(ax, ratio, label, color, min_total)`,
  `st.style(ax, xlabel, ylabel, title, legend=True, grid=True)`.
- Colours (fixed by sample identity, validated categorical palette): `st.COLORS = {"pure": "#2a78d6" (blue),
  "overlay": "#eb6834" (orange), "third": "#1baf7a" (aqua), "ink", "muted"}`; 2D maps use `Blues`.

### 6.3 Columns defined by `st.load()`
Event level: `hit_sel` (type-1 mask), `n_svt_hits`, `hit_is_mc`, `n_svt_hits_mc`, `n_svt_hits_bkg`, `n_tracks`.

Per hit (type-1 hits only, original order): `hit_layer`, `hit_time`, `hit_time_rel` (= hit_time − Sample.t0),
`hit_top`, `hit_is_mc` (1 = has MC truth / signal, 0 = overlay background), `hit_on_track`.

Per track (`trk_*`, same order as `track_*`): `trk_tt` (struct), `trk_particle` (majority MC index, −1 if none),
`trk_n_shared`, `trk_nhits`, `trk_purity` (= n_shared / n_hits), `trk_layers` (mask), `trk_good` (mask of layers whose
hit contains the truth particle), `trk_wrong` (= layers & ~good), `trk_first_layer`, `trk_last_layer`,
`trk_n_holes` (count of −1 in hit pattern), `trk_p`, `trk_time`, `trk_time_rel`, `trk_is_primary`,
`trk_matched` (= is_primary && purity ≥ min_purity), `trk_truth_p`, `trk_dp_rel` (= (p − p_true)/p_true;
meaningful only where matched).

Per primary truth particle (`tp_*`): `tp_idx`, `tp_pdg`, `tp_p`, `tp_tanl` (= py/√(px²+pz²); >0 top),
`tp_top`, `tp_layers` (mask of layers with a reconstructed truth hit), `tp_n_layers`, `tp_last_layer`,
`tp_n_svt_hits` (event occupancy copied per particle), `tp_ecal_e` (true ECal energy of its shower, GeV),
`tp_track` (best matched track index or −1), `tp_found`, `tp_n_tracks` (matched tracks; >1 = duplicates),
`tp_trk_nhits`, `tp_missing` (mask: truth layers not covered by a correct hit on the matched track).

Hit-time residuals on tracks with a truth particle: `dt_good`, `dt_wrong` (hit time − track time, for hits that do /
do not contain the track's truth particle).

### 6.4 C++ helpers (namespace `svt` in `svt_tracking.h`)
`kTrackHitType = 1`, `kNLayers = 14`; `popcount/first_layer/last_layer` (int and RVec overloads); `and_not(a,b)`;
`mask_to_layers(masks)`; `take_or(v, idx, fallback)`; `is_in(a, b)`; `primary_charged(pdg, parents, charge)`;
`hit_has(mc_list, ip)`; `truth_layer_mask(parts, hit_type, hit_layer, hit_mc)`; `truth_ecal_energy(parts,
ecal_parent, ecal_e)`; `struct TrackTruth {particle, n_shared, hit_mask, good_mask}` + `track_truth(trk_hits,
hit_layer, hit_mc)`; `match_track(parts, trk_particle, trk_shared, trk_nhits, min_purity)`; `n_matched_tracks(...)`;
`count_in_pattern(pattern, value)`; `hit_is_mc(hit_type, hit_mc)`; `hit_on_track(hit_type, trk_hits)`;
`hit_minus_track_time(trk_hits, trk_time, trk_particle, hit_time, hit_mc, good)`.

### 6.5 Gotchas learned
- The RDataFrame expression tokenizer **rejects `~`** → use `svt::and_not()`.
- Jitted lambdas inside `Define` strings that return via `Map` crashed cling; put logic in header functions instead.
- Nested branch type is `RVec<std::vector<int>>` (see 5.1).
- `Ratio.integral()` includes under/overflow (see 6.2).
- Track times must be compared relative to `Sample.t0` (sample-dependent offset).
- In the overlay, ~4 % of events have almost no background SVT hits (the overlaid pulser event had no SVT data);
  they populate the lowest-occupancy bin. Within them, found fraction of in-acceptance particles is ~0.9, but the
  composition varies strongly file-to-file.

---

## 7. Definitions used in step 1

- **Truth particle**: generated charged trident particle = daughter of the PDG-622 label (`svt::primary_charged`).
- **Truth layers**: layers where a reconstructed (type-1) strip hit lists the particle in `svt_hit_mc_part_idx`.
  Note this is *reconstructed* hits with truth, so MC hit killing/inefficiency is already folded in; there is no
  per-sensor "crossed" truth because `use_mc_scoring` is off.
- **In acceptance** (efficiency denominator): `tp_ecal_e > 0.1` GeV — SVT-independent, analogous to Zhaozhong's
  "truth positrons reaching the ECal".
- **Track–truth match**: majority particle with purity ≥ 0.5 (`MIN_PURITY`). A particle is **found** if it has
  such a track; best = most shared hits.
- **Matched track**: majority particle is a primary trident particle and purity ≥ 0.5.
- **In-time track**: `abs(trk_time_rel) < 6` ns.
- **Missing hit (layer L)**: truth hit in L, but the matched track has no hit from that particle in L
  (denominator: truth layers of found particles) — Zhaozhong's "disagreement/truth".
- **Wrong hit (layer L)**: hit on a matched track that does not contain the track's particle
  (denominator: hits on matched tracks) — "disagreement/recon".
- **Signal vs background hit** (overlay): `hit_is_mc` 1 / 0.

---

## 8. The step-1 notebook `SVT/Tracking_Efficiency_SVT_Hits.ipynb`

Structure (each section: one figure, short markdown interpretation; all booking in one cell, one `st.run(H)`):
0. Intro + samples/definitions (markdown). Setup cell with parameters `N_FILES = None` (set e.g. 20 for quick
   runs), `MIN_PURITY = 0.5`, `ACC = "tp_ecal_e > 0.1"`, `INTIME = "abs(trk_time_rel) < 6"`.
1. **Occupancy**: hits/event per sample; hits per layer, signal vs background (overlay).
2. **Truth view**: efficiency vs number of truth layers, true p, true tanλ, event occupancy; summary table.
3. **Track view**: matched fraction vs hits on track and vs last layer (pure / overlay all / overlay in-time);
   2D map (hits, last layer) for in-time overlay tracks; σ(Δp/p) vs hits; track composition table.
4. **Per layer**: missing and wrong hit fractions; table.
5. **Toward hit filtering**: hit time signal vs background; hit−track time for good vs wrong hits on matched tracks;
   signal/background kept vs hit-time window half-width.
6. Summary table + next steps (markdown).

The notebook was generated programmatically with nbformat (generator script was in a temporary scratch directory and
is not kept); edit the notebook directly from now on.

---

## 9. Results of step 1 (full samples: 1,009,136 pure, 993,781 overlay events)

### 9.1 Occupancy
- SVT strip hits/event: **pure 19.3**, **overlay 135.6** (117.1 background + ≈18 MC). Data run 14229: ≈136
  (same counting) — the overlay reproduces data occupancy well.
- Background per layer (overlay, hits/event): L1 ≈37, L2 ≈27, L3–L4 ≈11–12, L5–L6 ≈8.5, L7–L8 ≈3.5, L9–L14 ≈1–2.
  Signal ≈1.2–1.6 per layer. L1+L2 carry > half of the background.

### 9.2 Efficiency (denominator: trident e± with > 100 MeV in ECal)
| | pure | overlay |
|---|---|---|
| particles in acceptance | 1,184,401 | 1,157,926 |
| efficiency | 0.9481 ± 0.0002 | 0.9275 ± 0.0002 |
| fraction with < 8 truth layers | 0.042 | 0.062 |
| efficiency, ≥ 8 truth layers | 0.9895 | 0.9824 |
| efficiency, \|tanλ\| > 0.015 (outside beam gap) | 0.9905 | 0.9848 |
| found with > 1 track (duplicates) | 0.0005 | 0.0213 |

- vs truth layers (pure / overlay): ≤7 layers ≈0 / up to 0.40 (at 7; background hits complete tracks); 8: 0.745 /
  0.76; 9: 0.93 / 0.895; 10: 0.985 / 0.96; ≥11: ≈0.99–1.0.
- vs tanλ: ≈0.98–1.0 except |tanλ| < 0.015 (beam gap): ≈0.78 at 0.01–0.015, ≈0.05–0.1 below.
- vs p: drop below 0.6 GeV (0.3 at 0.3 GeV); overlay dips to ≈0.86 at 1.5–2.5 GeV vs ≈0.93 pure (**unexplained**).
- vs occupancy: **flat** ≈0.92–0.93 from 50 to 350 hits/event in the overlay (contrast with Zhaozhong's strong drop —
  his efficiency includes track-quality cuts; see §10).

### 9.3 Track view
| | pure | overlay |
|---|---|---|
| tracks / event | 1.35 | 2.62 |
| matched to trident e± | 1.000 | 0.509 |
| containing no MC hit at all | 0.000 | 0.415 |
| in time (\|t_rel\| < 6 ns) | 0.986 | 0.558 |
| matched tracks that are in time | 0.986 | 0.931 |
| matched fraction among in-time tracks | 1.000 | 0.850 |

Matched fraction vs hits on track 5…14 (overlay):
all tracks 0.21, 0.23, 0.21, 0.23, 0.35, 0.51, 0.61, 0.73, 0.62, 0.88;
in-time 0.49, 0.54, 0.52, 0.59, 0.73, 0.85, 0.91, 0.95, 0.90, 0.98. (Pure: 1.00 everywhere.)

Matched fraction vs last layer 6…14 (overlay):
all 0.16, 0.21, 0.21, 0.23, 0.33, 0.57, 0.76, 0.46, 0.57;
in-time 0.48, 0.55, 0.56, 0.52, 0.73, 0.84, 0.95, 0.82, 0.89.
Note the dips at 13 hits and at last layer 13 — these persist in-time; not yet understood (real background
tracks with full hit patterns? hole/slot geometry?).

2D map (in-time, overlay): best on the diagonal N_hits = last layer (tracks starting at L1 with no gaps); along
last layer 14 the matched fraction is 0.98 (14 hits), 0.90 (13), 0.83 (12), 0.75 (11), 0.70 (10), 0.71 (9),
0.65 (8), 0.61 (7), 0.60 (6).

Momentum resolution σ(Δp/p) (central-68 % half-width) of matched tracks vs hits 5…14:
pure 0.26, 0.39, 0.24, 0.10, 0.10, 0.054, 0.037, 0.025, 0.024, 0.022;
overlay 0.37, 0.42, 0.375, 0.29, 0.22, 0.125, 0.048, 0.030, 0.028, 0.024.

### 9.4 Per-layer missing and wrong hit fractions
| layer | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missing, pure | .0054 | .0063 | .0056 | .0064 | .0062 | .0081 | .0053 | .0055 | .0129 | .0169 | .0620 | .0656 | .0929 | .0793 |
| missing, overlay | .2063 | .1864 | .1098 | .1016 | .0536 | .0568 | .0361 | .0359 | .0336 | .0375 | .0907 | .0876 | .1233 | .1073 |
| wrong, pure | .0058 | .0053 | .0020 | .0022 | .0009 | .0010 | .0010 | .0011 | .0004 | .0005 | .0004 | .0004 | .0004 | .0003 |
| wrong, overlay | .2347 | .2014 | .0839 | .0829 | .0442 | .0460 | .0302 | .0321 | .0042 | .0061 | .0036 | .0036 | .0033 | .0039 |

- In L1–L8 (overlay) missing ≈ wrong → the KF takes a background hit **instead of** the particle's own hit. This is
  exactly what hit filtering should fix.
- L11–L14 missing 6–9 % **even in pure MC** — not background-driven; cause unknown (candidates: hole/slot sensor
  overlap, KF outlier rejection at track end, hit killing tuning).

### 9.5 Hit times (overlay, relative to Sample.t0)
- Signal hits: narrow peak (16–84 % ≈ ±3.5 ns) with a low plateau (~1e-4 relative) out to −75 ns.
- Background hits: ≈uniform over −75…+60 ns, with an edge structure near +70 ns.
- Hits on matched tracks, hit − track time: good hits sharply peaked; wrong hits broad over ±40 ns (the KF window).
- Window ±Δt around signal time, fraction kept (signal / background): ±5 ns 80.9 % / 9.4 %; ±10 ns 95.5 % / 18.9 %
  (117 → 22 background hits/event).
- Caveat: MC timing is idealised; data has worse resolution, per-sensor t0 calibration, and known data/MC time
  disagreement (Zhaozhong 2025-07-29 and 08-19 talks).

---

## 10. Interpretation and open questions

1. Finding efficiency is governed by the number of layers hit (threshold 8–10) and the beam gap; it is only weakly
   reduced by background (−0.6 % on the plateau) and flat in occupancy. Zhaozhong's strong occupancy dependence most
   likely reflects track *quality* (his efficiency required the core of Δp/p, N_hits > 8, χ² < 20) and/or an older
   reconstruction / different `mxTdif`. **Open:** reproduce his definition with this package to confirm (book
   `tp_found && |dp_rel| core && nhits > 8 && chi2 < 20` vs occupancy). The KF `mxTdif` value used in these
   recon steerings was not checked (code default `mxTdif = 30` ns in `KalmanParams.java`; steering may override).
2. Background mainly degrades quality: wrong hits in L1–L4 and many (mostly out-of-time) background tracks.
3. Track purity depends jointly on N_hits and last layer; the no-gap diagonal is the clean region → motivates a 2D
   track-quality cut (or a cut on number of holes / missing layers).
4. Open: overlay efficiency dip at 1.5–2.5 GeV; matched-fraction dips at 13 hits / last layer 13; L11–L14 missing
   hits in pure MC; whether my missing/wrong fractions (much larger in L1–L2 than Zhaozhong's 3 %/8.6 %) are due to
   sample/overlay/software differences.

---

## 11. Next steps (prioritised, with implementation hints)

**A. Make the package data-safe (small, do first if touching data).**
In data, tracks reference `svt_hit_type == 0`. Replace the hard-coded `kTrackHitType` by the type the tracks use:
e.g. a C++ helper `int track_hit_type(const RVec<std::vector<int>>& trk_hits, const RVecI& type, int fallback)` per
event (or detect once per sample from the first event with tracks and store it in `Sample`, e.g.
`Sample.hit_type`), and pass it to `hit_sel`, `truth_layer_mask`, `hit_is_mc`, `hit_on_track`. Truth-dependent
columns must be skipped for data (no `svt_hit_mc_part_idx`, `mc_part_*`): add a `has_truth` flag to `Sample` and
define only the non-truth columns otherwise. Add the data samples (`hps_014229_noskim_mdst`, `1pt92_unskimmed_data`)
to `SAMPLES` with their own `t0` (measure the matched/in-time track time peak; ≈ +1 ns for 14229).

**B. Quantify what a pre-tracking hit-time filter would remove (no re-reconstruction).** New notebook in
`SVT/Hit_Filtering/` (user created that directory for this). Per layer (and top/bottom):
- distribution of `hit_time_rel` for signal vs background hits per layer → per-layer signal efficiency vs
  background rejection curves;
- for **wrong hits on matched tracks**, fraction outside ±Δt (= would be removed) vs Δt, per layer; for **correct
  hits on matched tracks**, fraction lost. New C++ helper: per-hit lists of (layer, time_rel) for good and wrong
  on-track hits (extend `hit_minus_track_time` to return layer too, or return a struct).
- also check whether the hits that *would have been* correct (the missing truth hits in L1–L4) are in time → if
  yes, filtering should recover them.
- optimise a layer-dependent window (tight in L1–L4 where background dominates, loose in L9–L14).

**C. Reproduce Zhaozhong's definitions** with this package (efficiency with Δp/p core, N_hits > 8, χ² < 20 vs
occupancy; positron-only per-layer fractions) to reconcile §10.1.

**D. Validate hit times in data.** Using run 14229 (3.74 GeV) unskimmed MiniDSTs (have SVT hits): hit time minus
track time for hits on good tracks (e.g. FEE-like tracks, p ≈ 3.5–3.9 GeV, 13–14 hits, in time), per layer and per
sensor; compare widths with MC. Also the global hit-time distribution per layer (in-time peak over flat background).
Note the FEE skim MiniDSTs (`physrun2021/FEE/`) do **not** contain `svt_hit_*`; an FEE selection on the unskimmed
14229 files or remaking FEE MiniDSTs with SVT hits is needed.

**E. Additional hit variables for filtering.** Cluster amplitude/charge, number of strips, fit chi2 are not stored
(`use_svt_raw_hits` off; `svt_hit_edep` exists — check whether it is filled meaningfully in MC and data). Remake a
subset of MiniDSTs from `recon/*.slcio` with raw hits enabled (`make_mini_dst` switch for SVT raw hits; see
LcioReader `use_svt_raw_hits`) to build a multivariate/ML hit classifier (signal vs background labels from the overlay
truth).

**F. Test in reconstruction.** Implement the chosen filter in hps-java (hit selection before KF pattern recognition,
or the strip-clustering/time window, or `mxTdif`/`seedTimeSpread` in `KalmanParams`), re-run recon on the overlay
`readout/` (and pure `recon/`→ needs readout; pure only keeps recon) with the same steering, remake MiniDSTs, and
rerun the step-1 notebook to measure efficiency, purity, per-layer wrong/missing and resolution. Compare with
`mxTdif` variations (Zhaozhong's 5/10/40 ns).

**G. Further studies suggested by the gaps:** track-quality cut in (N_hits, last layer) or (N_hits, holes) optimised
for signal vs background tracks; per-sensor (not per-layer) treatment for L9–L14 hole/slot sensors and dead regions;
run dependence in data; 1.92 GeV samples.

---

## 12. User preferences (from the requests in this session)

- Reusable code goes into the Python package `Analysis` (repo root), not into notebooks.
- Notebooks must be **well written, understandable, efficient, uncluttered**, with **well-labelled plots**; short
  markdown interpretation per figure; one booking cell + one event loop pattern worked well.
- Uses ROOT RDataFrame + Python; matplotlib for figures. Keep consistent colours per sample.
- Don't commit or push unless asked. Note the user's git identity is configured (mholtrop).

---

## 13. Quick-start snippets

Minimal reproduction of a step-1 quantity:
```python
import sys; sys.path.insert(0, "/net/home/maurik/Analysis")
from Analysis import svt_tracking as st
import matplotlib.pyplot as plt
st.setup(32); st.mpl_style()
H = {}
for k in st.SAMPLES:
    df = st.load(k, n_files=20)                         # quick look; None = all files
    H[k] = dict(eff=st.book_ratio(df, "tp_n_layers", "tp_found", st.int_bins(0, 14), where="tp_ecal_e > 0.1"),
                wrong=st.book_ratio_of(df, "svt::mask_to_layers(trk_wrong[trk_matched])",
                                           "svt::mask_to_layers(trk_layers[trk_matched])", st.layer_bins()))
st.run(H)
fig, ax = plt.subplots()
for k in H:
    st.plot_ratio(ax, H[k]["eff"], label=st.SAMPLES[k].label, color=st.COLORS[k])
st.style(ax, "layers with a truth hit", "tracking efficiency")
```

Adding a new per-event C++ quantity: write an `inline` function in `Analysis/svt_tracking.h` (namespace `svt`,
argument types `const ROOT::RVec<int>&`, `const ROOT::RVec<std::vector<int>>&`, ...), add a `(name, expr)` entry to
the `defs` list in `load()`, document it in the module docstring. Restart the kernel after editing the header
(the interpreter declares it once).

Sanity checks that should still hold (20 files per sample): overlay ≈138 hits/event, ≈18 MC hits/event, 2.65
tracks/event; pure matched fraction 1.00; overlay efficiency (ACC) ≈0.93.

---

## 14. Critical review of step 1 (2026-11-06; checks on 30 files per sample)

**Truth definition: sound, with caveats.**
- `svt_hit_mc_part_idx` lists the particle of each SimTrackerHit related to any strip of the cluster (LcioReader.cxx ~l.1070).
  We found no `-1` indices, only 0.35 % of hits shared by more than one particle, and only 0.13 % of the "good" hits where the
  matched particle deposits less than half the charge. Delta rays are negligible (0.2 % of wrong hits).
- Wrong hits on matched tracks come from the other trident particles in pure MC (95 %) and from overlaid data hits in the overlay (97 %).
- **"Background" in the overlay has no truth.** The 41.5 % of tracks with no MC hit are a mix of real particles from the pulser
  event and genuine combinatorial fakes. Step 1 cannot separate the two, so they must not be called fakes.
- Truth layers (`tp_layers`) are built from **reconstructed** clusters, so the "truth" already contains the SVT hit inefficiency,
  dead channels and hit killing. Missing-hit fractions are therefore lower limits. True sensor crossings are in the LCIO
  `TrackerHits` (SimTrackerHit) collection, which is not in the MiniDST.

**Efficiency definition: incomplete.**
- **Denominator.**
  - It contains 3 charged primaries per event: the e⁺, the pair e⁻ **and the recoil e⁻**. These are not separated.
  - The acceptance is ECal-based. 0.7 % (pure) and 2.4 % (overlay) of accepted particles have no SVT hit; all of these have
    |tanλ| < 0.015, i.e. they pass through the beam gap. Particles with |tanλ| < 0.015 make up about 17 % of the denominator.
  - The ≥ 8-layer acceptance is circular, because it uses reconstructed hits.
- **Numerator.**
  - "Found" means a track with majority purity ≥ 0.5. There is no requirement on χ², Δp/p, track time or ECal match, so it is
    track-*finding* efficiency, not Zhaozhong's track–cluster efficiency.
  - Clones: in the overlay, 2.1 % of particles have two or more matched tracks (0.06 % in pure); these are not reported.
- **Pure vs overlay are not like-for-like.** They differ in beam spot (the overlay has 55 × 45 µm; the pure sample's beam spot was not
  checked), hps-java commit and readout steering. The overlay has more particles in the beam gap with no hits.
  Differences are not purely from background.
- **Other assumptions.** A single t0 per sample and an ad-hoc ±6 ns in-time cut. The signal hit-time resolution comes from
  simulation and has not been validated on data. `kTrackHitType` is hard-coded. `track_type` is not filtered (all tracks are
  type 1 in these files, but not guaranteed in others).

## 15. Step 1b (2026-11-06): tiered efficiency by charge and half

Notebook `SVT/Tracking_Efficiency_by_Charge_and_Half.ipynb` (full samples, ~80 s). Package additions:
- `Sample.hit_type` (None → `detect_hit_type()`) — this was task A, the hit-type fix.
- Columns `tp_pos`, `tp_gap`, `tp_acc`, `tp_trk_dp_rel`, the tier flags (`tp_clean`, `tp_good_p`, `tp_in_time`,
  `tp_ecal_match`) and `tp_tier`, plus `trk_top`, `trk_pos`, `trk_n_wrong`, `trk_cluster`.
- New `load()` arguments: `max_wrong=1`, `dp_cut=0.15`, `t_window=6`.
- Constants and helpers: `TIERS`, `TIER_DOC`, `CATEGORIES`, `CHARGE_COLORS`, `cat_style()`, `tier_efficiency()`.
- C++ helpers `svt::track_cluster` and `svt::cluster_is_true`.

Key results:
- The top half has 37 % more background (67.6 vs 49.4 hits/event, mostly L1–L4), so top tracks are less clean.
- Electrons are worse than positrons even in pure MC: found efficiency ≈ 5 % lower, σ(Δp/p) for bottom e⁻ is 6–9 %,
  and 6–9 % of tracks have no ECal cluster.
- L9 bottom e⁻ show an anomalous wrong-hit rate (13 % pure, 35 % overlay).
- The ECal acceptance covers e⁻ only at 0.8–1.8 GeV; use the SVT acceptance for electrons.

## 16. Review 2 (2026-11-07): top/bottom resolution explained, efficiency vs hits

Full details are in `SVT/Review.md`, "Review 2". The step-1 and step-1b analyses were checked and are correct, and
their numbers reproduce exactly.

**Bottom-electron resolution.** The poor resolution is real.
- The whole **L9 bottom electron-side stereo sensor is dead**: in the 2021 conditions, `svt_hit_module` 1 at L9 has
  0 hits, in MC and in data.
- Stereo layers are even in the top and odd in the bottom, so only bottom tracks lose a bend-plane measurement.
- At an equal number of stereo hits in L9–L14, top and bottom resolutions agree.
- L1 and L2 top also cover less of the area near the beam than the bottom; data show the same.

**Efficiency vs hits.** The step at 8–10 layers is the KF requirement set (`st.KF_REQUIREMENTS`).
- Efficiency depends on the numbers of stereo and axial layers, and on the seed patterns: one sensor per back
  module is poorly seeded.

**Package changes.**
- Stereo/axial columns, `load_hits()` + `DATA["run14229"]`, and `plot_eff_map()`.
- Bug fix: `_select` now wraps `expr` in parentheses.

**Open questions.**
- Electron "found" efficiency is 5 % below positrons in pure MC.
- The overlay efficiency dips at 1.5–2.5 GeV.
- 6–9 % of L11–L14 hits are missing in pure MC.
- Hit filters must be evaluated per half and category, and must not remove the scarce back-layer stereo hits of
  bottom electrons.

## 17. Step 2 (2026-11-07): ML hit filter, data fine-tuning, hps-java driver

Everything is in `SVT/Hit_Filtering/`; start with its `README.md`. The code is in `Analysis/svt_hitfilter.{h,py}`.

**Design**
- Per-hit GBDT (scikit-learn `HistGradientBoostingClassifier`) with 27 features: time, charge, cluster size and
  sensor, plus geometric context: partner sensor, target-pointing axial road in y/(z+1.1), road time, competitors.
- It runs before the KF as `SvtHitFilterDriver`, which replaces `StripClusterer_SiTrackerHitStrip1D` in place.
- The working point uses a KF findability emulator: ≥ 9/8 hits, ≥ 4/3 stereo, ≥ 2 axial, the 20 seed patterns.
  It agrees with the actual KF at 99.95 % in pure MC.

**Key findings**
- A time window is useless at zero track loss. Context features give +7 points of wrong-hit removal; GBDT beats an
  MLP.
- Data-driven labels cannot train the filter: positive-unlabelled (physics vs pulser triggers, TS bit 15) and
  golden-track tags both teach that wrong hits are signal (shown on MC).
- Data are used instead to reweight the MC inputs (control samples: golden-track hits; data pulser vs emulated MC
  pulser) and to set the working point (golden-track loss ≤ 0.3 %).
- The MC-only model would lose 1.5 % of golden tracks in data.
- Stage-2 (post-KF) hit cleaning is much stronger (57 % of the remaining wrong hits at 0.2 % correct-hit loss) but
  needs a KF hook or a GBL refit.

**Reconstruction test** (13 k held-out events, `Recon_Test.ipynb`)
- e⁺ efficiency unchanged; e⁻ about −1 %.
- Positron full chain +3 points; L1/L2 wrong hits 24/20 % → 17/15 %.
- Matched tracks 51 → 61 %; 20 % faster.

**Gotchas**
- Never put the same hits into two persistent LCIO collections: track→hit references get corrupted (the unfiltered
  copy is now transient).
- With implicit MT, `rdfentry_` is unique but neither the entry number nor reproducible. Use content keys.
- Overlay-MC event numbers repeat within a file.
- TensorFlow crashes when imported after ROOT's MT RDataFrame (and cannot use the RTX 5090).
- Building physics vs pulser samples from different overlay files mismatches the background: use an even/odd split.
- An all-hit occupancy feature leaks the signal into PU training.

## 18. Step 2b (2026-11-08): data/MC vs pure MC, noise floor, vertices

- **Data/MC is now judged against pure MC** (`SVT_Hit_Filter_ML.ipynb` §6b). The overlay MC's background is real
  pulser data.
  - Per-layer time σ68 of golden-track hits, data vs pure MC: L1–2 5.4/4.2, L3–4 2.9/4.2, L5–8 3.7/2.7,
    L9–14 2.6/2.5 ns.
  - L1–L4 charge is about 20 % higher in data.
  - The background alters simulated hits: L1–L2 partner rate 0.76 (pure) → 0.91 (overlay), and similar in data.
- **Reweighting reference.**
  - Pure-MC reference: bare features only. Overlay reference: as in v1.
  - Selected on data, by pulser hits removed at 0.3 % golden-track loss: 34.6 % vs 44.5 %.
  - The overlay reference is exported; the model is identical to v1.
- **Recon test** with 70 files × 3 jobs (`Recon_Test.ipynb`; baseline, baseline2 = identical rerun, filtered).
  - Noise floor: 1.6 %/1.5 % of found particles lost/gained between identical jobs.
  - Filter vs repeat, net found:
    - ECal acceptance: −0.01 vs −0.04 %;
    - SVT acceptance: −0.33 vs −0.14 %, with electrons at −0.67 %.
  - Quality gains as in v1, now on 116 k events.
- **V0 study** (§4, `v0_type == 1`, loose `PRESEL`).
  - Preselected true pairs +6.8 %.
  - σ68(z) 1.56 → 1.48 mm.
  - Wrong L1/L2 hit on true pairs 30 → 21 %.
  - Tails are within the run-to-run spread: baseline vs repeat differ by up to 30 % beyond 10 mm.
- **Open:** tight cuts (user will supply), seeded recon, per-layer MC time smearing.
- **Generators:** in the session scratchpad (`make_hitfilter_nb.py`, `make_recontest_nb.py`, `fill_text.py`). The
  notebooks themselves are the deliverables.
