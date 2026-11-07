# Critical review of step 1: truth and efficiency definitions

Date: 2026-11-06. Applies to `Tracking_Efficiency_SVT_Hits.ipynb` and the `Analysis/svt_tracking` package.
The checks below used 30 files of each sample (pure and overlay).

## 1. Truth definition: mostly sound

**How it works.** Each SVT hit (a strip cluster) has a list of MC particles. The list is built from the simulated hits
behind each strip of the cluster (`LcioReader.cxx`, around line 1070). A track's truth particle is the particle that
contributes to the most hits on the track.

**What the checks found:**
- **Truth links are complete.** No hit points to an unknown particle (index −1).
- **Shared hits are rare.** Only 0.35% of hits come from more than one particle.
- **Weak contributions are rare.** In only 0.13% of the correct hits on matched tracks does the matched particle deposit
  less than half the charge.
- **Delta rays are negligible.** They cause 0.2% of the wrong hits.
- **Where the wrong hits on matched tracks come from:**

  | Sample | Main source |
  |---|---|
  | pure | the other trident particles (95%) |
  | overlay | overlaid data hits (97%) |

**Caveats:**
1. **The overlaid background has no truth.** I called the 41.5% of overlay tracks with no MC hit "fakes", but that is
   wrong. They are a mix of real particles from the pulser event and genuine combinatorial fakes. The current data
   cannot tell the two apart.
2. **"Truth layers" come from reconstructed hits.** They are not where the particle actually crossed a sensor. So they
   already include the SVT hit inefficiency, dead channels and the hit killing applied in the MC reconstruction. As a
   result, the missing-hit fractions are lower limits. The true sensor crossings are in the LCIO `TrackerHits`
   collection (the simulated hits), but they are not in the MiniDST.
3. **Any contribution counts.** A hit is assigned to a particle even if the particle left only a little charge in it.
   This barely matters now (0.13%), but requiring a minimum charge fraction would make the definition robust.

## 2. Efficiency definition: incomplete

**Denominator problems:**
- **It includes the recoil electron.** Each event has 3 charged primaries (e⁺, the pair e⁻ and the recoil e⁻), and they
  are not separated.
- **The acceptance comes from the ECal, not the SVT geometry.**
  - 0.7% (pure) and 2.4% (overlay) of accepted particles never produce an SVT hit. All of them go through the beam gap
    (|tanλ| < 0.015).
  - Beam-gap particles make up about 17% of the denominator.
- **The "≥ 8 truth layers" acceptance is circular,** because it is built from reconstructed hits.

**Numerator problems:**
- **"Found" only means a track was found.** The track's majority of hits must come from the particle (purity ≥ 0.5).
  There is no cut on fit quality (χ²), momentum accuracy (Δp/p), track time or matching to an ECal cluster. So the
  0.93–0.95 is a *track-finding* efficiency. It is not comparable to Zhaozhong's numbers, which use track–cluster
  matching with quality cuts.
- **Duplicate tracks are not reported.** In the overlay, 2.1% of particles have two or more matched tracks (0.06% in
  pure).

**Pure vs overlay is not a like-for-like comparison.** The two samples differ in beam spot, hps-java version and
readout steering. So the differences between them are not caused by background alone. The overlay also has more
particles in the beam gap with no SVT hits.

**Smaller issues:**
- **Timing:** each sample uses a single time offset, and the in-time cut of ±6 ns was chosen by hand. The time
  resolution of signal hits comes from simulation and has not been checked against data.
- **Hit type is hard-coded.** Tracks use hit type 1 in the MC and pass5_v10 data, but type 0 in pass5_v9 data.
- **Tracks are not filtered by `track_type`.** All tracks are type 1 in these files, but other productions may differ.

## 3. Suggested next steps, in priority order

1. **Tighten the definitions.** This is cheap and needs only the MiniDSTs.
   - Report efficiency in tiers: found → clean (≤ 1 wrong hit or purity ≥ 0.8) → good quality (χ², Δp/p within 3σ)
     → in time → matched to an ECal cluster.
   - Report e⁺, pair e⁻ and recoil e⁻ separately, together with the duplicate-track rate.
   - Treat the beam gap as its own category.
   - Detect the hit type automatically and filter on `track_type`.
2. **Get the true sensor crossings.** Add the simulated `TrackerHits` to the MiniDST, or read them from the recon LCIO.
   This gives an acceptance that is not circular and the true per-layer hit efficiency.
3. **Set an upper bound with a perfect filter.**
   - Re-run reconstruction on the overlay readout files with every data hit removed. This shows the best any hit filter
     can do, and gives a baseline under identical conditions.
   - Also reconstruct the pulser events on their own, to separate real background tracks from combinatorial fakes.
4. **Develop the hit filter.**
   - Use the raw-hit information in `minidst_plus`: time, time error, amplitude and pulse-fit χ², plus cluster size, per
     layer.
   - Score each filter offline as wrong hits removed vs correct hits lost.
   - Re-run reconstruction for the best one or two candidates and compare against the upper bound from step 3.
5. **Validate on data.**
   - Compare hit-time and amplitude distributions on high-quality tracks in the unskimmed data (14 runs,
     `5percent/pass5_v9_unskimmed`).
   - Check the full chain against Zhaozhong's results: track–cluster matching and the trident pair yield.

---

# Review 2 (2026-11-07): top/bottom resolution and efficiency vs hits

**Is the analysis correct? Yes.** Every number in both notebooks reproduces exactly after the fixes below. The large
top/bottom difference in momentum resolution for electrons is a **real detector feature**, present in the MC
conditions and in data.

## Cause of the bottom-electron resolution (σ(Δp/p) 6–9 % vs 4 % top, pure MC)

- **The whole L9 bottom electron-side stereo sensor is dead.**
  - In the conditions: 640 of 640 strips in `svt_bad_channels` (collection 4544, "Dead Channels for 2021 Run",
    runs 14000–15000) of `hps_conditions_11102025_moller_5.db`.
  - In the data: `svt_hit_module` 1 in L9 has 0 hits in MC and in run 14229.
- **It removes a stereo measurement from the bottom only.** Stereo = even layers in the top and odd layers in the
  bottom (`svt_daq_map`), so L9 is stereo only in the bottom. Only the stereo strips measure the bend.
  - 61 % of bottom e⁻ have no stereo hit in L9–L14 (top: 23 %).
  - At equal stereo count, top and bottom resolutions agree: 10 %, 4 %, 2.6–3 % for 0, 1, 2–3 stereo hits.
- **The L9 bottom e⁻ "wrong hit" spike (13 % / 35 %) is a normalisation artefact** of the dead sensor. Per track the
  rate is below 0.1 %.
- **L1 and L2 top start further from the beam than in the bottom.** The L1 coverage begins at |tanλ| ≈ 0.02 (top)
  against 0.012 (bottom). Only 47 % of top e⁺ in the SVT acceptance have an L1 hit, against 93 % in the bottom.
  Data confirm it: on in-time tracks with ≥ 10 hits and p > 0.8 GeV, 60 % of top and 85 % of bottom tracks have an
  L1 hit. Hit killing is not the cause (≥ 98 % kept on average in L1–L2).
- Smearing (`posSmearing_pass5_v9.txt`, time smearing) is symmetric between top and bottom.

## Efficiency vs hits: the KF requirements

The production steering sets:
- ≥ 9 hits and ≥ 4 stereo hits in iteration 1;
- ≥ 8 hits and ≥ 3 stereo hits in iteration 2;
- ≥ 2 axial hits;
- a 5-hit seed from fixed module patterns;
- a 40 ns maximum time range.

In pure MC this explains the step at 8–10 layers:
- Efficiency is 0 below 3 stereo layers.
- At 8 layers (iteration 2 only) it is 0.76–0.86, at 9 layers 0.89–0.96.
- With only 3 stereo layers it saturates near 0.89.
- 8-layer patterns with one sensor per back module are rarely seeded (e.g. L3–L7 + L9, L11, L13: 9 %).

## Fixes and additions

- **Package bug:** `_select` built `expr[where]` without parentheses, so a compound expression with `where` was
  masked only on its last term. No existing notebook cell was affected; it is now `(expr)[where]`. The `t0`
  offsets are also parenthesised.
- **Package additions:**
  - columns `tp_stereo`, `tp_n_stereo`, `tp_n_axial`, `tp_trk_layers`, `tp_trk_stereo`, `trk_stereo`,
    `trk_n_stereo`, `trk_n_axial`;
  - C++ helpers `svt::stereo_layers` and `svt::axial_layers`, and the masks `kStereoTop/Bot` and
    `kFront/BackLayers`;
  - `KF_REQUIREMENTS`, `DEAD_SENSORS`, `DATA` + `load_hits()` (truth-free loader for data) and `plot_eff_map()`.
- **Notebooks:**
  - step 1: new §2b (efficiency maps in axial × stereo and front × back layers);
  - step 1b: new §4a (dead sensor, layer coverage, data cross-check, resolution vs back-layer stereo hits), with
    the §2, §5 and summary text corrected.
