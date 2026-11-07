# Location of HPS data on endeavour

Survey of 2021 (and 2019) HPS data on endeavour, made 2026-11-06.

* **E** = `/net/data/endeavour2/HPS/physrun2021`  (37 TB disk, network)
* **F** = `/fast/HPS`  (3.6 TB local NVMe on endeavour, fast)

Event counts are the MiniDST/ntuple entries. The detector was read with `dumpevent` from the first SLCIO file of
the sample (or the files it was made from). Every MiniDST has a `metadata_json` object that lists its input SLCIO file(s).

**SVT column:** H = SVT hits (`svt_hit_*`), T = per-hit MC truth (`svt_hit_mc_part_idx`), R = raw strip hits
(`svt_raw_hit_*`, amplitude/t0/χ²), raw = readout-level LCIO, sim = SLIC output, – = no SVT hit branches
(tracks only).

**\* Detector inferred** from the pass5_v9 production path in `metadata_json`: the input SLCIO files are at JLab
(`/lustre24/expphy/cache/hallb/hps/production/...`), not on endeavour. "SLAC" = input recon files at SLAC (`/sdf/...`).

## Home-made MC (erbaz, Oct 2026) — best documented, latest software

All in `E/home_made_mc/`. Each sample has an `MC_INFO.md` with the full chain, steering, versions and reproduction configs (`_config/`). 3.74 GeV, run-14272 conditions, local SQLite conditions DB, KF tracking, recon steering `PhysicsRun2021MCRecon_KF_WithSpacing_pass5_physics_HitSmear_HitKill_14272`. Overlay samples use real 2021 pulser data; no pulser event is shared between samples.

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/home_made_mc/tritrig_pure_run14272/minidst` | Tritrig, **no background** (pure) | 474 | 5 | 1.01 M | v9_14272 | H T |
| `E/home_made_mc/tritrig_pure_run14272/minidst_plus` | Same events + SVT/ECal/hodo raw hits | 474 | 8 | 1.01 M | v9_14272 | H T R |
| `E/home_made_mc/tritrig_pure_run14272/recon` | Recon LCIO (input of the above) | 474 | 17 |  | v9_14272 | H T R |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/minidst` | Tritrig + pulser overlay, beam spot 55×45 µm | 601 | 12 | 0.99 M | v9_14272 | H T |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/minidst_plus` | Same events + raw hits | 601 | 29 | 0.99 M | v9_14272 | H T R |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/recon` | Recon LCIO | 601 | 45 |  | v9_14272 | H T R |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/readout` | Readout LCIO (re-run recon from here) | 601 | 10 |  | v9_14272 | raw |
| `E/home_made_mc/tritrig_pulser_beamspot_run14272/{lhe,conv,mom,rot,sim,filt,overlay,spaced}` | All intermediate steps (LHE → SLIC → overlay → spacing) |  | 850 |  | v9_14272 | – |
| `E/home_made_mc/wab_pulser_run14272/minidst` | WAB + pulser overlay (also `recon/`, `lhe/`) | 720 | 2 | 0.20 M | v9_14272 | H T |
| `E/home_made_mc/ap_pulser_lheuniform_run14272/m*MeV/minidst` | A′ + pulser overlay, decay length uniform 0–150 mm; 12 masses 30–200 MeV (all steps kept) | 1,952 | 80 | 6.63 M | v9_14272 | H T |
| `E/home_made_mc/ap_pure_lheuniform_run14272/m*MeV/minidst` | A′, no overlay; masses 50, 100, 150, 200 MeV (also `recon/`) | 962 | 10 | 1.98 M | v9_14272 | H T |
| `E/home_made_mc/pulser_pieces` | Pulser data split into ~18k-event pieces, used for the overlays | 3,760 | 601 |  | v9_14272 | raw |

## Other 3.74 GeV MC (pass5 / v9)

Older productions; mostly no per-hit SVT truth. Detector from the first SLCIO file.

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `F/physrun2021/pass5_v9_mc_rad_readout_recon/minidst` | Rad tridents, **pure**, same recon as tritrig_pure (has `MC_INFO.md`) | 2,000 | 11 | 2.13 M | v9_14272 | H T |
| `F/physrun2021/pass5_v9_mc_rad_readout_recon/recon` | Recon LCIO of the above | 2,000 | 42 |  | v9_14272 | H T R |
| `F/physrun2021/pass5_v9_mc/rad_readout_slic` | SLIC output (input of the above, 20M events) | 2,000 | 95 |  | v9_14272 | sim |
| `E/3pt74_MC/tritrig_pulser_newmdst/mdst` | Tritrig + pulser (also `slcio/`) | 100 | 24 | 1.78 M | v9_14272 | H |
| `E/3pt74_MC/tritrig_pulser` | Tritrig + pulser, old MiniDST (`*.minidst`) | 89 | 19 | 1.55 M | ? | H |
| `E/3pt74_MC/rad_pulser/mdst` | Rad + pulser, hit smear/kill (SLAC; also `slcio/`) | 800 | 49 | 4.26 M | v9_14272 | H |
| `E/3pt74_MC/wab_pulser/mdst` | WAB + pulser (also `slcio/`) | 200 | 3 | 0.23 M | v9_14272 | H |
| `E/3pt74_MC/bh_pulser/mdst` | Bethe–Heitler + pulser (also `slcio/`) | 11 | 2 | 0.21 M | v8beta_template | H |
| `E/3pt74_MC/ap_pulser_truth_matched_mdst` | A′ 120 MeV + pulser (same events as `ap_pulser_mdst_new`, `ap_pulser`) | 200 | 36 | 3.02 M | v9_14272 | H T |
| `E/3pt74_MC/ap_pulser_120MEV_slcio` | Recon LCIO of the A′ 120 MeV sample | 200 | 148 |  | v9_14272 | H T R |
| `E/3pt74_MC/ap_prompt_pulser` | Prompt A′ 120 MeV + pulser (SLAC) | 40 | 91 | 6.24 M | v9_14272 | H |
| `E/3pt74_MC/ap_nooverlay/100MEV_mdst` | A′ 100 MeV, no overlay (also `100MEV_slcio`) | 194 | 1 | 0.25 M | v9_14596 | H T |
| `E/3pt74_latest_ML/tritrig_pure/mdst` | Tritrig pure, for ML (also `slcio/`) | 125 | 1.0 | 0.21 M | v8beta_template | H T |
| `E/MC_test/minidst/tritrig_pulser/pass5_v9/*` | Tritrig + pulser, hit smear/kill (`readout/`, `recon/` LCIO alongside) | 751 | 16 | 1.33 M | v9_14272 | H |
| `E/MC_test/slic` | Tritrig SLIC output | 7,999 | 337 |  | v9_14272 | sim |
| `E/MC_test/pulser` | 2021 pulser data (run 14275) in LCIO: the overlay source | 526 | 841 |  | v9_14272 | raw |
| `E/MC_test/run_beam/minidst` | Beam-on-target background test (nearly empty: <1 SVT hit/event) | 62 | 0.3 | 0.62 M | v9_14185 | H T |
| `E/moller_mc` | Møller MC, 1.92 GeV (pure / pulser, smearing tests; SLAC) | 105 | 21 | 3.64 M | ? | H |

## Pass6 (v10-dev) MC

SLAC production (M. Graham), low-p tracking (`pass6dev_lowp`). `README.md` in `tritrig_pulser_readout/`.

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/pass6_v10/mc/tritrig_pulser/mdst` | Tritrig + pulser (also `slcio/`) | 625 | 16 | 1.14 M | v10-dev_14272 | H T |
| `E/pass6_v10/mc/tritrig_pulser_readout` | Readout LCIO (re-run recon from here) | 125 | 2 |  | v10-dev_14272 | raw |
| `E/pass6_v10/mc/tritrig_3track/mdst` | Tritrig 3-track selection (also `slcio/`, hpstr `tuple/`) | 125 | 3 | 0.23 M | v10-dev_14272 | H T |
| `E/pass6_v10/mc/ap_pulser/120MeV/mdst` | A′ 120 MeV + pulser (also `slcio/`) | 10 | 2 | 0.15 M | v10-dev_14272 | H T |
| `E/pass6_v10/mc/*moller*` | Møller MC tests (1.92 GeV) | 3 | 7 | 0.36 M | v9_1pt92GeV_14629 | H T |

## 1.92 GeV MC

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/1pt92_MC/tritrig` | Tritrig + pulser (SLAC pass5_v9_1pt92GeV) | 100 | 0.6 | 0.08 M | v9_1pt92GeV_14629 | H |
| `E/1pt92_MC/tritrig_singles` | Tritrig + pulser, singles trigger (SLAC) | 100 | 5 | 0.50 M | v9_1pt92GeV_14629 | H |
| `E/1pt92_MC/wab` | WAB + pulser (SLAC) | 100 | 0.6 | 0.09 M | v9_1pt92GeV_14629 | H |

## Data, 3.74 GeV

Pass5_v9 unless noted. In pass5_v9 data the tracks point to `svt_hit_type` 0 (1 in MC and pass5_v10).

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/5percent/pass5_v9_unskimmed` | **Unskimmed**, 14 runs 14185–14337 | 410 | 468 | 57.35 M | v9_&lt;run&gt; * | H |
| `E/hps_014229_noskim_mdst` | Unskimmed, run 14229 | 45 | 53 | 6.47 M | v9_14229 * | H |
| `F/physrun2021/pass5_v10_test` | Run 14185 unskimmed, **pass5_v9 and pass5_v10** of the same files (12 + 12) | 24 | 29 | 3.74 M | v9 / v10 * | H |
| `E/5percent/physics` | 5 % sample, 74 runs (identical copy in `F/physrun2021/5percent`) | 496 | 320 | 115.08 M | v9_&lt;run&gt; * | – |
| `E/5percent/physics_newMDST` | 5 % v0 skim, 58 runs, newer MiniDST | 967 | 678 | 235.65 M | v9_&lt;run&gt; * | – |
| `E/5percent/physics_trackZ0Test` | 5 % v0 skim, 188 runs, track z0 test | 296 | 155 | 53.42 M | v9_&lt;run&gt; * | – |
| `E/5percent/hps_14231` | Run 14231 (also `hps_14231_recon/` v0 skim: slcio, mdst, hpstr) | 53 | 41 | 13.76 M | v9_14231 | – |
| `E/FEE` | FEE skim, runs 14271 and 14153 (PIN) | 442 | 124 | 74.88 M | ? | – |
| `E/3pt74_latest_ML/data/mdst/hps_014271` | Run 14271 v0 skim (also `slcio/`) | 52 | 25 | 8.11 M | v9_14271 | – |
| `E/pass6_v10/data/3pt74/mdst` | Pass6 v0 skim, runs 14185, 14222, 14520, 14750 (also `slcio/`) | 85 | 11 | 2.13 M | v10-dev_14185 | – |
| `F/physrun2021/new_dst` | Run 14476 v0 skim | 145 | 11 | 3.65 M | v9_14476 * | – |

## Data, 1.92 GeV

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/1pt92_unskimmed_data` | Unskimmed, run 14629 | 51 | 46 | 8.83 M | v9_1pt92GeV_14629 * | H |
| `E/mollers_newMDST` | Møller skim, 18 runs 14628–14673 | 235 | 65 | 23.50 M | v9_1pt92GeV_&lt;run&gt; * | – |
| `E/pass6_v10/data/1pt92/mdst` | Pass6 Møller skim, 18 runs 14628–14673 (also `slcio/`, no SVT hits) | 603 | 32 | 8.63 M | v9_1pt92GeV_&lt;run&gt; | – |

## hpstr ntuples

Tree `HPS_Event` unless noted.

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `E/hpstr/pass5_v9_preselected_data` | Data, vertex preselection (tree `preselection`), 87 runs | 142 | 25 | 33.04 M | v9_&lt;run&gt; * | – |
| `E/hpstr/tritrig_mc_hitsmearKill` | Tritrig + pulser | 11 | 16 | 0.98 M | v9_14272 | – |
| `E/3pt74_MC/hpstr` | Tritrig and WAB + pulser (also subdirs) | 56 | 63 | 4.14 M | v9_14272 | – |
| `E/3pt74_MC/ap_pulser_120MeV_hpstr` | A′ 120 MeV + pulser | 201 | 98 | 6.05 M | v9_14272 | – |

## 2019 data

| Location | Content | Files | Size (GB) | Events | Detector (HPS_Run2021Pass1_…) | SVT |
|---|---|---|---|---|---|---|
| `F/physrun2019/fee` | 2019 FEE, runs 10049, 10050 (pass0.0) | 32 | 8 | 3.71 M | (2019) | – |

## Notes

- Every MiniDST has tree `MiniDST`. `svt_hit_type` 0 and 1 are identical copies; tracks point to 1 in MC and pass5_v10, to 0 in pass5_v9 data.
- Pass6 data skims (`pass6_v10/data`) have no SVT hit collections, even in their LCIO files.
- Some overlay files lack the pulser SVT data for many events (e.g. A′ m105MeV file 0001): check the per-file mean SVT occupancy (~140 hits/event) first.
- `E/MC_test/run_beam*` is a beam-background MC test that is nearly empty; `E/MC_test/run_beam/work` holds ~140 GB of job scratch.
- `F/physrun2021/5percent` is a copy of `E/5percent/physics` (same files and size).
- Tools: `dumpevent <file> <n>`, `lcio_event_counter`, `make_mini_dst` in `/net/home/maurik/AlmaLinux/bin`.
