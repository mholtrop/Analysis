# SVT hit filter (ML) — HPS 2021

A gradient-boosted-tree classifier scores every SVT strip cluster before Kalman pattern recognition. Likely
background hits are removed, tuned so that good tracks are not lost. It runs in hps-java as a driver added by a
steering change.

## Results (2026-11-08)

- **Offline (MC truth, overlay test files 501–530):** at 0.3 % of found tracks lost, the MC-trained model removes
  71 % of the background hits and 39 % of the wrong hits on matched tracks. A time window at the same loss removes
  20 % and 0.2 %.
- **Data (run 14229):**
  - The MC-only model would lose 1.5 % of golden tracks.
  - The exported model was trained on MC truth with its inputs reweighted to data control samples. It is used at a
    threshold calibrated on data: it loses 0.3 % of golden tracks and removes 44.5 % of the pulser hits.
- **Data/MC (notebook §6b).** The simulation is judged against **pure MC**: in the overlay MC the background *is*
  real pulser data.
  - Biggest difference: the per-layer hit-time resolution. Data vs pure MC: L3–L4 2.9 vs 4.2 ns, L5–L8 3.7 vs
    2.7 ns, plus residual offsets of up to 0.5 ns between layers in data. The MC time smearing should be retuned
    per layer.
  - Charge in L1–L4 is about 20 % higher in data.
  - The real background itself changes simulated hits (pile-up charge and time tails, random partner hits). So the
    training sample (overlay MC) is reweighted against the overlay, not against the pure MC. A pure-MC-reference
    reweighting was tested and is worse on data: 35 % vs 44.5 % of pulser hits removed at equal golden-track loss.
- **hps-java reconstruction, 116 k held-out MC events** (`Recon_Test.ipynb`, files 0531–0600). Three jobs:
  baseline, a repeated baseline (the noise floor of the unseeded MC smearing), and with the filter.
  - Found particles: no gain. In the ECal acceptance, net −0.01 ± 0.04 % (repeat −0.04 %). In the SVT acceptance,
    −0.33 % (repeat −0.14 %), mostly electrons outside the ECal acceptance.
  - Track quality:
    - L1/L2 wrong hits 24 %/20 % → 17.5 %/15 %; L1 missing hits 21 % → 13 %;
    - positron full chain to the ECal match +2.5 / +3.4 points;
    - real trident tracks among all tracks 50 % → 61 %.
  - Vertices:
    - 7 % more true pairs pass a loose preselection;
    - true-pair σ68(z) 1.56 → 1.48 mm;
    - high-z tails not resolvable, because two identical baseline jobs already differ by up to 30 %.
  - Reconstruction is 5–20 % faster.
  - **Conclusion:** the filter does not find more tracks; it makes the found tracks better. Whether less background
    survives a displaced-vertex analysis needs the tight cuts (see Open items).

## Files

| path | what |
|---|---|
| `SVT_Hit_Filter_ML.ipynb` | design, model choice and benchmark, working point, data/MC, fine-tuning and working point on data, stage-2 study, export + Java parity test |
| `Recon_Test.ipynb` | hps-java reconstruction: baseline, repeated baseline (noise floor), with the driver. Tiers, wrong/missing hits, paired lost/gained, V0 vertex tails (`PRESEL` = placeholder for the tight cuts) |
| `../../Analysis/svt_hitfilter.h` | per-hit features (C++, used for training) — **mirrored in Java** |
| `../../Analysis/svt_hitfilter.py` | data sets, KF findability emulator, evaluation, reweighting, model export/import |
| `models/svt_hitfilter_v1.txt` | exported model (MC truth reweighted to run 14229); default threshold = run-14229 working point; `meta threshold_overlay_mc` = MC working point |
| `models/svt_hitfilter_mc_v1.txt` | MC-only model (for comparison) |
| `models/parity_overlay.csv` | raw inputs, features and scores of 300 MC events for the parity test |
| `hps-java/src/main/java/org/hps/recon/tracking/hitfilter/` | `SvtHitFilterDriver`, `SvtHitFilterFeatures`, `TreeEnsembleModel` (no dependencies) |
| `hps-java/src/test/java/.../SvtHitFilterParityTest.java` | Java vs Python features and scores |
| `hps-java/steering/` | production overlay-MC recon steering with the driver (written by the notebook, MC threshold) |
| `hps-java/build.sh` | compile against an hps-java distribution jar → `build/hps-svt-hitfilter.jar` |
| `hps-java/run_recon_test.sh` | reconstruct overlay readout files with/without the filter and make MiniDSTs (`MODES`: any of `baseline`, `baseline2`, `filtered`; `MAXJOBS`) |

Test reconstruction output: `/net/data/endeavour2/HPS/physrun2021/home_made_mc/hitfilter_test/{baseline,baseline2,filtered}`
(overlay readout files 0531–0600). Re-make with `MODES="baseline baseline2 filtered" ./run_recon_test.sh 531 600`.

## Use in hps-java

1. `hps-java/build.sh [hps-distribution-*-bin.jar]`
2. Add the jar to the class path:
   `java -cp hps-distribution-bin.jar:build/hps-svt-hitfilter.jar org.hps.job.JobManager ... steering.lcsim`.
   Alternatively, copy the three classes into hps-java under `tracking/src/main/java/org/hps/recon/tracking/hitfilter/`.
3. In the steering file, put `<driver name="SvtHitFilter"/>` in `<execute>` immediately before
   `KalmanPatRecDriver`, and define it:

```xml
<driver name="SvtHitFilter" type="org.hps.recon.tracking.hitfilter.SvtHitFilterDriver">
    <modelFile>/net/home/maurik/Analysis/SVT/Hit_Filtering/models/svt_hitfilter_v1.txt</modelFile>
    <timeOffset>0.4</timeOffset>   <!-- signal hit time of the data set [ns]: 0.4 run 14229, -0.1 overlay MC -->
    <!-- overlay MC: <threshold>0.009064</threshold> (meta threshold_overlay_mc); default = run-14229 point -->
</driver>
```

Optional parameters:
- `threshold`;
- `targetZ` (default −1.1 mm);
- `keepUnfiltered`: keeps the original hits as a *transient* collection `<name>_Unfiltered`;
- `dryRun`: count only;
- `maxHits`, `debug`.

At the end of the job the driver prints the fraction of hits kept per layer.

**Notes**
- **Never write the same hit objects into two persistent collections.** An early test did that, and the track→hit
  references in the output LCIO were corrupted, which halved the apparent efficiency. That is why the unfiltered
  copy is transient.
- **Calibrate `timeOffset` and the threshold for every new data set.** Use the median time of hits on good tracks,
  and the golden-track loss scan of notebook §6e.
- **Running recon locally needs a writable lcsim cache that contains the detector.** `run_recon_test.sh` seeds
  `~/.lcsim_hps2021` from `/net/home/erbaz/ws/.lcsim`. Use `make_mini_dst` from `/net/home/erbaz/hps-analysis/install`
  (LcioReader 1.6.0).

## Changing features

Edit `svt_hitfilter.h` **and** `SvtHitFilterFeatures.java` identically. Then retrain and re-export with the notebook,
and check that the parity test (§8) prints `PASSED`.

## Open items

- **Tight-cut vertex analysis.** Does less background survive the displaced-vertex selection? Replace `PRESEL` in
  `Recon_Test.ipynb` §4 by the analysis cuts. This needs more statistics or a high-z-enriched sample, and seeded MC
  recon.
- **Seeded MC recon.** `RawHitTimeSmearer`, `StripHitNHitsSmearer` and `TrackTimeSmearer` use an unseeded
  `new Random()`. Two identical jobs therefore differ by about 1.5 % of particles in each direction, and by up to
  30 % in vertex tails. A seed parameter would make baseline-vs-filter comparisons exact.
- **MC time smearing per layer.** In data vs pure MC, L3–L4 are better and L5–L8 worse than in the simulation
  (`timeSmearing-2nsL4L7-4nsL1L3.txt`); notebook §6b.
- **Electron loss outside the ECal acceptance** (about 0.4 % beyond the noise floor). Use a category- or
  layer-dependent working point.
- **Data reconstruction test.** Run run-14229 LCIO through recon with the driver: `dryRun` first, then filtering.
  Compare golden-track yields and trident distributions.
- **Stage 2.** Post-KF hit cleaning with hit − track time (notebook §7) removes 57 % of the remaining wrong hits at
  0.2 % correct-hit loss. It needs a hook in `KalmanPatRecHPS` or a GBL refit.
