#!/bin/bash
# End-to-end test of SvtHitFilterDriver in hps-java reconstruction.
# Reconstructs overlay-MC readout files twice with the production steering, without and with the hit filter
# (same jar, same conditions), and makes MiniDSTs of both, ready for SVT/Hit_Filtering/Recon_Test.ipynb.
#   [MODES="baseline filtered"] [MAXJOBS=24] ./run_recon_test.sh <first file number> <last file number> [output dir]
# Use file numbers not used for training (the notebook uses overlay files 1-100 and 501-530), e.g. 561-600.
# A mode whose name starts with "filtered" uses the hit-filter steering, any other mode the production steering:
# e.g. MODES=baseline2 repeats the baseline (the MC smearing drivers are not seeded) to measure the noise floor.
set -e
cd "$(dirname "$0")"
FIRST=${1:-1}; LAST=${2:-1}
OUT=${3:-/net/data/endeavour2/HPS/physrun2021/home_made_mc/hitfilter_test}
JAR=/net/home/erbaz/hps-java/distribution/target/hps-distribution-5.3.1-SNAPSHOT-bin.jar
DB=jdbc:sqlite:/net/home/erbaz/ws/hps_conditions_11102025_moller_5.db
SAMPLE=/net/data/endeavour2/HPS/physrun2021/home_made_mc/tritrig_pulser_beamspot_run14272
BASE_STEER=/net/home/erbaz/hps-java/steering-files/src/main/resources/org/hps/steering/recon/PhysicsRun2021MCRecon_KF_WithSpacing_pass5_physics_HitSmear_HitKill_14272.lcsim
FILT_STEER=$PWD/steering/PhysicsRun2021MCRecon_KF_WithSpacing_pass5_physics_HitSmear_HitKill_14272_SvtHitFilter.lcsim
# make_mini_dst as in the production (LcioReader 1.6.0); needs ROOT and the LcioReader libraries
source /net/data/endeavour2/Tools/root-master-gcc11.5.0/bin/thisroot.sh
export LD_LIBRARY_PATH=/net/home/maurik/AlmaLinux/lib:/net/home/erbaz/hps-analysis/install/lib:$LD_LIBRARY_PATH
MKDST=/net/home/erbaz/hps-analysis/install/bin/make_mini_dst
./build.sh "$JAR" > /dev/null
# private copy of the filter jar: a later build (or a second instance of this script) must not overwrite the jar
# that running jobs have open
FJAR=$(mktemp "${TMPDIR:-/tmp}/hps-svt-hitfilter.XXXXXX.jar"); cp build/hps-svt-hitfilter.jar "$FJAR"
for mode in ${MODES:-baseline filtered}; do mkdir -p "$OUT/$mode"; done
# lcsim cache: must be writable (field map) and contain the detector; seeded from /net/home/erbaz/ws/.lcsim
LCSIM_CACHE=${LCSIM_CACHE:-/net/home/maurik/.lcsim_hps2021}
[ -d "$LCSIM_CACHE/.lcsim" ] || { mkdir -p "$LCSIM_CACHE"; cp -r /net/home/erbaz/ws/.lcsim "$LCSIM_CACHE/"; }
export JAVA_TOOL_OPTIONS="-Dorg.lcsim.cacheDir=$LCSIM_CACHE -Xmx4g"
for i in $(seq -f "%04g" "$FIRST" "$LAST"); do
  IN=$SAMPLE/readout/tritrig_pulser_3pt74_readout_$i.slcio
  for mode in ${MODES:-baseline filtered}; do
    steer=$BASE_STEER; [[ $mode == filtered* ]] && steer=$FILT_STEER
    o=$OUT/$mode/tritrig_pulser_3pt74_recon_$i
    if [ ! -s "$o.root" ]; then
      while [ "$(jobs -rp | wc -l)" -ge "${MAXJOBS:-24}" ]; do wait -n || true; done
      ( java -Dorg.hps.conditions.url=$DB -cp "$JAR:$FJAR" org.hps.job.JobManager \
          -d HPS_Run2021Pass1_v9_14272 -R 14272 -DoutputFile="$o" -i "$IN" "$steer" > "$o.log" 2>&1 \
        && $MKDST -a -o "$o.root" "$o.slcio" >> "$o.log" 2>&1 \
        && echo "$mode $i done" ) &
    fi
  done
done
wait
rm -f "$FJAR"
grep -h -A3 "SvtHitFilterDriver:" "$OUT"/filtered*/*.log 2>/dev/null | head -20 || true
