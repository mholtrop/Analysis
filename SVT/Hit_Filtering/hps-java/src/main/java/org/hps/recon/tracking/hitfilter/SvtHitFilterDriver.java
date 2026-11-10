package org.hps.recon.tracking.hitfilter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.lcsim.event.EventHeader;
import org.lcsim.event.RawTrackerHit;
import org.lcsim.event.TrackerHit;
import org.lcsim.recon.tracking.digitization.sisim.SiTrackerHitStrip1D;
import org.lcsim.util.Driver;

/**
 * Machine-learned SVT hit filter, run before Kalman pattern recognition.
 *
 * <p>Every 1D strip cluster in {@code StripClusterer_SiTrackerHitStrip1D} is scored with a gradient-boosted-tree model
 * (see {@link SvtHitFilterFeatures} for the inputs and {@link TreeEnsembleModel} for the format). Clusters with a
 * score below the threshold are removed, and the collection is replaced in place (same name, same class), exactly
 * like {@code StripHitKiller} does. The Kalman driver therefore needs no change.
 *
 * <p>Steering: insert after the strip clustering (and, in MC, after {@code StripHitKiller} and
 * {@code StripHitNHitsSmearer}) and before {@code KalmanPatRecDriver}:
 *
 * <pre>
 *   &lt;driver name="SvtHitFilter"/&gt;            &lt;!-- in &lt;execute&gt;, just before KalmanPatRecDriver --&gt;
 *   ...
 *   &lt;driver name="SvtHitFilter" type="org.hps.recon.tracking.hitfilter.SvtHitFilterDriver"&gt;
 *       &lt;modelFile&gt;/path/to/svt_hitfilter_v1.txt&lt;/modelFile&gt;
 *       &lt;timeOffset&gt;-0.1&lt;/timeOffset&gt;     &lt;!-- signal hit time of this data set [ns] --&gt;
 *   &lt;/driver&gt;
 * </pre>
 *
 * Parameters (all optional except modelFile): {@code threshold} (default: the model's working point),
 * {@code timeOffset} (default: model meta "t0"), {@code targetZ} (default: model meta "targetZ"),
 * {@code collectionName}, {@code keepUnfiltered} (keep the original hits as the transient collection
 * {@code <collection>_Unfiltered}, not written to LCIO),
 * {@code dryRun} (score and count only, keep all hits), {@code maxHits} (skip filtering above this many hits; 0 = no
 * limit), {@code debug}.
 */
public class SvtHitFilterDriver extends Driver {

    private static final Logger LOGGER = Logger.getLogger(SvtHitFilterDriver.class.getPackage().getName());

    private String modelFile = null;
    private String collectionName = "StripClusterer_SiTrackerHitStrip1D";
    private double threshold = Double.NaN;
    private double timeOffset = Double.NaN;
    private double targetZ = Double.NaN;
    private boolean keepUnfiltered = false;
    private boolean dryRun = false;
    private int maxHits = 0;
    private boolean debug = false;

    private TreeEnsembleModel model;
    private long nEvents = 0, nIn = 0, nOut = 0;
    private final long[] nInLayer = new long[15], nOutLayer = new long[15];

    public void setModelFile(String s) { modelFile = s; }
    public void setCollectionName(String s) { collectionName = s; }
    public void setThreshold(double v) { threshold = v; }
    public void setTimeOffset(double v) { timeOffset = v; }
    public void setTargetZ(double v) { targetZ = v; }
    public void setKeepUnfiltered(boolean b) { keepUnfiltered = b; }
    public void setDryRun(boolean b) { dryRun = b; }
    public void setMaxHits(int n) { maxHits = n; }
    public void setDebug(boolean b) { debug = b; }

    @Override
    protected void startOfData() {
        if (modelFile == null) throw new RuntimeException("SvtHitFilterDriver: modelFile is not set");
        try {
            model = TreeEnsembleModel.load(modelFile);
        } catch (IOException e) {
            throw new RuntimeException("SvtHitFilterDriver: cannot read model " + modelFile, e);
        }
        if (Double.isNaN(threshold)) threshold = model.getThreshold();
        if (Double.isNaN(timeOffset)) timeOffset = parseMeta("t0", 0.0);
        if (Double.isNaN(targetZ)) targetZ = parseMeta("targetZ", -1.1);
        LOGGER.config(String.format("SvtHitFilterDriver: model %s (%d stages), threshold %.6g, t0 %.3f ns, targetZ %.3f mm%s",
                modelFile, model.getNumberOfStages(), threshold, timeOffset, targetZ, dryRun ? ", DRY RUN" : ""));
    }

    private double parseMeta(String key, double fallback) {
        String v = model.getMeta(key);
        if (v == null) return fallback;
        try {
            return Double.parseDouble(v.trim().split("\\s+")[0]);
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    @Override
    protected void process(EventHeader event) {
        if (!event.hasCollection(TrackerHit.class, collectionName)) return;
        List<TrackerHit> hits = event.get(TrackerHit.class, collectionName);
        ++nEvents;
        final int n = hits.size();
        nIn += n;
        if (n == 0 || (maxHits > 0 && n > maxHits)) {
            nOut += n;
            return;
        }
        int[] layer = new int[n], module = new int[n], nstrip = new int[n];
        double[] time = new double[n], y = new double[n], z = new double[n], edep = new double[n];
        for (int i = 0; i < n; ++i) {
            TrackerHit h = hits.get(i);
            List<?> raw = h.getRawHits();
            RawTrackerHit r0 = (RawTrackerHit) raw.get(0);
            layer[i] = r0.getIdentifierFieldValue("layer");
            module[i] = r0.getIdentifierFieldValue("module");
            nstrip[i] = raw.size();
            time[i] = h.getTime();
            double[] pos = h.getPosition();
            y[i] = pos[1];
            z[i] = pos[2];
            edep[i] = h.getdEdx();
        }
        float[][] f = SvtHitFilterFeatures.compute(layer, module, time, y, z, edep, nstrip, timeOffset, targetZ);
        List<TrackerHit> kept = new ArrayList<>(n);
        for (int i = 0; i < n; ++i) {
            final double s = model.score(f[i]);
            final boolean keep = s >= threshold;
            if (layer[i] >= 1 && layer[i] <= 14) {
                ++nInLayer[layer[i]];
                if (keep) ++nOutLayer[layer[i]];
            }
            if (keep) kept.add(hits.get(i));
        }
        nOut += kept.size();
        if (debug)
            LOGGER.info(String.format("SvtHitFilterDriver: event %d: %d -> %d hits", event.getEventNumber(), n, kept.size()));
        if (dryRun) return;
        if (keepUnfiltered) {
            // Transient: available to later drivers, never written to LCIO. Writing the same hit objects in two
            // persistent collections corrupts the track -> hit references in the output file (seen in tests).
            List<TrackerHit> all = new ArrayList<>(hits);
            event.put(collectionName + "_Unfiltered", all, SiTrackerHitStrip1D.class, 0, toString());
            event.getMetaData(all).setTransient(true);
        }
        event.remove(collectionName);
        event.put(collectionName, kept, SiTrackerHitStrip1D.class, 0, toString());
    }

    @Override
    protected void endOfData() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("SvtHitFilterDriver: %d events, %d -> %d hits (%.1f%% kept)%s%n", nEvents, nIn, nOut,
                nIn > 0 ? 100.0 * nOut / nIn : 0.0, dryRun ? " [dry run: nothing removed]" : ""));
        sb.append("  layer:      ");
        for (int l = 1; l <= 14; ++l) sb.append(String.format("%6d", l));
        sb.append(String.format("%n  kept [%%]:  "));
        for (int l = 1; l <= 14; ++l) sb.append(String.format("%6.1f", nInLayer[l] > 0 ? 100.0 * nOutLayer[l] / nInLayer[l] : 0.0));
        LOGGER.log(Level.INFO, sb.toString());
        System.out.println(sb);
    }
}
