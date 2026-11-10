package org.hps.recon.tracking.hitfilter;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gradient-boosted decision trees exported from scikit-learn's HistGradientBoostingClassifier by
 * {@code Analysis.svt_hitfilter.export_model()}. No external dependencies.
 *
 * <p>A model is a chain of binary-classification stages. Stage k reads the base features plus the probabilities of
 * stages 0..k-1; the score of a hit is the probability of the last stage. Arithmetic follows scikit-learn: raw =
 * baseline + sum of leaf values, decision {@code x <= threshold} goes left (NaN follows missing_go_to_left),
 * probability = 1 / (1 + exp(-raw)).
 */
public final class TreeEnsembleModel {

    private static final class Tree {
        boolean[] leaf;
        int[] feature, left, right;
        boolean[] missingLeft;
        double[] threshold, value;

        double eval(double[] x) {
            int node = 0;
            while (!leaf[node]) {
                final double v = x[feature[node]];
                final boolean goLeft = Double.isNaN(v) ? missingLeft[node] : v <= threshold[node];
                node = goLeft ? left[node] : right[node];
            }
            return value[node];
        }
    }

    private static final class Stage {
        String name;
        int nInputs;
        double baseline;
        Tree[] trees;

        double probability(double[] x) {
            double raw = baseline;
            for (Tree t : trees) raw += t.eval(x);
            return 1.0 / (1.0 + Math.exp(-raw));
        }
    }

    private final List<Stage> stages = new ArrayList<>();
    private final Map<String, String> meta = new HashMap<>();
    private String[] featureNames;
    private double threshold = Double.NaN;

    /** Read a model from a file path, or, if no such file exists, from the class path. */
    public static TreeEnsembleModel load(String pathOrResource) throws IOException {
        if (Files.exists(Paths.get(pathOrResource))) {
            try (InputStream in = Files.newInputStream(Paths.get(pathOrResource))) {
                return read(in);
            }
        }
        InputStream in = TreeEnsembleModel.class.getResourceAsStream(pathOrResource.startsWith("/") ? pathOrResource
                : "/" + pathOrResource);
        if (in == null) throw new IOException("SVT hit-filter model not found: " + pathOrResource);
        try (InputStream s = in) {
            return read(s);
        }
    }

    public static TreeEnsembleModel read(InputStream input) throws IOException {
        TreeEnsembleModel m = new TreeEnsembleModel();
        BufferedReader r = new BufferedReader(new InputStreamReader(input, StandardCharsets.UTF_8));
        String line = r.readLine();
        if (line == null || !line.startsWith("hps-svt-hitfilter")) throw new IOException("not an SVT hit-filter model");
        while ((line = r.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] w = line.split("\\s+");
            switch (w[0]) {
                case "meta":
                    m.meta.put(w[1], w.length > 2 ? line.substring(line.indexOf(w[1]) + w[1].length()).trim() : "");
                    break;
                case "features":
                    int nf = Integer.parseInt(w[1]);
                    m.featureNames = new String[nf];
                    System.arraycopy(w, 2, m.featureNames, 0, nf);
                    break;
                case "threshold":
                    m.threshold = Double.parseDouble(w[1]);
                    break;
                case "stage":
                    Stage s = new Stage();
                    s.name = w[1];
                    s.nInputs = Integer.parseInt(w[2]);
                    s.baseline = Double.parseDouble(w[3]);
                    s.trees = new Tree[Integer.parseInt(w[4])];
                    for (int k = 0; k < s.trees.length; ++k) s.trees[k] = readTree(r);
                    m.stages.add(s);
                    break;
                default:
                    throw new IOException("unexpected line in model: " + line);
            }
        }
        if (m.stages.isEmpty() || m.featureNames == null) throw new IOException("incomplete model");
        if (m.featureNames.length != SvtHitFilterFeatures.N_FEATURES)
            throw new IOException("model has " + m.featureNames.length + " features, code computes "
                    + SvtHitFilterFeatures.N_FEATURES);
        for (int i = 0; i < m.featureNames.length; ++i)
            if (!m.featureNames[i].equals(SvtHitFilterFeatures.NAMES[i]))
                throw new IOException("feature " + i + " is " + m.featureNames[i] + " in the model but "
                        + SvtHitFilterFeatures.NAMES[i] + " in the code");
        return m;
    }

    private static Tree readTree(BufferedReader r) throws IOException {
        String[] h = r.readLine().trim().split("\\s+");
        if (!h[0].equals("tree")) throw new IOException("expected 'tree'");
        int n = Integer.parseInt(h[1]);
        Tree t = new Tree();
        t.leaf = new boolean[n];
        t.feature = new int[n];
        t.left = new int[n];
        t.right = new int[n];
        t.missingLeft = new boolean[n];
        t.threshold = new double[n];
        t.value = new double[n];
        for (int i = 0; i < n; ++i) {
            String[] w = r.readLine().trim().split("\\s+");
            t.leaf[i] = Integer.parseInt(w[0]) == 1;
            t.feature[i] = Integer.parseInt(w[1]);
            t.threshold[i] = Double.parseDouble(w[2]);
            t.left[i] = Integer.parseInt(w[3]);
            t.right[i] = Integer.parseInt(w[4]);
            t.missingLeft[i] = Integer.parseInt(w[5]) == 1;
            t.value[i] = Double.parseDouble(w[6]);
        }
        return t;
    }

    /** Score of one hit (probability of the last stage). */
    public double score(float[] features) {
        double[] x = new double[features.length + stages.size()];
        for (int i = 0; i < features.length; ++i) x[i] = features[i];
        double p = Double.NaN;
        int n = features.length;
        for (Stage s : stages) {
            if (s.nInputs != n) throw new IllegalStateException("stage " + s.name + " expects " + s.nInputs + " inputs");
            p = s.probability(x);
            x[n++] = p;
        }
        return p;
    }

    /** Working-point threshold stored with the model (hits with score below it are removed). */
    public double getThreshold() {
        return threshold;
    }

    /** Free-form metadata written at export (e.g. t0, targetZ, training samples). */
    public String getMeta(String key) {
        return meta.get(key);
    }

    public int getNumberOfStages() {
        return stages.size();
    }
}
