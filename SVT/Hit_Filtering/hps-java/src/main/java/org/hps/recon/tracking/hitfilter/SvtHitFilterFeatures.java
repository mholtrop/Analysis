package org.hps.recon.tracking.hitfilter;

/**
 * Per-hit features of the SVT hit filter.
 *
 * <p>This is a line-by-line port of {@code svt::hf::features()} in {@code Analysis/svt_hitfilter.h} (the code used
 * to train the model). Any change must be made in both places, and {@link SvtHitFilterParityTest} must pass.
 *
 * <p>Inputs are the 1D strip clusters of one event (collection {@code StripClusterer_SiTrackerHitStrip1D}), in
 * collection order: layer (1-14) and module from the raw-hit cell ID, time [ns], global y and z [mm], dE/dx (as
 * written to LCIO EDep), and the number of strips. Stereo sensors are the even layers in the top half and the odd
 * layers in the bottom half; module even = top, odd = bottom; in L9-L14 modules 0/1 are electron-side, 2/3
 * positron-side sensors.
 *
 * <p>Feature order (27): layer, top, stereo, side, t, q, nstrip, absy, tl, nsame, pair_dt, pair_dy, has_anchor,
 * nroad_tight, nroad_loose, dtl_m1..dtl_m7, noot_half, road_adt, road_at, nroad_t, ncomp. See the header for the
 * definitions.
 */
public final class SvtHitFilterFeatures {

    public static final int N_FEATURES = 27;
    public static final String[] NAMES = {"layer", "top", "stereo", "side", "t", "q", "nstrip", "absy", "tl", "nsame",
        "pair_dt", "pair_dy", "has_anchor", "nroad_tight", "nroad_loose", "dtl_m1", "dtl_m2", "dtl_m3", "dtl_m4",
        "dtl_m5", "dtl_m6", "dtl_m7", "noot_half", "road_adt", "road_at", "nroad_t", "ncomp"};

    static final double SAME_DT = 10.0, PAIR_DY = 4.0, PAIR_DT = 8.0, ROAD_DT = 8.0, TIGHT = 0.002, LOOSE = 0.006;
    static final double OOT_T = 25.0;
    static final double NO_PAIR_DT = 100.0, NO_PAIR_DY = 50.0, NO_DTL = 1.0, NO_ROAD_T = 100.0, COMP_DY = 2.0;

    private SvtHitFilterFeatures() {
    }

    static boolean isStereo(int layer, boolean top) {
        return top ? (layer % 2 == 0) : (layer % 2 == 1);
    }

    /**
     * Compute the features of all hits of one event.
     *
     * @return array [n][N_FEATURES] in the order of the input hits
     */
    public static float[][] compute(int[] layer, int[] module, double[] time, double[] y, double[] z, double[] edep,
            int[] nstrip, double t0, double zT) {
        final int n = layer.length;
        int[] L = new int[n], M = new int[n], half = new int[n], side = new int[n], mod = new int[n];
        boolean[] st = new boolean[n];
        double[] t = new double[n], yy = new double[n], tl = new double[n];
        for (int a = 0; a < n; ++a) {
            L[a] = layer[a];
            M[a] = module[a];
            half[a] = (M[a] % 2 == 0) ? 1 : 0;
            side[a] = (M[a] >= 2) ? 1 : 0;
            mod[a] = (L[a] - 1) / 2;
            st[a] = isStereo(L[a], half[a] == 1);
            t[a] = time[a] - t0;
            yy[a] = y[a];
            tl[a] = y[a] / (z[a] - zT) * (half[a] == 1 ? 1.0 : -1.0);
        }
        int ntop = 0, nbot = 0;
        for (int a = 0; a < n; ++a) {
            if (Math.abs(t[a]) > OOT_T) {
                if (half[a] == 1) ++ntop; else ++nbot;
            }
        }
        float[][] out = new float[n][N_FEATURES];

        int[] anchor = new int[n];
        double[] pdt = new double[n], pdy = new double[n];
        int[] nsame = new int[n], ncomp = new int[n];
        for (int a = 0; a < n; ++a) {
            pdt[a] = NO_PAIR_DT;
            pdy[a] = NO_PAIR_DY;
            double bestDt = 1e9;
            int best = -1;
            for (int b = 0; b < n; ++b) {
                if (b == a || half[b] != half[a]) continue;
                final double dt = Math.abs(t[b] - t[a]);
                if (L[b] == L[a] && M[b] == M[a]) {
                    if (dt < SAME_DT) {
                        ++nsame[a];
                        if (Math.abs(yy[b] - yy[a]) < COMP_DY) ++ncomp[a];
                    }
                    continue;
                }
                if (mod[b] != mod[a] || side[b] != side[a]) continue;
                final double dy = Math.abs(yy[b] - yy[a]);
                if (dy < PAIR_DY) {
                    pdt[a] = Math.min(pdt[a], dt);
                    if (st[a] && !st[b] && dt < PAIR_DT && dt < bestDt) {
                        bestDt = dt;
                        best = b;
                    }
                }
                if (dt < PAIR_DT) pdy[a] = Math.min(pdy[a], dy);
            }
            anchor[a] = st[a] ? best : a;
        }

        double[][] dtl = new double[n][7];
        int[][] bestHit = new int[n][7];
        for (int a = 0; a < n; ++a) {
            for (int m = 0; m < 7; ++m) {
                dtl[a][m] = NO_DTL;
                bestHit[a][m] = -1;
            }
            if (st[a]) continue;
            dtl[a][mod[a]] = 0.0;
            for (int b = 0; b < n; ++b) {
                if (b == a || st[b] || half[b] != half[a] || mod[b] == mod[a]) continue;
                if (Math.abs(t[b] - t[a]) >= ROAD_DT) continue;
                final double d = Math.abs(tl[b] - tl[a]);
                if (d < dtl[a][mod[b]]) {
                    dtl[a][mod[b]] = d;
                    bestHit[a][mod[b]] = b;
                }
            }
        }

        for (int a = 0; a < n; ++a) {
            float[] f = out[a];
            f[0] = L[a];
            f[1] = half[a];
            f[2] = st[a] ? 1 : 0;
            f[3] = side[a];
            f[4] = (float) t[a];
            f[5] = (float) (edep[a] * 1e6);
            f[6] = nstrip[a];
            f[7] = (float) Math.abs(yy[a]);
            f[8] = (float) tl[a];
            f[9] = nsame[a];
            f[10] = (float) pdt[a];
            f[11] = (float) pdy[a];
            final int an = anchor[a];
            f[12] = an >= 0 ? 1 : 0;
            int nt = 0, nl = 0;
            for (int m = 0; m < 7; ++m) {
                final double d = an >= 0 ? dtl[an][m] : NO_DTL;
                f[15 + m] = (float) d;
                if (an >= 0 && m != mod[an]) {
                    if (d < TIGHT) ++nt;
                    if (d < LOOSE) ++nl;
                }
            }
            f[13] = nt;
            f[14] = nl;
            f[22] = half[a] == 1 ? ntop : nbot;
            double sum = 0.0;
            int nr = 0;
            if (an >= 0) {
                for (int m = 0; m < 7; ++m) {
                    if (m != mod[an] && bestHit[an][m] >= 0 && dtl[an][m] < LOOSE) {
                        sum += t[bestHit[an][m]];
                        ++nr;
                    }
                }
                if (an != a) {
                    sum += t[an];
                    ++nr;
                }
            }
            if (nr > 0) {
                final double tr = sum / nr;
                f[23] = (float) Math.abs(t[a] - tr);
                f[24] = (float) Math.abs(tr);
            } else {
                f[23] = (float) NO_ROAD_T;
                f[24] = (float) NO_ROAD_T;
            }
            f[25] = nr;
            f[26] = ncomp[a];
        }
        return out;
    }
}
