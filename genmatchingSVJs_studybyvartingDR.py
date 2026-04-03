#!/usr/bin/env python3
import os
import argparse
import numpy as np
import awkward as ak
import uproot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Optional CMS style
try:
    import mplhep as hep
    plt.style.use(hep.style.CMS)
    HAVE_HEP = True
except Exception:
    HAVE_HEP = False


# =========================================================
# Helpers
# =========================================================
FATJET_R = 0.8

DEFAULT_DR_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Dark-sector PIDs
DARK_HADRONS = [4900111, 4900211, 4900113, 4900213]
DARK_QUARKS  = [4900101, 4900102, 4900103, 4900104, 4900105, 4900106]

# Candidate branch names for optional jet observables
RECO_OPTIONAL_BRANCHES = {
    "msoftdrop": [
        "FatJet/FatJet.SoftDroppedP4[5]",
        "FatJet/FatJet.SoftDroppedP4_fE",
        "FatJet/FatJet.TrimmedP4[5]",
    ],
    "tau1": ["FatJet/FatJet.Tau[0]", "FatJet/FatJet.Tau1"],
    "tau2": ["FatJet/FatJet.Tau[1]", "FatJet/FatJet.Tau2"],
    "tau3": ["FatJet/FatJet.Tau[2]", "FatJet/FatJet.Tau3"],
}

GEN_OPTIONAL_BRANCHES = {
    "msoftdrop": [
        "GenFatJet/GenFatJet.SoftDroppedP4[5]",
        "GenFatJet/GenFatJet.SoftDroppedP4_fE",
        "GenFatJet/GenFatJet.TrimmedP4[5]",
    ],
    "tau1": ["GenFatJet/GenFatJet.Tau[0]", "GenFatJet/GenFatJet.Tau1"],
    "tau2": ["GenFatJet/GenFatJet.Tau[1]", "GenFatJet/GenFatJet.Tau2"],
    "tau3": ["GenFatJet/GenFatJet.Tau[2]", "GenFatJet/GenFatJet.Tau3"],
}


def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = calc_dphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def robust_range(x, lo=1.0, hi=99.0):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    a, b = np.percentile(x, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b):
        return None
    if a == b:
        pad = 1.0 if a == 0 else 0.1 * abs(a)
        a -= pad
        b += pad
    return a, b


def safe_hist_range(name, arrays):
    merged = []
    for arr in arrays:
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            merged.append(arr)

    if not merged:
        return None

    x = np.concatenate(merged)

    # Hand-tuned ranges for common jet observables
    lname = name.lower()
    if lname in ["eta"]:
        return (-4.0, 4.0)
    if lname in ["phi"]:
        return (-np.pi, np.pi)
    if lname in ["tau1", "tau2", "tau3", "tau21", "tau32"]:
        rr = robust_range(x, 0.5, 99.5)
        if rr is None:
            return None
        return (max(0.0, rr[0]), rr[1])
    if "mass" in lname or lname == "m":
        rr = robust_range(x, 0.5, 99.5)
        if rr is None:
            return None
        return (max(0.0, rr[0]), rr[1])
    if lname == "pt":
        rr = robust_range(x, 0.5, 99.5)
        if rr is None:
            return None
        return (max(0.0, rr[0]), rr[1])

    return robust_range(x, 1.0, 99.0)


def flatten_numeric(ak_array):
    out = ak.to_numpy(ak.flatten(ak_array, axis=None))
    return out[np.isfinite(out)]


def discover_first_existing(tree, candidates):
    keys = set(tree.keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def get_particle_axes(gen_pid, gen_eta, gen_phi, pid_list, skip_nonfinite=True):
    """
    Return eta/phi arrays for selected ancestor particles in one event.
    """
    pids = np.abs(ak.to_numpy(gen_pid))
    mask = np.isin(pids, pid_list)

    eta = ak.to_numpy(gen_eta)[mask]
    phi = ak.to_numpy(gen_phi)[mask]

    if skip_nonfinite:
        good = np.isfinite(eta) & np.isfinite(phi)
        eta = eta[good]
        phi = phi[good]

    return eta, phi


def closest_match_indices(src_eta, src_phi, trg_eta, trg_phi, dr_max):
    """
    For each source object, find closest target object within dr_max.
    Returns:
      matched_src_indices
      matched_trg_indices
      matched_dr
    Not one-to-one yet.
    """
    out_src = []
    out_trg = []
    out_dr = []

    if len(src_eta) == 0 or len(trg_eta) == 0:
        return out_src, out_trg, out_dr

    for i in range(len(src_eta)):
        drs = calc_dr(src_eta[i], src_phi[i], trg_eta, trg_phi)
        if len(drs) == 0:
            continue
        j = int(np.argmin(drs))
        dmin = float(drs[j])
        if dmin <= dr_max:
            out_src.append(i)
            out_trg.append(j)
            out_dr.append(dmin)

    return out_src, out_trg, out_dr


def greedy_unique_pairs(src_eta, src_phi, trg_eta, trg_phi, dr_max):
    """
    One-to-one greedy matching based on smallest dR pairs.
    Returns unique pairs (src_idx, trg_idx, dr).
    """
    candidates = []
    if len(src_eta) == 0 or len(trg_eta) == 0:
        return []

    for i in range(len(src_eta)):
        drs = calc_dr(src_eta[i], src_phi[i], trg_eta, trg_phi)
        for j, d in enumerate(drs):
            if d <= dr_max:
                candidates.append((float(d), i, j))

    candidates.sort(key=lambda x: x[0])

    used_src = set()
    used_trg = set()
    matched = []

    for d, i, j in candidates:
        if i in used_src or j in used_trg:
            continue
        used_src.add(i)
        used_trg.add(j)
        matched.append((i, j, d))

    return matched


def select_by_ancestor_match(jet_eta, jet_phi, anc_eta, anc_phi, dr_max):
    """
    Jet selected if matched to any ancestor axis within dr_max.
    Returns selected jet indices and their minimum dR-to-ancestor.
    """
    idxs = []
    drmins = []

    if len(jet_eta) == 0 or len(anc_eta) == 0:
        return idxs, drmins

    for i in range(len(jet_eta)):
        drs = calc_dr(jet_eta[i], jet_phi[i], anc_eta, anc_phi)
        if len(drs) == 0:
            continue
        dmin = float(np.min(drs))
        if dmin <= dr_max:
            idxs.append(i)
            drmins.append(dmin)

    return idxs, drmins


def add_values(store, prefix, jet_arrays, indices):
    """
    Append selected jet observable values into global store.
    Keeps only 1D numeric arrays and skips object / record branches safely.
    """
    if len(indices) == 0:
        return

    idx = np.asarray(indices, dtype=int)

    for k, arr in jet_arrays.items():
        full_key = f"{prefix}_{k}"
        if full_key not in store:
            store[full_key] = []

        try:
            vals = np.asarray(arr)

            # skip non-1D things
            if vals.ndim != 1:
                continue

            # skip non-numeric/object arrays
            if not np.issubdtype(vals.dtype, np.number):
                continue

            vals = vals[idx]
            vals = vals[np.isfinite(vals)]

            if len(vals):
                store[full_key].append(vals.astype(float, copy=False))

        except Exception:
            # silently skip weird branches
            continue


def finalize_store(store):
    out = {}
    for k, chunks in store.items():
        if len(chunks) == 0:
            out[k] = np.array([], dtype=float)
        else:
            out[k] = np.concatenate(chunks)
    return out


def compute_tau_ratios(obs):
    if "tau1" in obs and "tau2" in obs:
        tau1 = np.asarray(obs["tau1"], dtype=float)
        tau2 = np.asarray(obs["tau2"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            obs["tau21"] = np.where(tau1 > 0, tau2 / tau1, np.nan)

    if "tau2" in obs and "tau3" in obs:
        tau2 = np.asarray(obs["tau2"], dtype=float)
        tau3 = np.asarray(obs["tau3"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            obs["tau32"] = np.where(tau2 > 0, tau3 / tau2, np.nan)

    return obs


# =========================================================
# Plotting
# =========================================================
CATEGORY_STYLE = {
    "reco_genmatched":      {"label": "Reco jet, gen-matched",      "ls": "-",  "lw": 2.3},
    "gen_genmatched":       {"label": "Gen jet, gen-matched",       "ls": "--", "lw": 2.3},
    "reco_ancestormatched": {"label": "Reco jet, ancestor-matched", "ls": "-",  "lw": 1.8},
    "gen_ancestormatched":  {"label": "Gen jet, ancestor-matched",  "ls": "--", "lw": 1.8},
}


def plot_variable_scan_pdf(dr_results, var_name, outpdf, normalize=True, bins=60):
    dr_values = sorted(dr_results.keys())
    if len(dr_values) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, dr_cut in zip(axes, dr_values):
        data = dr_results[dr_cut]

        arrays = [
            data.get(f"reco_genmatched_{var_name}", np.array([])),
            data.get(f"gen_genmatched_{var_name}", np.array([])),
            data.get(f"reco_ancestormatched_{var_name}", np.array([])),
            data.get(f"gen_ancestormatched_{var_name}", np.array([])),
        ]
        hrange = safe_hist_range(var_name, arrays)
        if hrange is None:
            ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(fr"$\Delta R < {dr_cut:.1f}$")
            continue

        for cat, style in CATEGORY_STYLE.items():
            key = f"{cat}_{var_name}"
            x = np.asarray(data.get(key, np.array([])))
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue

            ax.hist(
                x,
                bins=bins,
                range=hrange,
                density=normalize,
                histtype="step",
                linewidth=style["lw"],
                linestyle=style["ls"],
                label=f'{style["label"]} (N={len(x)})'
            )

        ax.set_title(fr"$\Delta R < {dr_cut:.1f}$")
        ax.set_xlabel(var_name)
        ax.set_ylabel("Normalized jets" if normalize else "Jets")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)

    fig.suptitle(f"{var_name}: gen-match vs ancestor-match", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outpdf)
    plt.close(fig)


def plot_match_efficiency(summary, outpdf):
    drs = np.array(summary["dr"])
    fig, ax = plt.subplots(figsize=(9, 7))

    for key, label in [
        ("frac_reco_genmatched", "Reco matched to gen"),
        ("frac_gen_genmatched", "Gen matched to reco"),
        ("frac_reco_ancestormatched", "Reco matched to ancestor"),
        ("frac_gen_ancestormatched", "Gen matched to ancestor"),
    ]:
        ax.plot(drs, summary[key], marker="o", linewidth=2, label=label)

    ax.set_xlabel(r"Matching threshold $\Delta R_{\max}$")
    ax.set_ylabel("Matched fraction")
    ax.set_title("Matched jet fractions vs matching threshold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpdf)
    plt.close(fig)


def plot_mean_scan(dr_results, var_name, outpdf):
    drs = sorted(dr_results.keys())
    fig, ax = plt.subplots(figsize=(9, 7))

    for cat, style in CATEGORY_STYLE.items():
        means = []
        errs = []
        xs = []

        for dr in drs:
            x = np.asarray(dr_results[dr].get(f"{cat}_{var_name}", np.array([])))
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            xs.append(dr)
            means.append(np.mean(x))
            errs.append(np.std(x) / np.sqrt(len(x)))

        if len(xs):
            ax.errorbar(
                xs, means, yerr=errs,
                marker="o", linestyle=style["ls"], linewidth=2,
                label=style["label"]
            )

    ax.set_xlabel(r"Matching threshold $\Delta R_{\max}$")
    ax.set_ylabel(f"Mean {var_name}")
    ax.set_title(f"Mean {var_name} vs matching threshold")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpdf)
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Gen-matching study for SVJ jets")
    parser.add_argument(
        "--input",
        type=str,
        default="/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260331_1625/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-20.4_mrho-84.86_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root",
        help="Path to Delphes ROOT file"
    )
    parser.add_argument(
        "--tree",
        type=str,
        default="Delphes;1",
        help="TTree path"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="genmatch_study_scan",
        help="Output directory"
    )
    parser.add_argument(
        "--dr-values",
        type=float,
        nargs="+",
        default=DEFAULT_DR_VALUES,
        help="List of dR thresholds to scan"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=-1,
        help="Max events to process (-1 means all)"
    )
    parser.add_argument(
        "--ancestor-mode",
        type=str,
        default="dark_quark",
        choices=["dark_quark", "dark_hadron"],
        help="Which ancestor axis to use for ancestor matching"
    )
    parser.add_argument(
        "--min-reco-pt",
        type=float,
        default=0.0,
        help="Optional reco fatjet pT cut"
    )
    parser.add_argument(
        "--min-gen-pt",
        type=float,
        default=0.0,
        help="Optional gen fatjet pT cut"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Opening {args.input}")
    with uproot.open(args.input) as f:
        tree = f[args.tree]
        keys = set(tree.keys())

        # -------------------------
        # Required branches
        # -------------------------
        req = [
            "GenParticle/GenParticle.PID",
            "GenParticle/GenParticle.Eta",
            "GenParticle/GenParticle.Phi",
            "GenFatJet/GenFatJet.PT",
            "GenFatJet/GenFatJet.Eta",
            "GenFatJet/GenFatJet.Phi",
            "GenFatJet/GenFatJet.Mass",
            "FatJet/FatJet.PT",
            "FatJet/FatJet.Eta",
            "FatJet/FatJet.Phi",
            "FatJet/FatJet.Mass",
        ]
        missing = [b for b in req if b not in keys]
        if missing:
            raise RuntimeError(f"Missing required branches:\n  " + "\n  ".join(missing))

        # -------------------------
        # Load core arrays
        # -------------------------
        print("Loading core branches...")
        gen_pid = tree["GenParticle/GenParticle.PID"].array()
        gen_eta = tree["GenParticle/GenParticle.Eta"].array()
        gen_phi = tree["GenParticle/GenParticle.Phi"].array()

        gf_pt   = tree["GenFatJet/GenFatJet.PT"].array()
        gf_eta  = tree["GenFatJet/GenFatJet.Eta"].array()
        gf_phi  = tree["GenFatJet/GenFatJet.Phi"].array()
        gf_mass = tree["GenFatJet/GenFatJet.Mass"].array()

        rf_pt   = tree["FatJet/FatJet.PT"].array()
        rf_eta  = tree["FatJet/FatJet.Eta"].array()
        rf_phi  = tree["FatJet/FatJet.Phi"].array()
        rf_mass = tree["FatJet/FatJet.Mass"].array()

        # -------------------------
        # Load optional branches
        # -------------------------
        reco_optional = {}
        gen_optional = {}

        for outname, candidates in RECO_OPTIONAL_BRANCHES.items():
            found = discover_first_existing(tree, candidates)
            if found is not None:
                reco_optional[outname] = tree[found].array()
                print(f"[Reco optional] {outname:>8s} <- {found}")

        for outname, candidates in GEN_OPTIONAL_BRANCHES.items():
            found = discover_first_existing(tree, candidates)
            if found is not None:
                gen_optional[outname] = tree[found].array()
                print(f"[Gen  optional] {outname:>8s} <- {found}")

    n_events = len(rf_pt)
    if args.max_events > 0:
        n_events = min(n_events, args.max_events)

    print(f"Processing {n_events} events")

    ancestor_pid_list = DARK_QUARKS if args.ancestor_mode == "dark_quark" else DARK_HADRONS
    print(f"Ancestor mode = {args.ancestor_mode}")
    print(f"dR scan = {args.dr_values}")

    dr_results = {}
    summary = {
        "dr": [],
        "frac_reco_genmatched": [],
        "frac_gen_genmatched": [],
        "frac_reco_ancestormatched": [],
        "frac_gen_ancestormatched": [],
    }

    # =====================================================
    # Loop over dR thresholds
    # =====================================================
    for dr_cut in sorted(args.dr_values):
        print(f"\n=== Running dR threshold {dr_cut:.2f} ===")

        store = {}

        total_reco_jets = 0
        total_gen_jets = 0

        matched_reco_gen = 0
        matched_gen_gen = 0
        matched_reco_anc = 0
        matched_gen_anc = 0

        for iev in range(n_events):
            # ---------- event arrays ----------
            # reco
            reco_obs = {
                "pt":   ak.to_numpy(rf_pt[iev]),
                "eta":  ak.to_numpy(rf_eta[iev]),
                "phi":  ak.to_numpy(rf_phi[iev]),
                "mass": ak.to_numpy(rf_mass[iev]),
            }
            for name, arr in reco_optional.items():
                reco_obs[name] = ak.to_numpy(arr[iev])
            reco_obs = compute_tau_ratios(reco_obs)

            # gen
            gen_obs = {
                "pt":   ak.to_numpy(gf_pt[iev]),
                "eta":  ak.to_numpy(gf_eta[iev]),
                "phi":  ak.to_numpy(gf_phi[iev]),
                "mass": ak.to_numpy(gf_mass[iev]),
            }
            for name, arr in gen_optional.items():
                gen_obs[name] = ak.to_numpy(arr[iev])
            gen_obs = compute_tau_ratios(gen_obs)

            # optional pt cuts
            reco_keep = np.ones(len(reco_obs["pt"]), dtype=bool)
            gen_keep  = np.ones(len(gen_obs["pt"]), dtype=bool)

            if args.min_reco_pt > 0:
                reco_keep &= (reco_obs["pt"] >= args.min_reco_pt)
            if args.min_gen_pt > 0:
                gen_keep &= (gen_obs["pt"] >= args.min_gen_pt)

            for k in reco_obs:
                reco_obs[k] = np.asarray(reco_obs[k])[reco_keep]
            for k in gen_obs:
                gen_obs[k] = np.asarray(gen_obs[k])[gen_keep]

            total_reco_jets += len(reco_obs["pt"])
            total_gen_jets += len(gen_obs["pt"])

            # ---------- ancestor axes ----------
            anc_eta, anc_phi = get_particle_axes(
                gen_pid[iev], gen_eta[iev], gen_phi[iev], ancestor_pid_list
            )

            # ---------- one-to-one reco<->gen matching ----------
            pairs = greedy_unique_pairs(
                gen_obs["eta"], gen_obs["phi"],
                reco_obs["eta"], reco_obs["phi"],
                dr_cut
            )

            gen_pair_idx  = [p[0] for p in pairs]
            reco_pair_idx = [p[1] for p in pairs]

            matched_gen_gen += len(gen_pair_idx)
            matched_reco_gen += len(reco_pair_idx)

            add_values(store, "gen_genmatched", gen_obs, gen_pair_idx)
            add_values(store, "reco_genmatched", reco_obs, reco_pair_idx)

            # ---------- ancestor matching ----------
            gen_anc_idx, _ = select_by_ancestor_match(
                gen_obs["eta"], gen_obs["phi"], anc_eta, anc_phi, dr_cut
            )
            reco_anc_idx, _ = select_by_ancestor_match(
                reco_obs["eta"], reco_obs["phi"], anc_eta, anc_phi, dr_cut
            )

            matched_gen_anc += len(gen_anc_idx)
            matched_reco_anc += len(reco_anc_idx)

            add_values(store, "gen_ancestormatched", gen_obs, gen_anc_idx)
            add_values(store, "reco_ancestormatched", reco_obs, reco_anc_idx)

            if iev % 5000 == 0 and iev > 0:
                print(f"  processed {iev}/{n_events}")

        dr_results[dr_cut] = finalize_store(store)

        frac_reco_gen = matched_reco_gen / total_reco_jets if total_reco_jets > 0 else 0.0
        frac_gen_gen  = matched_gen_gen  / total_gen_jets  if total_gen_jets  > 0 else 0.0
        frac_reco_anc = matched_reco_anc / total_reco_jets if total_reco_jets > 0 else 0.0
        frac_gen_anc  = matched_gen_anc  / total_gen_jets  if total_gen_jets  > 0 else 0.0

        summary["dr"].append(dr_cut)
        summary["frac_reco_genmatched"].append(frac_reco_gen)
        summary["frac_gen_genmatched"].append(frac_gen_gen)
        summary["frac_reco_ancestormatched"].append(frac_reco_anc)
        summary["frac_gen_ancestormatched"].append(frac_gen_anc)

        print(
            f"  Fractions: "
            f"Reco-gen={frac_reco_gen:.4f}, "
            f"Gen-gen={frac_gen_gen:.4f}, "
            f"Reco-anc={frac_reco_anc:.4f}, "
            f"Gen-anc={frac_gen_anc:.4f}"
        )

    # =====================================================
    # Save summary text
    # =====================================================
    txt_path = os.path.join(args.outdir, "matching_summary.txt")
    with open(txt_path, "w") as fout:
        fout.write(f"Input file: {args.input}\n")
        fout.write(f"Tree: {args.tree}\n")
        fout.write(f"Ancestor mode: {args.ancestor_mode}\n")
        fout.write(f"Events processed: {n_events}\n")
        fout.write(f"dR values: {sorted(args.dr_values)}\n\n")

        fout.write(
            "dr_cut  frac_reco_genmatched  frac_gen_genmatched  "
            "frac_reco_ancestormatched  frac_gen_ancestormatched\n"
        )
        for i, dr in enumerate(summary["dr"]):
            fout.write(
                f"{dr:0.2f}  "
                f"{summary['frac_reco_genmatched'][i]:0.6f}  "
                f"{summary['frac_gen_genmatched'][i]:0.6f}  "
                f"{summary['frac_reco_ancestormatched'][i]:0.6f}  "
                f"{summary['frac_gen_ancestormatched'][i]:0.6f}\n"
            )

    # =====================================================
    # Determine which variables exist
    # =====================================================
    example_dr = sorted(dr_results.keys())[0]
    all_keys = dr_results[example_dr].keys()

    variables = sorted(list(set(k.split("_", 2)[-1] for k in all_keys)))
    print("\nVariables found in result store:")
    for v in variables:
        print("  ", v)

    # =====================================================
    # Make PDFs
    # =====================================================
    eff_pdf = os.path.join(args.outdir, "matching_efficiency_vs_dr.pdf")
    plot_match_efficiency(summary, eff_pdf)

    for var in variables:
        outpdf = os.path.join(args.outdir, f"{var}_overlay_scan.pdf")
        plot_variable_scan_pdf(dr_results, var, outpdf, normalize=True, bins=60)

        meanpdf = os.path.join(args.outdir, f"{var}_mean_vs_dr.pdf")
        plot_mean_scan(dr_results, var, meanpdf)

    # =====================================================
    # One combined multipage PDF
    # =====================================================
    combined_pdf = os.path.join(args.outdir, "all_variables_combined.pdf")
    with PdfPages(combined_pdf) as pdf:
        # efficiency page
        fig, ax = plt.subplots(figsize=(9, 7))
        drs = np.array(summary["dr"])
        for key, label in [
            ("frac_reco_genmatched", "Reco matched to gen"),
            ("frac_gen_genmatched", "Gen matched to reco"),
            ("frac_reco_ancestormatched", "Reco matched to ancestor"),
            ("frac_gen_ancestormatched", "Gen matched to ancestor"),
        ]:
            ax.plot(drs, summary[key], marker="o", linewidth=2, label=label)
        ax.set_xlabel(r"Matching threshold $\Delta R_{\max}$")
        ax.set_ylabel("Matched fraction")
        ax.set_title("Matched jet fractions vs matching threshold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # overlay pages
        for var in variables:
            dr_values = sorted(dr_results.keys())
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            for ax, dr_cut in zip(axes, dr_values):
                data = dr_results[dr_cut]
                arrays = [
                    data.get(f"reco_genmatched_{var}", np.array([])),
                    data.get(f"gen_genmatched_{var}", np.array([])),
                    data.get(f"reco_ancestormatched_{var}", np.array([])),
                    data.get(f"gen_ancestormatched_{var}", np.array([])),
                ]
                hrange = safe_hist_range(var, arrays)
                if hrange is None:
                    ax.text(0.5, 0.5, "No entries", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(fr"$\Delta R < {dr_cut:.1f}$")
                    continue

                for cat, style in CATEGORY_STYLE.items():
                    key = f"{cat}_{var}"
                    x = np.asarray(data.get(key, np.array([])))
                    x = x[np.isfinite(x)]
                    if len(x) == 0:
                        continue
                    ax.hist(
                        x,
                        bins=60,
                        range=hrange,
                        density=True,
                        histtype="step",
                        linewidth=style["lw"],
                        linestyle=style["ls"],
                        label=f'{style["label"]} (N={len(x)})'
                    )
                ax.set_title(fr"{var}, $\Delta R < {dr_cut:.1f}$")
                ax.set_xlabel(var)
                ax.set_ylabel("Normalized jets")
                ax.grid(alpha=0.25)

            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
            fig.suptitle(f"{var}: gen-match vs ancestor-match", fontsize=16, y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print("\nDone.")
    print(f"Output directory: {args.outdir}")
    print(f"Summary text:      {txt_path}")
    print(f"Combined PDF:      {combined_pdf}")


if __name__ == "__main__":
    main()