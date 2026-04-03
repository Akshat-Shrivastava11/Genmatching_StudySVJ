#!/usr/bin/env python3
import sys
import os
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/SVJ_Training_2D_20260331_1625/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-10.8_mrho-87.63_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root"
PLOT_DIR = "plots_fullymatched_study"
os.makedirs(PLOT_DIR, exist_ok=True)

FATJET_R = 0.8
MAX_EVENTS = 2000
MAX_SAMPLES_PER_CAT = 50

# Dark sector IDs
DARK_HADRONS = {4900111, 4900211, -4900211, 4900113, 4900213, -4900213}
DARK_QUARKS  = {
    4900101, 4900102, 4900103, 4900104, 4900105, 4900106,
    -4900101, -4900102, -4900103, -4900104, -4900105, -4900106
}

# Delphes GenParticle.Status==1 usually means final state
INVISIBLE_PIDS_ABS = {12, 14, 16, 18, 4900101, 4900102, 4900103, 4900104, 4900105, 4900106}


def calc_dphi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = calc_dphi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def wrap_phi_np(dphi):
    dphi = np.array(dphi, dtype=float)
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def is_visible_final_state(pid, status):
    apid = abs(int(pid))
    if int(status) != 1:
        return False
    if apid in INVISIBLE_PIDS_ABS:
        return False
    return True


def get_descendants(idx, d1s, d2s, n_parts, visited=None):
    """Return all descendants of particle idx recursively."""
    if visited is None:
        visited = set()
    out = []

    if idx < 0 or idx >= n_parts or idx in visited:
        return out
    visited.add(idx)

    d1 = int(d1s[idx])
    d2 = int(d2s[idx])

    if d1 < 0:
        return out

    if d2 < d1:
        d2 = d1

    for child in range(d1, d2 + 1):
        if child < 0 or child >= n_parts:
            continue
        out.append(child)
        out.extend(get_descendants(child, d1s, d2s, n_parts, visited))

    return out


def match_dark_hadron_to_best_genfatjet(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt):
    """Match a dark hadron to nearest GenFatJet within cone."""
    if len(gf_eta_evt) == 0:
        return None, None
    drs = calc_dr(dh_eta, dh_phi, gf_eta_evt, gf_phi_evt)
    best = int(np.argmin(drs))
    best_dr = float(drs[best])
    if best_dr < FATJET_R:
        return best, best_dr
    return None, best_dr


def safe_hist(ax, data, bins=50, range=None, label=None, density=False, histtype="step", lw=2):
    data = np.asarray(data)
    if len(data) == 0:
        return
    ax.hist(data, bins=bins, range=range, label=label, density=density, histtype=histtype, linewidth=lw)


def choose_representative_sample(samples, mode):
    if len(samples) == 0:
        return None
    if mode == "full":
        # prefer many descendants
        return max(samples, key=lambda s: (s["n_vis"], s["gf_pt"]))
    if mode == "partial":
        # prefer ambiguous containment near 50%
        return min(samples, key=lambda s: abs(s["frac_inside"] - 0.5))
    if mode == "none":
        # prefer many descendants and energetic jet
        return max(samples, key=lambda s: (s["n_vis"], s["gf_pt"]))
    return samples[0]


def plot_3d_match_category(sample, category_label, outpath, fatjet_r=0.8):
    """
    3D plot in centered jet coordinates:
      x = Δη
      y = Δφ
      z = pT
    """
    gf_eta = float(sample["gf_eta"])
    gf_phi = float(sample["gf_phi"])
    gf_pt  = float(sample["gf_pt"])

    desc_eta = np.array(sample["desc_eta"], dtype=float)
    desc_phi = np.array(sample["desc_phi"], dtype=float)
    desc_pt  = np.array(sample["desc_pt"], dtype=float)
    inside_mask = np.array(sample["inside_mask"], dtype=bool)

    d_eta = desc_eta - gf_eta
    d_phi = wrap_phi_np(desc_phi - gf_phi)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2 * np.pi, 300)
    cone_x = fatjet_r * np.cos(theta)
    cone_y = fatjet_r * np.sin(theta)
    cone_z = np.zeros_like(theta)

    ax.plot(cone_x, cone_y, cone_z, color="black", linestyle="--", linewidth=2.0, label=f"Jet cone R={fatjet_r}")

    for x, y, z, inside in zip(d_eta, d_phi, desc_pt, inside_mask):
        ax.plot([x, x], [y, y], [0, z],
                color="green" if inside else "red",
                alpha=0.25, linewidth=1.0)

    if np.any(inside_mask):
        ax.scatter(
            d_eta[inside_mask], d_phi[inside_mask], desc_pt[inside_mask],
            s=np.clip(desc_pt[inside_mask] * 2.0, 20, 250),
            c="green", alpha=0.85,
            edgecolors="black", linewidths=0.4,
            label="Visible descendants inside cone"
        )

    if np.any(~inside_mask):
        ax.scatter(
            d_eta[~inside_mask], d_phi[~inside_mask], desc_pt[~inside_mask],
            s=np.clip(desc_pt[~inside_mask] * 2.0, 20, 250),
            c="red", alpha=0.75,
            edgecolors="black", linewidths=0.4,
            label="Visible descendants outside cone"
        )

    ax.scatter(
        [0], [0], [0],
        c="blue", s=120, marker="X",
        edgecolors="black", linewidths=0.8,
        label=f"Matched GenFatJet axis\n$p_T$={gf_pt:.1f} GeV"
    )

    ax.set_xlabel(r"$\Delta \eta$")
    ax.set_ylabel(r"$\Delta \phi$")
    ax.set_zlabel(r"$p_T$ [GeV]")
    ax.set_title(f"{category_label}: Dark Hadron Visible Descendants\nCentered on Matched GenFatJet Axis")

    xy_lim = max(
        fatjet_r + 0.25,
        np.max(np.abs(d_eta)) if len(d_eta) else 1.0,
        np.max(np.abs(d_phi)) if len(d_phi) else 1.0
    )
    ax.set_xlim(-xy_lim, xy_lim)
    ax.set_ylim(-xy_lim, xy_lim)

    zmax = max(20.0, np.max(desc_pt) * 1.15 if len(desc_pt) else 20.0)
    ax.set_zlim(0, zmax)

    ax.view_init(elev=24, azim=-56)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_3d_match_gallery(full_samples, partial_samples, none_samples, outpath, fatjet_r=0.8):
    """
    Side-by-side 3D comparison:
      Fully matched | Partially matched | Unmatched
    """
    chosen = [
        ("Fully matched", choose_representative_sample(full_samples, "full"), "green"),
        ("Partially matched", choose_representative_sample(partial_samples, "partial"), "orange"),
        ("Unmatched", choose_representative_sample(none_samples, "none"), "red"),
    ]

    fig = plt.figure(figsize=(18, 6))

    for i, (label, sample, color_hint) in enumerate(chosen, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")

        if sample is None:
            ax.set_title(f"{label}\n(no sample found)")
            continue

        gf_eta = float(sample["gf_eta"])
        gf_phi = float(sample["gf_phi"])
        gf_pt  = float(sample["gf_pt"])

        desc_eta = np.array(sample["desc_eta"], dtype=float)
        desc_phi = np.array(sample["desc_phi"], dtype=float)
        desc_pt  = np.array(sample["desc_pt"], dtype=float)
        inside_mask = np.array(sample["inside_mask"], dtype=bool)

        d_eta = desc_eta - gf_eta
        d_phi = wrap_phi_np(desc_phi - gf_phi)

        theta = np.linspace(0, 2 * np.pi, 300)
        cone_x = fatjet_r * np.cos(theta)
        cone_y = fatjet_r * np.sin(theta)
        cone_z = np.zeros_like(theta)
        ax.plot(cone_x, cone_y, cone_z, color="black", linestyle="--", linewidth=1.8)

        for x, y, z, inside in zip(d_eta, d_phi, desc_pt, inside_mask):
            ax.plot([x, x], [y, y], [0, z],
                    color="green" if inside else "red",
                    alpha=0.22, linewidth=0.9)

        if np.any(inside_mask):
            ax.scatter(
                d_eta[inside_mask], d_phi[inside_mask], desc_pt[inside_mask],
                s=np.clip(desc_pt[inside_mask] * 2.0, 20, 220),
                c="green", alpha=0.85, edgecolors="black", linewidths=0.35
            )
        if np.any(~inside_mask):
            ax.scatter(
                d_eta[~inside_mask], d_phi[~inside_mask], desc_pt[~inside_mask],
                s=np.clip(desc_pt[~inside_mask] * 2.0, 20, 220),
                c="red", alpha=0.75, edgecolors="black", linewidths=0.35
            )

        ax.scatter([0], [0], [0], c="blue", s=100, marker="X")

        ax.set_title(f"{label}\nJet $p_T$={gf_pt:.1f} GeV", color=color_hint)
        ax.set_xlabel(r"$\Delta \eta$")
        ax.set_ylabel(r"$\Delta \phi$")
        ax.set_zlabel(r"$p_T$ [GeV]")

        xy_lim = max(
            fatjet_r + 0.25,
            np.max(np.abs(d_eta)) if len(d_eta) else 1.0,
            np.max(np.abs(d_phi)) if len(d_phi) else 1.0
        )
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)

        zmax = max(20.0, np.max(desc_pt) * 1.15 if len(desc_pt) else 20.0)
        ax.set_zlim(0, zmax)

        ax.view_init(elev=24, azim=-56)
        ax.grid(True, alpha=0.2)

    plt.suptitle("3D Comparison of Dark-Hadron Descendant Containment Categories", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_3d_category_gallery(samples, category_label, outpath, fatjet_r=0.8, n_show=9):
    """
    Make a gallery of 3D plots for one category.
    """
    n_show = min(n_show, len(samples))
    if n_show == 0:
        print(f"[WARNING] No samples available for {category_label}")
        return

    ncols = 3
    nrows = int(np.ceil(n_show / ncols))

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows))

    for i in range(n_show):
        sample = samples[i]
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")

        gf_eta = float(sample["gf_eta"])
        gf_phi = float(sample["gf_phi"])
        gf_pt  = float(sample["gf_pt"])

        desc_eta = np.array(sample["desc_eta"], dtype=float)
        desc_phi = np.array(sample["desc_phi"], dtype=float)
        desc_pt  = np.array(sample["desc_pt"], dtype=float)
        inside_mask = np.array(sample["inside_mask"], dtype=bool)

        d_eta = desc_eta - gf_eta
        d_phi = wrap_phi_np(desc_phi - gf_phi)

        theta = np.linspace(0, 2 * np.pi, 300)
        cone_x = fatjet_r * np.cos(theta)
        cone_y = fatjet_r * np.sin(theta)
        cone_z = np.zeros_like(theta)
        ax.plot(cone_x, cone_y, cone_z, color="black", linestyle="--", linewidth=1.4)

        for x, y, z, inside in zip(d_eta, d_phi, desc_pt, inside_mask):
            ax.plot(
                [x, x], [y, y], [0, z],
                color="green" if inside else "red",
                alpha=0.18, linewidth=0.8
            )

        if np.any(inside_mask):
            ax.scatter(
                d_eta[inside_mask], d_phi[inside_mask], desc_pt[inside_mask],
                s=np.clip(desc_pt[inside_mask] * 1.8, 18, 180),
                c="green", alpha=0.8, edgecolors="black", linewidths=0.25
            )

        if np.any(~inside_mask):
            ax.scatter(
                d_eta[~inside_mask], d_phi[~inside_mask], desc_pt[~inside_mask],
                s=np.clip(desc_pt[~inside_mask] * 1.8, 18, 180),
                c="red", alpha=0.75, edgecolors="black", linewidths=0.25
            )

        ax.scatter([0], [0], [0], c="blue", s=70, marker="X")

        frac_inside = sample["frac_inside"]
        n_vis = sample["n_vis"]

        ax.set_title(
            f"{category_label}\n"
            f"jet $p_T$={gf_pt:.0f} GeV, "
            f"$f_{{in}}$={frac_inside:.2f}, "
            f"$N_{{vis}}$={n_vis}",
            fontsize=10
        )

        xy_lim = max(
            fatjet_r + 0.25,
            np.max(np.abs(d_eta)) if len(d_eta) else 1.0,
            np.max(np.abs(d_phi)) if len(d_phi) else 1.0
        )
        ax.set_xlim(-xy_lim, xy_lim)
        ax.set_ylim(-xy_lim, xy_lim)

        zmax = max(20.0, np.max(desc_pt) * 1.15 if len(desc_pt) else 20.0)
        ax.set_zlim(0, zmax)

        ax.set_xlabel(r"$\Delta \eta$", fontsize=9)
        ax.set_ylabel(r"$\Delta \phi$", fontsize=9)
        ax.set_zlabel(r"$p_T$", fontsize=9)
        ax.view_init(elev=24, azim=-56)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"3D Gallery: {category_label}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
print(f"Opening {ROOT_FILE}...")
with uproot.open(ROOT_FILE) as f:
    tree = f["Delphes;1"]

    print("Loading GenParticles...")
    gen_pid    = tree["GenParticle/GenParticle.PID"].array()
    gen_eta    = tree["GenParticle/GenParticle.Eta"].array()
    gen_phi    = tree["GenParticle/GenParticle.Phi"].array()
    gen_pt     = tree["GenParticle/GenParticle.PT"].array()
    gen_mass   = tree["GenParticle/GenParticle.Mass"].array()
    gen_status = tree["GenParticle/GenParticle.Status"].array()
    gen_m1     = tree["GenParticle/GenParticle.M1"].array()
    gen_d1     = tree["GenParticle/GenParticle.D1"].array()
    gen_d2     = tree["GenParticle/GenParticle.D2"].array()

    print("Loading GenFatJets...")
    gf_pt   = tree["GenFatJet/GenFatJet.PT"].array()
    gf_eta  = tree["GenFatJet/GenFatJet.Eta"].array()
    gf_phi  = tree["GenFatJet/GenFatJet.Phi"].array()
    gf_mass = tree["GenFatJet/GenFatJet.Mass"].array()

    print("Loading Reco FatJets...")
    rf_pt   = tree["FatJet/FatJet.PT"].array()
    rf_eta  = tree["FatJet/FatJet.Eta"].array()
    rf_phi  = tree["FatJet/FatJet.Phi"].array()
    rf_mass = tree["FatJet/FatJet.Mass"].array()

num_events = min(MAX_EVENTS, len(gf_pt))
print(f"Analyzing {num_events} events...")

# Storage
cat_counts = {"full": 0, "partial": 0, "none": 0, "no_genfatjet_match": 0, "no_visible_desc": 0}

full_dh_mass, partial_dh_mass, none_dh_mass = [], [], []
full_gf_mass, partial_gf_mass, none_gf_mass = [], [], []
full_gf_pt, partial_gf_pt, none_gf_pt = [], [], []
full_frac, partial_frac, none_frac = [], [], []
full_nvis, partial_nvis, none_nvis = [], [], []
bestdr_all = []

# These must be OUTSIDE the loop
full_samples = []
partial_samples = []
none_samples = []

for ev in range(num_events):
    pids_evt    = ak.to_numpy(gen_pid[ev])
    eta_evt     = ak.to_numpy(gen_eta[ev])
    phi_evt     = ak.to_numpy(gen_phi[ev])
    pt_evt      = ak.to_numpy(gen_pt[ev])
    mass_evt    = ak.to_numpy(gen_mass[ev])
    status_evt  = ak.to_numpy(gen_status[ev])
    d1_evt      = ak.to_numpy(gen_d1[ev])
    d2_evt      = ak.to_numpy(gen_d2[ev])

    gf_eta_evt  = ak.to_numpy(gf_eta[ev])
    gf_phi_evt  = ak.to_numpy(gf_phi[ev])
    gf_pt_evt   = ak.to_numpy(gf_pt[ev])
    gf_mass_evt = ak.to_numpy(gf_mass[ev])

    dh_indices = np.where(np.isin(pids_evt, list(DARK_HADRONS)))[0]
    if len(dh_indices) == 0:
        continue

    for dh_idx in dh_indices:
        dh_eta = eta_evt[dh_idx]
        dh_phi = phi_evt[dh_idx]
        dh_mass = mass_evt[dh_idx]

        if np.isnan(dh_eta) or np.isnan(dh_phi) or np.isinf(dh_eta) or np.isinf(dh_phi):
            continue

        best_gf_idx, best_dr = match_dark_hadron_to_best_genfatjet(
            dh_eta, dh_phi, gf_eta_evt, gf_phi_evt
        )
        if best_dr is not None:
            bestdr_all.append(best_dr)

        desc = sorted(set(get_descendants(dh_idx, d1_evt, d2_evt, len(pids_evt))))
        if len(desc) == 0:
            continue

        vis_desc = []
        for j in desc:
            if is_visible_final_state(pids_evt[j], status_evt[j]):
                if not (
                    np.isnan(eta_evt[j]) or np.isnan(phi_evt[j]) or
                    np.isinf(eta_evt[j]) or np.isinf(phi_evt[j])
                ):
                    vis_desc.append(j)

        if len(vis_desc) == 0:
            cat_counts["no_visible_desc"] += 1
            continue

        if best_gf_idx is None:
            cat_counts["no_genfatjet_match"] += 1
            continue

        drs_desc = calc_dr(
            eta_evt[vis_desc],
            phi_evt[vis_desc],
            gf_eta_evt[best_gf_idx],
            gf_phi_evt[best_gf_idx],
        )

        inside = drs_desc < FATJET_R
        n_inside = int(np.sum(inside))
        n_total = len(vis_desc)
        frac_inside = n_inside / max(n_total, 1)

        if n_inside == n_total:
            category = "full"
        elif n_inside > 0:
            category = "partial"
        else:
            category = "none"

        cat_counts[category] += 1

        sample = {
            "event": ev,
            "dh_idx": int(dh_idx),
            "gf_idx": int(best_gf_idx),
            "gf_eta": float(gf_eta_evt[best_gf_idx]),
            "gf_phi": float(gf_phi_evt[best_gf_idx]),
            "gf_pt": float(gf_pt_evt[best_gf_idx]),
            "gf_mass": float(gf_mass_evt[best_gf_idx]),
            "dh_mass": float(dh_mass),
            "desc_eta": eta_evt[vis_desc].astype(float),
            "desc_phi": phi_evt[vis_desc].astype(float),
            "desc_pt": pt_evt[vis_desc].astype(float),
            "inside_mask": inside.astype(bool),
            "frac_inside": float(frac_inside),
            "n_vis": int(n_total),
        }

        if category == "full":
            full_dh_mass.append(dh_mass)
            full_gf_mass.append(gf_mass_evt[best_gf_idx])
            full_gf_pt.append(gf_pt_evt[best_gf_idx])
            full_frac.append(frac_inside)
            full_nvis.append(n_total)
            if len(full_samples) < MAX_SAMPLES_PER_CAT:
                full_samples.append(sample)

        elif category == "partial":
            partial_dh_mass.append(dh_mass)
            partial_gf_mass.append(gf_mass_evt[best_gf_idx])
            partial_gf_pt.append(gf_pt_evt[best_gf_idx])
            partial_frac.append(frac_inside)
            partial_nvis.append(n_total)
            if len(partial_samples) < MAX_SAMPLES_PER_CAT:
                partial_samples.append(sample)

        else:
            none_dh_mass.append(dh_mass)
            none_gf_mass.append(gf_mass_evt[best_gf_idx])
            none_gf_pt.append(gf_pt_evt[best_gf_idx])
            none_frac.append(frac_inside)
            none_nvis.append(n_total)
            if len(none_samples) < MAX_SAMPLES_PER_CAT:
                none_samples.append(sample)

print("\nCategory counts:")
for k, v in cat_counts.items():
    print(f"  {k:18s}: {v}")

print(f"\nStored samples for 3D plots:")
print(f"  full_samples    = {len(full_samples)}")
print(f"  partial_samples = {len(partial_samples)}")
print(f"  none_samples    = {len(none_samples)}")

# 1) Dark hadron mass distribution by category
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, full_dh_mass,    bins=50, label="Fully matched", density=True)
safe_hist(ax, partial_dh_mass, bins=50, label="Partially matched", density=True)
safe_hist(ax, none_dh_mass,    bins=50, label="Unmatched", density=True)
ax.set_xlabel("Dark hadron mass [GeV]")
ax.set_ylabel("Normalized entries")
ax.set_title("Dark Hadron Mass: Full vs Partial vs None")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/darkhadron_mass_by_match_category.pdf")
plt.close(fig)

# 2) Matched GenFatJet mass by category
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, full_gf_mass,    bins=60, label="Fully matched", density=True)
safe_hist(ax, partial_gf_mass, bins=60, label="Partially matched", density=True)
safe_hist(ax, none_gf_mass,    bins=60, label="Unmatched", density=True)
ax.set_xlabel("Matched GenFatJet mass [GeV]")
ax.set_ylabel("Normalized entries")
ax.set_title("Matched GenFatJet Mass by Category")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/genfatjet_mass_by_match_category.pdf")
plt.close(fig)

# 3) Matched GenFatJet pT by category
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, full_gf_pt,    bins=60, label="Fully matched", density=True)
safe_hist(ax, partial_gf_pt, bins=60, label="Partially matched", density=True)
safe_hist(ax, none_gf_pt,    bins=60, label="Unmatched", density=True)
ax.set_xlabel("Matched GenFatJet pT [GeV]")
ax.set_ylabel("Normalized entries")
ax.set_title("Matched GenFatJet pT by Category")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/genfatjet_pt_by_match_category.pdf")
plt.close(fig)

# 4) Fraction of visible descendants inside matched jet
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, full_frac,    bins=20, range=(0, 1.05), label="Fully matched", density=True)
safe_hist(ax, partial_frac, bins=20, range=(0, 1.05), label="Partially matched", density=True)
safe_hist(ax, none_frac,    bins=20, range=(0, 1.05), label="Unmatched", density=True)
ax.set_xlabel("Fraction of visible descendants inside matched GenFatJet")
ax.set_ylabel("Normalized entries")
ax.set_title("Containment Fraction")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/containment_fraction.pdf")
plt.close(fig)

# 5) Number of visible descendants
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, full_nvis,    bins=np.arange(0, 21) - 0.5, label="Fully matched", density=True)
safe_hist(ax, partial_nvis, bins=np.arange(0, 21) - 0.5, label="Partially matched", density=True)
safe_hist(ax, none_nvis,    bins=np.arange(0, 21) - 0.5, label="Unmatched", density=True)
ax.set_xlabel("Number of visible stable descendants of dark hadron")
ax.set_ylabel("Normalized entries")
ax.set_title("Visible Descendant Multiplicity")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/visible_descendant_multiplicity.pdf")
plt.close(fig)

# 6) Side-by-side comparison: full vs not-full
notfull_dh_mass = partial_dh_mass + none_dh_mass
notfull_gf_mass = partial_gf_mass + none_gf_mass
notfull_gf_pt   = partial_gf_pt + none_gf_pt
notfull_frac    = partial_frac + none_frac

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

safe_hist(axes[0, 0], full_dh_mass,    bins=50, label="Fully matched", density=True)
safe_hist(axes[0, 0], notfull_dh_mass, bins=50, label="Partial/Unmatched", density=True)
axes[0, 0].set_xlabel("Dark hadron mass [GeV]")
axes[0, 0].set_ylabel("Normalized entries")
axes[0, 0].set_title("Dark Hadron Mass")
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

safe_hist(axes[0, 1], full_gf_mass,    bins=60, label="Fully matched", density=True)
safe_hist(axes[0, 1], notfull_gf_mass, bins=60, label="Partial/Unmatched", density=True)
axes[0, 1].set_xlabel("Matched GenFatJet mass [GeV]")
axes[0, 1].set_ylabel("Normalized entries")
axes[0, 1].set_title("Matched GenFatJet Mass")
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

safe_hist(axes[1, 0], full_gf_pt,    bins=60, label="Fully matched", density=True)
safe_hist(axes[1, 0], notfull_gf_pt, bins=60, label="Partial/Unmatched", density=True)
axes[1, 0].set_xlabel("Matched GenFatJet pT [GeV]")
axes[1, 0].set_ylabel("Normalized entries")
axes[1, 0].set_title("Matched GenFatJet pT")
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

safe_hist(axes[1, 1], full_frac,    bins=20, range=(0, 1.05), label="Fully matched", density=True)
safe_hist(axes[1, 1], notfull_frac, bins=20, range=(0, 1.05), label="Partial/Unmatched", density=True)
axes[1, 1].set_xlabel("Containment fraction")
axes[1, 1].set_ylabel("Normalized entries")
axes[1, 1].set_title("Containment")
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/full_vs_notfull_side_by_side.pdf")
plt.close(fig)

# 7) Best dark-hadron to GenFatJet axis DR
fig, ax = plt.subplots(figsize=(8, 6))
safe_hist(ax, bestdr_all, bins=50, range=(0, 1.5), label="Best DR", density=False, histtype="stepfilled", lw=1)
ax.axvline(FATJET_R, color="red", linestyle="--", label=f"R={FATJET_R}")
ax.set_xlabel(r"Best $\Delta R$(dark hadron, GenFatJet)")
ax.set_ylabel("Counts")
ax.set_title("Dark Hadron to Best GenFatJet Axis Distance")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/darkhadron_to_genfatjet_bestdr.pdf")
plt.close(fig)

# 3D plots for one representative sample in each category
full_rep = choose_representative_sample(full_samples, "full")
partial_rep = choose_representative_sample(partial_samples, "partial")
none_rep = choose_representative_sample(none_samples, "none")

if full_rep is not None:
    plot_3d_match_category(
        full_rep,
        category_label="Fully matched",
        outpath=f"{PLOT_DIR}/3D_fully_matched_example.pdf",
        fatjet_r=FATJET_R
    )

if partial_rep is not None:
    plot_3d_match_category(
        partial_rep,
        category_label="Partially matched",
        outpath=f"{PLOT_DIR}/3D_partially_matched_example.pdf",
        fatjet_r=FATJET_R
    )

if none_rep is not None:
    plot_3d_match_category(
        none_rep,
        category_label="Unmatched",
        outpath=f"{PLOT_DIR}/3D_unmatched_example.pdf",
        fatjet_r=FATJET_R
    )

plot_3d_match_gallery(
    full_samples,
    partial_samples,
    none_samples,
    outpath=f"{PLOT_DIR}/3D_match_category_gallery.pdf",
    fatjet_r=FATJET_R
)


plot_3d_category_gallery(
    full_samples,
    category_label="Fully matched",
    outpath=f"{PLOT_DIR}/3D_gallery_fully_matched.pdf",
    fatjet_r=FATJET_R,
    n_show=9
)

plot_3d_category_gallery(
    partial_samples,
    category_label="Partially matched",
    outpath=f"{PLOT_DIR}/3D_gallery_partially_matched.pdf",
    fatjet_r=FATJET_R,
    n_show=9
)

plot_3d_category_gallery(
    none_samples,
    category_label="Unmatched",
    outpath=f"{PLOT_DIR}/3D_gallery_unmatched.pdf",
    fatjet_r=FATJET_R,
    n_show=9
)

print(f"\nSaved plots in: {PLOT_DIR}")
