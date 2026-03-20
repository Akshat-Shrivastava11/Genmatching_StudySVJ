import sys
import os
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import networkx as nx

# --- CONFIGURATION ---
ROOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/lustre/research/hep/akshriva/SVJ_RandD/trainingdata_maker/training_data/SVJ_Training_2D_20260316_2324/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-10.08_mrho-88.96_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root"
PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

FATJET_R = 0.8
DARK_HADRONS = [4900111, 4900211, 4900113, 4900213]
DARK_QUARKS  = [4900101, 4900102, 4900103, 4900104, 4900105, 4900106]
# Dictionary of names for clarity
names = {
    21: "g", 22: "gamma", 111: "pi0", 211: "pi+", -211: "pi-",
    1: "d", -1: "d~", 2: "u", -2: "u~", 3: "s", -3: "s~", 
    4: "c", -4: "c~", 5: "b", -5: "b~", 6: "t", -6: "t~",
    51: "DM_51", -51: "DM_51~", # Pythia invisible Dark Matter
    53: "DM_53", -53: "DM_53~", # Pythia invisible Dark Matter
    551: "eta_b", 553: "Upsilon", # Bottomonium mesons
    4900111: "DARK_PI0", 4900211: "DARK_PI+", -4900211: "DARK_PI-",
    4900113: "DARK_RHO0", 4900213: "DARK_RHO+", -4900213: "DARK_RHO-",
    4900101: "DARK_q1", -4900101: "DARK_q1~", 
    4900102: "DARK_q2", -4900102: "DARK_q2~", # The specific Dark Quark in your graph
    4900021: "DARK_g"
}
def get_name(pid): return names.get(pid, str(pid))

def calc_dr(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return np.sqrt(deta**2 + dphi**2)

print(f"Opening {ROOT_FILE}...")
with uproot.open(ROOT_FILE) as f:
    tree = f["Delphes;1"]
    
    gen_pid = tree["GenParticle/GenParticle.PID"].array()
    gen_pt  = tree["GenParticle/GenParticle.PT"].array()
    gen_eta = tree["GenParticle/GenParticle.Eta"].array()
    gen_phi = tree["GenParticle/GenParticle.Phi"].array()
    gen_m1  = tree["GenParticle/GenParticle.M1"].array()
    gen_d1  = tree["GenParticle/GenParticle.D1"].array()
    gen_d2  = tree["GenParticle/GenParticle.D2"].array()

    gf_pt  = tree["GenFatJet/GenFatJet.PT"].array()
    gf_eta = tree["GenFatJet/GenFatJet.Eta"].array()
    gf_phi = tree["GenFatJet/GenFatJet.Phi"].array()

    rf_pt  = tree["FatJet/FatJet.PT"].array()
    rf_eta = tree["FatJet/FatJet.Eta"].array()
    rf_phi = tree["FatJet/FatJet.Phi"].array()

    pf_pt  = tree["ParticleFlowCandidate/ParticleFlowCandidate.PT"].array()
    pf_eta = tree["ParticleFlowCandidate/ParticleFlowCandidate.Eta"].array()
    pf_phi = tree["ParticleFlowCandidate/ParticleFlowCandidate.Phi"].array()

num_events = min(1000, len(rf_pt))

# =========================================================================
# PART 1: GEOMETRIC DELTA-R MATCHING PLOTS
# =========================================================================
print("\nGenerating Matching Plots...")
dr_GenJet_RecoJet, dr_DarkHadron_GenJet, dr_DarkQuark_RecoJet = [], [], []

for i in range(num_events):
    # A. Reco to GenJet
    if len(gf_pt[i]) > 0 and len(rf_pt[i]) > 0:
        for g_idx in range(len(gf_pt[i])):
            drs = calc_dr(gf_eta[i][g_idx], gf_phi[i][g_idx], rf_eta[i], rf_phi[i])
            dr_GenJet_RecoJet.append(np.min(drs))

    # B. Dark Hadron to GenJet
    pids = np.abs(gen_pid[i])
    dh_mask = np.isin(pids, DARK_HADRONS)
    if np.any(dh_mask) and len(gf_pt[i]) > 0:
        dhs_eta, dhs_phi = gen_eta[i][dh_mask], gen_phi[i][dh_mask]
        for dh_e, dh_p in zip(dhs_eta, dhs_phi):
            drs = calc_dr(dh_e, dh_p, gf_eta[i], gf_phi[i])
            dr_DarkHadron_GenJet.append(np.min(drs))

    # C. Dark Quark to RecoJet
    pids_np = ak.to_numpy(np.abs(gen_pid[i]))
    dq_mask = np.isin(pids_np, DARK_QUARKS)
    if np.any(dq_mask) and len(rf_pt[i]) > 0:
        dqs_eta, dqs_phi = ak.to_numpy(gen_eta[i])[dq_mask], ak.to_numpy(gen_phi[i])[dq_mask]
        for dq_e, dq_p in zip(dqs_eta, dqs_phi):
            if np.isnan(dq_e) or np.isnan(dq_p) or np.isinf(dq_e): continue
            drs = calc_dr(dq_e, dq_p, ak.to_numpy(rf_eta[i]), ak.to_numpy(rf_phi[i]))
            if len(drs) > 0: dr_DarkQuark_RecoJet.append(np.min(drs))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Semivisible Jet (SVJ) Matching Validation", fontsize=16)

axes[0].hist(dr_GenJet_RecoJet, bins=50, range=(0, 1.5), color='blue', alpha=0.7)
axes[0].axvline(FATJET_R, color='red', linestyle='--', label=f'AK8 Cone ($\Delta R={FATJET_R}$)')
axes[0].set_title(r"Closest Reco FatJet to GenFatJet"); axes[0].set_xlabel(r"$\Delta R$"); axes[0].legend()

axes[1].hist(dr_DarkHadron_GenJet, bins=50, range=(0, 1.5), color='purple', alpha=0.7)
axes[1].axvline(FATJET_R, color='red', linestyle='--')
axes[1].set_title(r"Closest GenFatJet to Dark Hadron"); axes[1].set_xlabel(r"$\Delta R$")

axes[2].hist(dr_DarkQuark_RecoJet, bins=50, range=(0, 3.0), color='green', alpha=0.7)
axes[2].axvline(FATJET_R, color='red', linestyle='--')
axes[2].set_title(r"Closest Reco FatJet to Initial Dark Quark"); axes[2].set_xlabel(r"$\Delta R$")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/svj_matching_validation.pdf")

# =========================================================================
# PART 2: JET SUBSTRUCTURE PROFILES
# =========================================================================
print("Generating Substructure Plots...")
dh_counts, dh_pts, jet_pts = [], [], []
scatter_event_idx = None

for i in range(num_events):
    dh_mask = np.isin(np.abs(gen_pid[i]), DARK_HADRONS)
    if np.any(dh_mask) and len(rf_pt[i]) > 0:
        dhs_pt, dhs_eta, dhs_phi = gen_pt[i][dh_mask], gen_eta[i][dh_mask], gen_phi[i][dh_mask]
        j_eta, j_phi, j_pt = rf_eta[i][0], rf_phi[i][0], rf_pt[i][0] # Leading jet
        
        in_jet = calc_dr(dhs_eta, dhs_phi, j_eta, j_phi) < FATJET_R
        n_in_jet = np.sum(in_jet)
        
        if n_in_jet > 0:
            dh_counts.append(n_in_jet)
            dh_pts.extend(dhs_pt[in_jet])
            jet_pts.append(j_pt)
            scatter_event_idx = i 

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(dh_counts, bins=15, range=(0, 15), color='purple', alpha=0.7)
axes[0].set_title("Dark Hadrons per FatJet"); axes[0].set_xlabel("Number of Dark Hadrons")

axes[1].hist(jet_pts, bins=40, range=(0, 1000), color='blue', alpha=0.5, label='Reco FatJet $p_T$')
axes[1].hist(dh_pts, bins=40, range=(0, 1000), color='purple', alpha=0.7, label='Dark Hadron $p_T$')
axes[1].set_title("Kinematic Sharing"); axes[1].set_xlabel("$p_T$ (GeV)"); axes[1].legend()

if scatter_event_idx is not None:
    i = scatter_event_idx
    j_eta, j_phi = rf_eta[i][0], rf_phi[i][0]
    
    pf_in = calc_dr(pf_eta[i], pf_phi[i], j_eta, j_phi) < FATJET_R
    axes[2].scatter(ak.to_numpy(pf_eta[i][pf_in]), ak.to_numpy(pf_phi[i][pf_in]), 
                    s=ak.to_numpy(pf_pt[i][pf_in]), color='blue', alpha=0.5, label='Visible Constituents')
    
    dh_mask = np.isin(np.abs(gen_pid[i]), DARK_HADRONS)
    dh_in = calc_dr(gen_eta[i][dh_mask], gen_phi[i][dh_mask], j_eta, j_phi) < FATJET_R
    axes[2].scatter(ak.to_numpy(gen_eta[i][dh_mask][dh_in]), ak.to_numpy(gen_phi[i][dh_mask][dh_in]), 
                    s=ak.to_numpy(gen_pt[i][dh_mask][dh_in]), color='red', marker='X', label='Dark Hadrons')

    axes[2].add_patch(plt.Circle((j_eta, j_phi), FATJET_R, color='black', fill=False, linestyle='--'))
    axes[2].set_title("SVJ 2D Substructure Profile"); axes[2].set_xlabel("$\eta$"); axes[2].set_ylabel("$\phi$")
    axes[2].legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/svj_substructure_profile.pdf")


# =========================================================================
# PART 2.5: nSVJs PER EVENT (USING GEN-MATCHING LOGIC)
# =========================================================================
# print("\nCalculating nSVJs per event using strict Gen-Matching...")

# n_svjs_per_event = []
# MATCHING_R = 0.4 

# for i in range(num_events):
#     n_svj = 0
    
#     # Extract arrays for this specific event
#     pids = np.abs(gen_pid[i])
#     dh_mask = np.isin(pids, DARK_HADRONS)
#     dhs_eta = ak.to_numpy(gen_eta[i][dh_mask])
#     dhs_phi = ak.to_numpy(gen_phi[i][dh_mask])
    
#     gfs_eta = ak.to_numpy(gf_eta[i])
#     gfs_phi = ak.to_numpy(gf_phi[i])
    
#     rfs_pt  = ak.to_numpy(rf_pt[i])
#     rfs_eta = ak.to_numpy(rf_eta[i])
#     rfs_phi = ak.to_numpy(rf_phi[i])
    
#     # Step 1 & 2: Form GenJets and mark them if they contain Dark Hadrons
#     genjet_is_svj = np.zeros(len(gfs_eta), dtype=bool)
#     if len(dhs_eta) > 0 and len(gfs_eta) > 0:
#         for g_idx in range(len(gfs_eta)):
#             drs = calc_dr(gfs_eta[g_idx], gfs_phi[g_idx], dhs_eta, dhs_phi)
#             if np.any(drs < FATJET_R):  # If a dark hadron is inside the GenJet cone
#                 genjet_is_svj[g_idx] = True
                
#     # Step 3 & 4: Match RecoJets to GenJets, mark as SVJ if matched GenJet is SVJ
#     for r_idx in range(len(rfs_pt)):
#         # Apply a basic pT cut so we don't count ultra-soft pileup/noise as signal jets
#         if rfs_pt[r_idx] < 100.0: 
#             continue 
            
#         if len(gfs_eta) > 0:
#             drs = calc_dr(rfs_eta[r_idx], rfs_phi[r_idx], gfs_eta, gfs_phi)
#             closest_gen_idx = np.argmin(drs)
            
#             # If the closest GenJet is within matching distance AND is marked as an SVJ
#             if drs[closest_gen_idx] < MATCHING_R and genjet_is_svj[closest_gen_idx]:
#                 n_svj += 1
                
#     n_svjs_per_event.append(n_svj)
# Helper function for recursive ancestry check
def has_dark_ancestor(idx, all_pids, all_m1, dark_hadrons_set, cache):
    if idx < 0 or idx >= len(all_pids): return False
    if idx in cache: return cache[idx]
    
    # Check if the current particle itself is a Dark Hadron
    if abs(all_pids[idx]) in dark_hadrons_set:
        cache[idx] = True
        return True
    
    parent = int(all_m1[idx])
    # Recursive step
    result = has_dark_ancestor(parent, all_pids, all_m1, dark_hadrons_set, cache) if parent >= 0 else False
    
    cache[idx] = result
    return result

# =========================================================================
# PART 2.5: nSVJs PER EVENT (RECURSIVE ANCESTRY MATCHING)
# =========================================================================
print("\nCalculating nSVJs per event using Recursive Ancestry-based Gen-Matching...")

n_svjs_per_event = []
MATCHING_R = 0.4 
dark_hadrons_set = set(DARK_HADRONS)

for i in range(num_events):
    n_svj = 0
    event_cache = {} # Reset cache for every event
    
    all_pids = ak.to_numpy(gen_pid[i])
    all_m1   = ak.to_numpy(gen_m1[i])
    gfs_eta  = ak.to_numpy(gf_eta[i])
    gfs_phi  = ak.to_numpy(gf_phi[i])
    
    genjet_is_svj = np.zeros(len(gfs_eta), dtype=bool)
    
    # Loop over GenJets to determine "Semivisible" status
    for g_idx in range(len(gfs_eta)):
        # Find indices of all GenParticles within the GenJet cone
        drs_to_particles = calc_dr(gfs_eta[g_idx], gfs_phi[g_idx], gen_eta[i], gen_phi[i])
        indices_in_cone = np.where(drs_to_particles < FATJET_R)[0]
        
        # Check every particle in the cone for dark ancestry
        for p_idx in indices_in_cone:
            if has_dark_ancestor(p_idx, all_pids, all_m1, dark_hadrons_set, event_cache):
                genjet_is_svj[g_idx] = True
                break # One dark constituent is enough to mark the whole GenJet

    # Match RecoJets to GenJets
    rfs_pt  = ak.to_numpy(rf_pt[i])
    rfs_eta = ak.to_numpy(rf_eta[i])
    rfs_phi = ak.to_numpy(rf_phi[i])

    for r_idx in range(len(rfs_pt)):
        if rfs_pt[r_idx] < 100.0: continue 
            
        if len(gfs_eta) > 0:
            drs = calc_dr(rfs_eta[r_idx], rfs_phi[r_idx], gfs_eta, gfs_phi)
            closest_gen_idx = np.argmin(drs)
            
            if drs[closest_gen_idx] < MATCHING_R and genjet_is_svj[closest_gen_idx]:
                n_svj += 1
                
    n_svjs_per_event.append(n_svj)
# --- PLOT nSVJs ---
print("Generating nSVJs Histogram...")
fig_nsvj, ax_nsvj = plt.subplots(figsize=(8, 6))

# Bins from -0.5 to 5.5 to center the bars nicely on integers 0, 1, 2, 3...
bins = np.arange(-0.5, 6.5, 1)
counts, _, bars = ax_nsvj.hist(n_svjs_per_event, bins=bins, color='darkorange', alpha=0.8, edgecolor='black')

ax_nsvj.set_title("Number of Semivisible Jets (nSVJs) per Event\n" + r"($s$-channel, $M_{Z'} = 2000$ GeV)", fontsize=16)
ax_nsvj.set_xlabel("nSVJs", fontsize=14)
ax_nsvj.set_ylabel("Number of Events", fontsize=14)
ax_nsvj.set_xticks(range(6))

# Add text labels on top of the bars to show exact counts
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax_nsvj.text(bar.get_x() + bar.get_width()/2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/nSVJs_histogram.pdf")
print(f"Saved nSVJs plot to {PLOT_DIR}/nSVJs_histogram.pdf")

 

 
# =========================================================================
# PART 3: NETWORKX DECAY CHAIN VISUALIZER (Event 0)
# =========================================================================
print("\nGenerating Pythia Decay Graph...")

ev = 1
pids = gen_pid[ev]
d1s  = gen_d1[ev]
d2s  = gen_d2[ev]

pids_np = np.abs(ak.to_numpy(pids))
dark_quark_idx = np.where(np.isin(pids_np, DARK_QUARKS))[0]

if len(dark_quark_idx) > 0:
    first_dq_idx = dark_quark_idx[0]
    
    # Initialize Directed Graph
    G = nx.DiGraph()
    
    def build_graph(idx, depth=0, max_depth=6):
        if depth > max_depth or idx < 0 or idx >= len(pids): return
        
        pid = pids[idx]
        # FIX: Removed the ":" from the ID string to prevent pydot parsing errors
        node_label = f'"{get_name(pid)}\n(ID {idx})"'
        G.add_node(idx, label=node_label, pid=np.abs(pid))
        
        d1, d2 = d1s[idx], d2s[idx]
        if d1 >= 0 and d2 >= 0 and d1 <= d2:
            for d_idx in range(d1, d2 + 1):
                child_pid = pids[d_idx]
                child_label = f'"{get_name(child_pid)}\n(ID {d_idx})"'
                G.add_node(d_idx, label=child_label, pid=np.abs(child_pid))
                G.add_edge(idx, d_idx)
                build_graph(d_idx, depth + 1, max_depth)
        elif d1 >= 0:
            child_pid = pids[d1]
            child_label = f'"{get_name(child_pid)}\n(ID {d1})"'
            G.add_node(d1, label=child_label, pid=np.abs(child_pid))
            G.add_edge(idx, d1)
            build_graph(d1, depth + 1, max_depth)

    # Build the graph starting from the first dark quark
    build_graph(first_dq_idx, max_depth=6)
    
    # Plotting the Graph
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use graphviz layout for a perfect top-down tree
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
    except ImportError:
        print("  Graphviz not installed. Falling back to spring layout...")
        pos = nx.spring_layout(G, k=0.9, iterations=50)

    # Color nodes based on what they are
    color_map = []
    for node, data in G.nodes(data=True):
        pid = data.get('pid', 0)
        if pid in DARK_QUARKS: color_map.append('lightgreen')
        elif pid == 4900021: color_map.append('lightblue') # Dark Gluon
        elif pid in DARK_HADRONS: color_map.append('plum') # Dark Mesons
        else: color_map.append('lightgray') # SM Particles

    # Clean the quotes off the labels before drawing
    labels = {n: d['label'].strip('"') for n, d in G.nodes(data=True)}
    
    nx.draw(G, pos, ax=ax, labels=labels, with_labels=True, 
            node_color=color_map, node_size=2000, font_size=8, 
            font_weight='bold', edge_color='gray', arrows=True)
            
    ax.set_title("Pythia Dark Shower & Hadronization String Diagram", fontsize=18)
    plt.savefig(f"{PLOT_DIR}/decay_chain_graph.pdf")
    print(f"Saved Pythia Decay Graph to {PLOT_DIR}/decay_chain_graph.pdf")





else:
    print("No Dark Quarks found in Event 0. Cannot build graph.")
    
print("\n[DONE] All validation checks and plots complete!")

# =========================================================================
# PART 4: SVJ vs BACKGROUND 2D SUBSTRUCTURE GALLERY (TRUE 2x5, USING
#         RECURSIVE ANCESTRY-BASED GEN-MATCHING CONSISTENT WITH nSVJs)
# =========================================================================
print("\nExtracting jet samples for 2D Substructure Gallery (recursive ancestry matching)...")

svj_samples = []
bkg_samples = []
MATCHING_R = 0.4
N_SHOW = 5

# Helper function to center jet constituents at (0,0)
def center_coords(etas, phis, center_eta, center_phi):
    d_eta = etas - center_eta
    d_phi = phis - center_phi
    d_phi = np.where(d_phi > np.pi, d_phi - 2*np.pi, d_phi)
    d_phi = np.where(d_phi < -np.pi, d_phi + 2*np.pi, d_phi)
    return d_eta, d_phi

# Search through events until we have 5 SVJ jets and 5 background jets
for i in range(num_events):
    if len(svj_samples) >= N_SHOW and len(bkg_samples) >= N_SHOW:
        break

    event_cache = {}

    # ---------- event-level arrays ----------
    all_pids = ak.to_numpy(gen_pid[i])
    all_m1   = ak.to_numpy(gen_m1[i])

    gfs_eta = ak.to_numpy(gf_eta[i])
    gfs_phi = ak.to_numpy(gf_phi[i])

    rfs_pt  = ak.to_numpy(rf_pt[i])
    rfs_eta = ak.to_numpy(rf_eta[i])
    rfs_phi = ak.to_numpy(rf_phi[i])

    pfs_pt  = ak.to_numpy(pf_pt[i])
    pfs_eta = ak.to_numpy(pf_eta[i])
    pfs_phi = ak.to_numpy(pf_phi[i])

    dh_mask = np.isin(np.abs(gen_pid[i]), DARK_HADRONS)
    dhs_pt  = ak.to_numpy(gen_pt[i][dh_mask])
    dhs_eta = ak.to_numpy(gen_eta[i][dh_mask])
    dhs_phi = ak.to_numpy(gen_phi[i][dh_mask])

    # ---------- label GenJets using recursive ancestry ----------
    genjet_is_svj = np.zeros(len(gfs_eta), dtype=bool)

    for g_idx in range(len(gfs_eta)):
        drs_to_particles = calc_dr(gfs_eta[g_idx], gfs_phi[g_idx], gen_eta[i], gen_phi[i])
        indices_in_cone = np.where(ak.to_numpy(drs_to_particles) < FATJET_R)[0]

        for p_idx in indices_in_cone:
            if has_dark_ancestor(p_idx, all_pids, all_m1, dark_hadrons_set, event_cache):
                genjet_is_svj[g_idx] = True
                break

    # ---------- loop over reco jets ----------
    for r_idx in range(len(rfs_pt)):
        if len(svj_samples) >= N_SHOW and len(bkg_samples) >= N_SHOW:
            break

        if rfs_pt[r_idx] < 100.0:
            continue

        j_eta, j_phi, j_pt = rfs_eta[r_idx], rfs_phi[r_idx], rfs_pt[r_idx]

        # Reco -> closest GenJet matching
        is_svj = False
        if len(gfs_eta) > 0:
            drs = calc_dr(j_eta, j_phi, gfs_eta, gfs_phi)
            closest_gen = np.argmin(drs)
            if drs[closest_gen] < MATCHING_R and genjet_is_svj[closest_gen]:
                is_svj = True

        # Visible PF constituents inside jet
        pf_drs = calc_dr(j_eta, j_phi, pfs_eta, pfs_phi)
        pf_in = pf_drs < FATJET_R
        c_eta, c_phi, c_pt = pfs_eta[pf_in], pfs_phi[pf_in], pfs_pt[pf_in]

        # Dark hadrons inside jet (for overlay only)
        dh_drs = calc_dr(j_eta, j_phi, dhs_eta, dhs_phi)
        dh_in = dh_drs < FATJET_R
        d_eta, d_phi, d_pt = dhs_eta[dh_in], dhs_phi[dh_in], dhs_pt[dh_in]

        # Center around jet axis
        c_deta, c_dphi = center_coords(c_eta, c_phi, j_eta, j_phi)
        d_deta, d_dphi = center_coords(d_eta, d_phi, j_eta, j_phi)

        sample_dict = {
            "event": i,
            "jet_index": r_idx,
            "pt": j_pt,
            "c_deta": c_deta,
            "c_dphi": c_dphi,
            "c_pt": c_pt,
            "d_deta": d_deta,
            "d_dphi": d_dphi,
            "d_pt": d_pt
        }

        if is_svj and len(svj_samples) < N_SHOW:
            svj_samples.append(sample_dict)

        elif (not is_svj) and len(bkg_samples) < N_SHOW:
            # Avoid nearly empty junk jets in the gallery
            if len(c_pt) > 5:
                bkg_samples.append(sample_dict)

# --- PLOT THE GALLERY ---
print("Generating 2x5 Gallery Plot...")
fig_gal, axes_gal = plt.subplots(2, N_SHOW, figsize=(4.2 * N_SHOW, 8.5))
fig_gal.suptitle(
    r"Jet Substructure Gallery (Centered on Jet Axis $\Delta\eta, \Delta\phi$)",
    fontsize=20
)

for col in range(N_SHOW):
    # =========================================================
    # TOP ROW: SVJ
    # =========================================================
    ax_svj = axes_gal[0, col]

    if col < len(svj_samples):
        svj = svj_samples[col]

        ax_svj.scatter(
            svj["c_deta"], svj["c_dphi"],
            s=np.clip(svj["c_pt"] * 0.5, 8, 250),
            c="blue", alpha=0.5, label="Visible Constituents"
        )

        if len(svj["d_pt"]) > 0:
            ax_svj.scatter(
                svj["d_deta"], svj["d_dphi"],
                s=np.clip(svj["d_pt"] * 0.5, 12, 300),
                c="red", marker="X", alpha=0.8, label="Dark Hadrons"
            )

        ax_svj.set_title(
            f"SVJ  ($p_T$={svj['pt']:.0f} GeV)",
            fontsize=11, color="darkred"
        )
    else:
        ax_svj.set_title("No SVJ sample", fontsize=11, color="gray")

    ax_svj.add_patch(
        plt.Circle((0, 0), FATJET_R, color="black", fill=False, linestyle="--", linewidth=1.2)
    )
    ax_svj.set_xlim(-1.0, 1.0)
    ax_svj.set_ylim(-1.0, 1.0)
    ax_svj.set_aspect("equal", adjustable="box")
    ax_svj.grid(alpha=0.2)

    if col == 0:
        ax_svj.legend(loc="upper right", fontsize=9)

    # =========================================================
    # BOTTOM ROW: BACKGROUND
    # =========================================================
    ax_bkg = axes_gal[1, col]

    if col < len(bkg_samples):
        bkg = bkg_samples[col]

        ax_bkg.scatter(
            bkg["c_deta"], bkg["c_dphi"],
            s=np.clip(bkg["c_pt"] * 0.5, 8, 250),
            c="blue", alpha=0.5, label="Visible Constituents"
        )

        if len(bkg["d_pt"]) > 0:
            ax_bkg.scatter(
                bkg["d_deta"], bkg["d_dphi"],
                s=np.clip(bkg["d_pt"] * 0.5, 12, 300),
                c="red", marker="X", alpha=0.8, label="Dark Hadrons"
            )

        ax_bkg.set_title(
            f"SM Jet ($p_T$={bkg['pt']:.0f} GeV)",
            fontsize=11, color="darkblue"
        )
    else:
        ax_bkg.set_title("No background sample", fontsize=11, color="gray")

    ax_bkg.add_patch(
        plt.Circle((0, 0), FATJET_R, color="black", fill=False, linestyle="--", linewidth=1.2)
    )
    ax_bkg.set_xlim(-1.0, 1.0)
    ax_bkg.set_ylim(-1.0, 1.0)
    ax_bkg.set_aspect("equal", adjustable="box")
    ax_bkg.grid(alpha=0.2)

for ax in axes_gal.flatten():
    ax.set_xlabel(r"$\Delta\eta$", fontsize=11)
    ax.set_ylabel(r"$\Delta\phi$", fontsize=11)
    ax.tick_params(labelsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{PLOT_DIR}/jet_substructure_gallery.pdf")
print(f"Saved Gallery Plot to {PLOT_DIR}/jet_substructure_gallery.pdf")

# =========================================================================
# PART 5: 3D KINEMATIC DECAY TREE (Side-View for nSVJs = 0,1,2,3,4)
# =========================================================================
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

print("\nGenerating Side-View 3D Kinematic Decay Trees for nSVJs = 0,1,2,3,4...")

# ---------------------------------------------------------
# Find one representative event for each nSVJ category
# ---------------------------------------------------------
target_nsvj_values = [0, 1, 2, 3, 4]
representative_events = {}

for i, nsvj in enumerate(n_svjs_per_event):
    if nsvj in target_nsvj_values and nsvj not in representative_events:
        representative_events[nsvj] = i
    if len(representative_events) == len(target_nsvj_values):
        break

print("Representative events found:")
for n in target_nsvj_values:
    if n in representative_events:
        print(f"  nSVJ = {n}  --> event {representative_events[n]}")
    else:
        print(f"  nSVJ = {n}  --> no event found")

# ---------------------------------------------------------
# Helper: choose node color
# ---------------------------------------------------------
def get_particle_color(pid):
    if pid == 4900023:
        return 'red'         # Z'
    elif pid in DARK_QUARKS:
        return 'orange'      # dark quarks
    elif pid == 4900021:
        return 'salmon'      # dark gluon
    elif pid in DARK_HADRONS:
        return 'purple'      # dark hadrons
    elif pid in [1, 2, 3, 4, 5, 6]:
        return 'green'       # SM quarks
    elif pid in [11, 13, 15]:
        return 'blue'        # charged leptons
    elif pid == 22:
        return 'lightgreen'  # photons
    else:
        return 'lightblue'   # anything else

# ---------------------------------------------------------
# Helper: make one side-view plot for one event
# ---------------------------------------------------------
def make_3d_sideview_plot(ev, nsvj_label, outpath, max_depth=16):
    pids = gen_pid[ev]
    etas = gen_eta[ev]
    phis = gen_phi[ev]
    d1s  = gen_d1[ev]
    d2s  = gen_d2[ev]

    pids_np_signed = ak.to_numpy(pids)
    pids_np_abs = np.abs(pids_np_signed)

    dark_quark_idx = np.where(np.isin(pids_np_abs, DARK_QUARKS))[0]
    if len(dark_quark_idx) == 0:
        print(f"  Event {ev}: no dark quark found, skipping nSVJ={nsvj_label}")
        return

    first_dq_idx = dark_quark_idx[0]
    zprime_idx = np.where(pids_np_abs == 4900023)[0]
    start_idx = zprime_idx[0] if len(zprime_idx) > 0 else first_dq_idx

    node_coords = {}
    edges = []

    def build_3d_data(idx, depth=0):
        if depth > max_depth or idx < 0 or idx >= len(pids):
            return
        if idx in node_coords:
            return

        eta = float(etas[idx])
        phi = float(phis[idx])
        pid_abs = int(pids_np_abs[idx])

        if np.isnan(eta) or np.isinf(eta) or np.abs(eta) > 10:
            eta = 0.0
        if np.isnan(phi) or np.isinf(phi):
            phi = 0.0

        node_coords[idx] = (depth, phi, eta, pid_abs)

        d1 = int(d1s[idx])
        d2 = int(d2s[idx])

        if d1 >= 0 and d2 >= 0 and d1 <= d2:
            for d_idx in range(d1, d2 + 1):
                if 0 <= d_idx < len(pids):
                    edges.append((idx, d_idx))
                    build_3d_data(d_idx, depth + 1)
        elif d1 >= 0 and d1 < len(pids):
            edges.append((idx, d1))
            build_3d_data(d1, depth + 1)

    build_3d_data(start_idx, depth=0)

    if len(node_coords) == 0:
        print(f"  Event {ev}: empty decay tree, skipping nSVJ={nsvj_label}")
        return

    fig_3d = plt.figure(figsize=(14, 8))
    ax_3d = fig_3d.add_axes([0.05, 0.1, 0.65, 0.8], projection='3d')

    # Draw edges
    for parent, child in edges:
        if parent in node_coords and child in node_coords:
            p_depth, p_phi, p_eta, _ = node_coords[parent]
            c_depth, c_phi, c_eta, _ = node_coords[child]
            ax_3d.plot(
                [p_depth, c_depth],
                [p_phi, c_phi],
                [p_eta, c_eta],
                color='gray', linewidth=0.7, alpha=0.35
            )

    # Draw nodes
    for idx, (depth, phi, eta, pid) in node_coords.items():
        color = get_particle_color(pid)
        ax_3d.scatter(
            depth, phi, eta,
            c=color, s=42,
            edgecolors='white', linewidth=0.35,
            alpha=0.95
        )

    legend_elements = [
        mlines.Line2D([0], [0], marker='o', color='w', label="Z' (4900023)", markerfacecolor='red', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='Dark Quarks', markerfacecolor='orange', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='Dark Gluons', markerfacecolor='salmon', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='Dark Hadrons', markerfacecolor='purple', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='SM Quarks', markerfacecolor='green', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='SM Leptons', markerfacecolor='blue', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='Photons', markerfacecolor='lightgreen', markersize=10),
        mlines.Line2D([0], [0], marker='o', color='w', label='Other Particles', markerfacecolor='lightblue', markersize=10),
    ]

    ax_3d.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=10,
        frameon=True,
        shadow=True,
        title="Particle Species"
    )

    # Side view
    ax_3d.view_init(elev=0, azim=-90)

    ax_3d.set_xlabel('Decay Depth', labelpad=10)
    ax_3d.set_ylabel(r'$\phi$', labelpad=10)
    ax_3d.set_zlabel(r'$\eta$', labelpad=10)

    plt.title(f"Side-View: Decay Tree for Event {ev} (nSVJs = {nsvj_label})", fontsize=16)
    plt.savefig(outpath, bbox_inches='tight')
    plt.close(fig_3d)

    print(f"  Saved {outpath}")

# ---------------------------------------------------------
# Make plots for nSVJ = 0,1,2,3,4
# ---------------------------------------------------------
for n in target_nsvj_values:
    if n in representative_events:
        ev = representative_events[n]
        outpath = f"{PLOT_DIR}/3D_decay_tree_nSVJ{n}.pdf"
        make_3d_sideview_plot(ev, n, outpath)