import sys
import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt

ROOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/training_data/SVJ_Training_2D_20260316_2324/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-10.08_mrho-88.96_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root"
PLOT_DIR = "plots"
import os
if not os.path.exists(PLOT_DIR): os.makedirs(PLOT_DIR)

# Standard AK8 FatJet Cone Size
FATJET_R = 0.8

def calc_dr(eta1, phi1, eta2, phi2):
    """Calculates Delta R between two sets of eta/phi arrays."""
    deta = eta1 - eta2
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return np.sqrt(deta**2 + dphi**2)

print(f"Opening {ROOT_FILE}...")
with uproot.open(ROOT_FILE) as f:
    tree = f["Delphes;1"]
    
    # 1. Load GenParticles
    print("Loading GenParticles...")
    gen_pid = tree["GenParticle/GenParticle.PID"].array()
    gen_eta = tree["GenParticle/GenParticle.Eta"].array()
    gen_phi = tree["GenParticle/GenParticle.Phi"].array()
    
    # Mother/Daughter indices for the decay chain crawler
    gen_m1  = tree["GenParticle/GenParticle.M1"].array()
    gen_d1  = tree["GenParticle/GenParticle.D1"].array()
    gen_d2  = tree["GenParticle/GenParticle.D2"].array()

    # 2. Load GenFatJets (Clustered from stable SM GenParticles)
    print("Loading GenFatJets...")
    gf_pt  = tree["GenFatJet/GenFatJet.PT"].array()
    gf_eta = tree["GenFatJet/GenFatJet.Eta"].array()
    gf_phi = tree["GenFatJet/GenFatJet.Phi"].array()

    # 3. Load Reco FatJets
    print("Loading Reco FatJets...")
    rf_pt  = tree["FatJet/FatJet.PT"].array()
    rf_eta = tree["FatJet/FatJet.Eta"].array()
    rf_phi = tree["FatJet/FatJet.Phi"].array()

# =========================================================================
# PART 1: GEOMETRIC DELTA-R MATCHING & PLOTTING
# =========================================================================
print("\nPerforming Geometric Matching...")

dr_GenJet_RecoJet = []
dr_DarkHadron_GenJet = []
dr_DarkQuark_RecoJet = []

# Dark Sector PIDs
DARK_HADRONS = [4900111, 4900211, 4900113, 4900213]
#DARK_QUARKS  = [4900001, 4900002, 4900003, 4900004, 4900005, 4900006]
DARK_QUARKS  = [4900101, 4900102, 4900103, 4900104, 4900105, 4900106]
print(f"Analyzing up to {min(1000, len(rf_pt))} events for matching validation...")
print(f"Dark Hadrons PIDs: {DARK_HADRONS}")
print(f"Dark Quarks PIDs: {DARK_QUARKS}")
num_events = min(1000, len(rf_pt)) # Analyze up to 1000 events for quick plotting

for i in range(num_events):
    # --- A. Match GenFatJet to Reco FatJet ---
    for g_idx in range(len(gf_pt[i])):
        if len(rf_pt[i]) > 0:
            drs = calc_dr(gf_eta[i][g_idx], gf_phi[i][g_idx], rf_eta[i], rf_phi[i])
            dr_GenJet_RecoJet.append(np.min(drs)) # Distance to closest RecoJet

    # --- B. Match Dark Hadrons to GenFatJet ---
    pids = np.abs(gen_pid[i])
    dh_mask = np.isin(pids, DARK_HADRONS)
    if np.any(dh_mask) and len(gf_pt[i]) > 0:
        dhs_eta = gen_eta[i][dh_mask]
        dhs_phi = gen_phi[i][dh_mask]
        for dh_e, dh_p in zip(dhs_eta, dhs_phi):
            drs = calc_dr(dh_e, dh_p, gf_eta[i], gf_phi[i])
            dr_DarkHadron_GenJet.append(np.min(drs))

   # --- C. Match Dark Quarks to Reco FatJet (Optional Check) ---
    # Convert to numpy to avoid awkward array masking bugs
    pids_np = ak.to_numpy(np.abs(gen_pid[i]))
    dq_mask = np.isin(pids_np, DARK_QUARKS)
    
    if np.any(dq_mask) and len(rf_pt[i]) > 0:
        dqs_eta = ak.to_numpy(gen_eta[i])[dq_mask]
        dqs_phi = ak.to_numpy(gen_phi[i])[dq_mask]
        
        for dq_e, dq_p in zip(dqs_eta, dqs_phi):
            # Delphes often assigns NaN/Inf to intermediate hard-scatter particles.
            # We must skip them or matplotlib will plot a blank histogram!
            if np.isnan(dq_e) or np.isnan(dq_p) or np.isinf(dq_e):
                continue
                
            drs = calc_dr(dq_e, dq_p, ak.to_numpy(rf_eta[i]), ak.to_numpy(rf_phi[i]))
            if len(drs) > 0:
                dr_DarkQuark_RecoJet.append(np.min(drs))

# --- GENERATE PLOTS ---
print("Generating Matching Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Semivisible Jet (SVJ) Matching Validation", fontsize=16)

# Plot 1: Reco vs Gen FatJet
axes[0].hist(dr_GenJet_RecoJet, bins=50, range=(0, 1.5), color='blue', alpha=0.7)
axes[0].axvline(FATJET_R, color='red', linestyle='--', label=f'AK8 Cone ($\Delta R={FATJET_R}$)')
axes[0].set_title(r"Closest Reco FatJet to GenFatJet")
axes[0].set_xlabel(r"$\Delta R$")
axes[0].set_ylabel("Counts")
axes[0].legend()

# Plot 2: Dark Hadron vs GenFatJet
axes[1].hist(dr_DarkHadron_GenJet, bins=50, range=(0, 1.5), color='purple', alpha=0.7)
axes[1].axvline(FATJET_R, color='red', linestyle='--')
axes[1].set_title(r"Closest GenFatJet to Dark Hadron")
axes[1].set_xlabel(r"$\Delta R$")

# Plot 3: Dark Quark vs Reco FatJet
axes[2].hist(dr_DarkQuark_RecoJet, bins=50, range=(0, 3.0), color='green', alpha=0.7)
axes[2].axvline(FATJET_R, color='red', linestyle='--')
axes[2].set_title(r"Closest Reco FatJet to Initial Dark Quark")
axes[2].set_xlabel(r"$\Delta R$")

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/svj_matching_validation.pdf")
print(f"Saved matching plots to {PLOT_DIR}/svj_matching_validation.pdf")


# =========================================================================
# PART 2: PYTHIA DECAY CHAIN CRAWLER (For 1 Event)
# =========================================================================
print("\n" + "="*50)
print("   SVJ DECAY CHAIN CRAWLER (EVENT 0)")
print("="*50)

ev = 0
pids = gen_pid[ev]
m1s  = gen_m1[ev]
d1s  = gen_d1[ev]
d2s  = gen_d2[ev]

# Dictionary of names for clarity
# Dictionary of names for clarity
names = {
    21: "g", 22: "gamma", 111: "pi0", 211: "pi+", -211: "pi-",
    2: "u", -2: "u~", 1: "d", -1: "d~", 3: "s", -3: "s~",
    4900111: "DARK_PI0", 4900211: "DARK_PI+", -4900211: "DARK_PI-",
    4900113: "DARK_RHO0", 4900213: "DARK_RHO+", -4900213: "DARK_RHO-",
    4900101: "DARK_q", -4900101: "DARK_q~", 4900021: "DARK_g",
    4900001: "DARK_q_alt", -4900001: "DARK_q~_alt" # Kept just in case
}

def get_name(pid):
    return names.get(pid, str(pid))

def print_tree(idx, indent=0, max_depth=5):
    if indent > max_depth or idx < 0 or idx >= len(pids): return
    
    pid = pids[idx]
    prefix = "  " * indent + "|-> "
    print(f"{prefix}Idx:{idx:<4} {get_name(pid):<10} (PID: {pid})")
    
    d1, d2 = d1s[idx], d2s[idx]
    if d1 >= 0 and d2 >= 0 and d1 <= d2:
        for d_idx in range(d1, d2 + 1):
            print_tree(d_idx, indent + 1, max_depth)
    elif d1 >= 0:
        print_tree(d1, indent + 1, max_depth)

# Convert to numpy to safely search for any Dark Quark PID
pids_np = ak.to_numpy(np.abs(pids))
dark_quark_idx = np.where(np.isin(pids_np, DARK_QUARKS))[0]

if len(dark_quark_idx) > 0:
    print(f"Tracing cascade from the first Dark Quark (Idx: {dark_quark_idx[0]}):")
    print_tree(dark_quark_idx[0], max_depth=6)
else:
    print("No Dark Quarks found in Event 0.")

# Convert to numpy to safely search the array
pids_np = np.abs(ak.to_numpy(pids))

# Search for ANY PID that exists in your DARK_QUARKS list
dark_quark_idx = np.where(np.isin(pids_np, DARK_QUARKS))[0]

if len(dark_quark_idx) > 0:
    first_dq_idx = dark_quark_idx[0]
    first_dq_pid = pids_np[first_dq_idx]
    
    print(f"\nTracing cascade from the first Dark Quark (Idx: {first_dq_idx}, PID: {first_dq_pid}):")
    # Bumped max_depth to 6 because Pythia showers can get deep!
    print_tree(first_dq_idx, max_depth=6) 
else:
    print("No Dark Quarks found in Event 0.")