"""
investigate_pfcands.py
Usage: python3 investigate_pfcands.py /path/to/events.root
"""
import sys
import uproot
import awkward as ak
import numpy as np

ROOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/training_data/SVJ_Training_2D_20260316_2324/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-10.08_mrho-88.96_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root"

def calc_dr(eta1, phi1, eta2, phi2):
    dphi = phi2 - phi1
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2*np.pi, dphi)
    return np.sqrt((eta2 - eta1)**2 + dphi**2)

print(f"\nOpening {ROOT_FILE}...")
f = uproot.open(ROOT_FILE)
t = f["Delphes;1"]

# ── 1. All branches ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  1. ALL BRANCHES")
print("="*60)
prefixes = {}
for b in t.keys():
    pfx = b.split("/")[0]
    prefixes.setdefault(pfx, []).append(b)
for pfx, blist in sorted(prefixes.items()):
    print(f"\n[{pfx}]")
    for b in blist:
        print(f"  {b}")

# ── 2. GenFatJet / Constituent branches ─────────────────────────────
print("\n" + "="*60)
print("  2. GenFatJet + Constituent BRANCHES")
print("="*60)
for b in t.keys():
    if "GenFatJet" in b or "Constituent" in b:
        print(f"  {b}")

# ── 3. DarkHadron branches + stats ──────────────────────────────────
print("\n" + "="*60)
print("  3. DARKHADRON BRANCHES")
print("="*60)
for b in t.keys():
    if "DarkHadron" in b:
        print(f"  {b}")

print("\n--- DarkHadronJet stats (500 events) ---")
for key in ["DarkHadronJet/DarkHadronJet.PT",
            "DarkHadronJet/DarkHadronJet.Eta",
            "DarkHadronJet/DarkHadronJet.Mass"]:
    try:
        arr = t[key].array(entry_stop=500, library="ak")
        counts = ak.num(arr)
        flat = ak.flatten(arr)
        print(f"  {key.split('.')[-1]:<10}  mult: min={ak.min(counts)} mean={float(ak.mean(counts)):.1f} max={ak.max(counts)} | "
              f"val: min={float(ak.min(flat)):.2f} mean={float(ak.mean(flat)):.2f} max={float(ak.max(flat)):.2f}")
    except Exception as e:
        print(f"  {key}: ERROR {e}")

# ── 4. PF candidate stats ────────────────────────────────────────────
print("\n" + "="*60)
print("  4. PF CANDIDATE STATS (500 events)")
print("="*60)
for key in ["ParticleFlowCandidate/ParticleFlowCandidate.PT",
            "ParticleFlowCandidate/ParticleFlowCandidate.Eta",
            "ParticleFlowCandidate/ParticleFlowCandidate.Phi",
            "ParticleFlowCandidate/ParticleFlowCandidate.E"]:
    try:
        arr = t[key].array(entry_stop=500, library="ak")
        counts = ak.num(arr)
        flat = ak.flatten(arr)
        print(f"  {key.split('.')[-1]:<10}  mult: min={ak.min(counts)} mean={float(ak.mean(counts)):.1f} max={ak.max(counts)} | "
              f"val: min={float(ak.min(flat)):.3f} mean={float(ak.mean(flat)):.3f} max={float(ak.max(flat)):.3f}")
    except Exception as e:
        print(f"  {key}: ERROR {e}")

# total PF per event
try:
    arr = t["ParticleFlowCandidate/ParticleFlowCandidate.PT"].array(entry_stop=500, library="ak")
    counts = ak.to_numpy(ak.num(arr))
    print(f"\n  TOTAL PF per event: min={counts.min()}  mean={counts.mean():.1f}  max={counts.max()}")
    print(f"  MAX_PARTICLES=240: {'OK' if counts.max() <= 240 else f'EXCEEDS — max is {counts.max()}, raise to at least {counts.max()}'}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 5. DarkHadronJet <-> GenFatJet deltaR matching ───────────────────
print("\n" + "="*60)
print("  5. DeltaR: DarkHadronJet <-> GenFatJet (100 events)")
print("="*60)
try:
    dh_eta = t["DarkHadronJet/DarkHadronJet.Eta"].array(entry_stop=100, library="ak")
    dh_phi = t["DarkHadronJet/DarkHadronJet.Phi"].array(entry_stop=100, library="ak")
    gf_eta = t["GenFatJet/GenFatJet.Eta"].array(entry_stop=100, library="ak")
    gf_phi = t["GenFatJet/GenFatJet.Phi"].array(entry_stop=100, library="ak")

    drs = []
    for i in range(100):
        if len(gf_eta[i]) == 0: continue
        for dh_e, dh_p in zip(ak.to_numpy(dh_eta[i]), ak.to_numpy(dh_phi[i])):
            dr = calc_dr(dh_e, dh_p, ak.to_numpy(gf_eta[i]), ak.to_numpy(gf_phi[i]))
            drs.append(np.min(dr))

    drs = np.array(drs)
    print(f"  min={drs.min():.3f}  mean={drs.mean():.3f}  max={drs.max():.3f}")
    print(f"  Fraction within DR<0.4: {(drs < 0.4).mean():.1%}")
    print(f"  Fraction within DR<0.6: {(drs < 0.6).mean():.1%}")
    print(f"  Fraction within DR<0.8: {(drs < 0.8).mean():.1%}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 6. DarkHadron mass label ─────────────────────────────────────────
print("\n" + "="*60)
print("  6. DARK HADRON MASS LABEL")
print("="*60)
try:
    arr = t["DarkHadronCandidate/DarkHadronCandidate.Mass"].array(entry_stop=500, library="ak")
    flat = ak.to_numpy(ak.flatten(arr)) / 1000.0  # MeV -> GeV
    print(f"  mA (dark pion mass): min={flat.min():.2f}  mean={flat.mean():.2f}  max={flat.max():.2f} GeV")
except Exception as e:
    print(f"  ERROR: {e}")

 
# ── 7. Constituent / Particles branch stats ──────────────────────────
print("\n" + "="*60)
print("  7. CONSTITUENT & PARTICLES BRANCH STATS (100 events)")
print("="*60)
 
constituent_branches = [
    ("FatJet",       "FatJet/FatJet.Particles"),
    ("FatJet",       "FatJet/FatJet.Constituents"),
    ("GenFatJet",    "GenFatJet/GenFatJet.Particles"),
    ("GenFatJet",    "GenFatJet/GenFatJet.Constituents"),
    ("DarkHadronJet","DarkHadronJet/DarkHadronJet.Particles"),
    ("DarkHadronJet","DarkHadronJet/DarkHadronJet.Constituents"),
]
 
for label, key in constituent_branches:
    try:
        arr = t[key].array(entry_stop=100, library="ak")
        # counts per jet (jagged x2: events -> jets -> constituents)
        per_jet = ak.num(arr, axis=2)   # constituents per jet
        per_event = ak.sum(per_jet, axis=1)  # total per event
        flat_per_jet = ak.to_numpy(ak.flatten(per_jet))
        print(f"\n  [{label}]  {key.split('/')[-1]}")
        print(f"    per-jet:   min={flat_per_jet.min()}  mean={flat_per_jet.mean():.1f}  max={flat_per_jet.max()}")
        ev_np = ak.to_numpy(per_event)
        print(f"    per-event: min={ev_np.min()}  mean={ev_np.mean():.1f}  max={ev_np.max()}")
    except Exception as e:
        # Try as flat refs (some Delphes versions store as TRefArray -> ints)
        try:
            arr = t[key].array(entry_stop=100, library="ak")
            counts = ak.to_numpy(ak.num(ak.flatten(arr, axis=1)))
            print(f"\n  [{label}]  {key.split('/')[-1]}  (flat ref array)")
            print(f"    per-jet refs: min={counts.min()}  mean={counts.mean():.1f}  max={counts.max()}")
        except Exception as e2:
            print(f"\n  [{label}]  {key.split('/')[-1]}  ERROR: {e2}")
 

# ── 8. Gen-matched FatJet stable constituents (Status==1) ────────────
print("\n" + "="*60)
print("  8. GEN-MATCHED FATJET STABLE CONSTITUENTS (Status==1)")
print("="*60)
try:
    gen_pid    = t["GenParticle/GenParticle.PID"].array(entry_stop=10, library="ak")
    gen_status = t["GenParticle/GenParticle.Status"].array(entry_stop=10, library="ak")
    gen_pt     = t["GenParticle/GenParticle.PT"].array(entry_stop=10, library="ak")
    gen_eta    = t["GenParticle/GenParticle.Eta"].array(entry_stop=10, library="ak")
    gen_phi    = t["GenParticle/GenParticle.Phi"].array(entry_stop=10, library="ak")
    gen_e      = t["GenParticle/GenParticle.E"].array(entry_stop=10, library="ak")
    gen_mass   = t["GenParticle/GenParticle.Mass"].array(entry_stop=10, library="ak")
    fj_parts   = t["FatJet/FatJet.Particles"].array(entry_stop=10, library="ak")["refs"]
 
    # --- Sample event 0 printout ---
    ev = 0
    print(f"\n  Event {ev} — FatJet 0 stable constituents (Status==1):")
    print(f"  {'ref':<6} {'PID':<10} {'pT':<8} {'eta':<8} {'phi':<8} {'E':<8} {'mass':<8}")
    if len(fj_parts[ev]) > 0:
        pid_ev    = ak.to_numpy(gen_pid[ev])
        status_ev = ak.to_numpy(gen_status[ev])
        pt_ev     = ak.to_numpy(gen_pt[ev])
        eta_ev    = ak.to_numpy(gen_eta[ev])
        phi_ev    = ak.to_numpy(gen_phi[ev])
        e_ev      = ak.to_numpy(gen_e[ev])
        mass_ev   = ak.to_numpy(gen_mass[ev])
        refs      = ak.to_numpy(fj_parts[ev][0])
 
        stable = [(r, pid_ev[r], pt_ev[r], eta_ev[r], phi_ev[r], e_ev[r], mass_ev[r])
                  for r in refs if r < len(pid_ev) and status_ev[r] == 1]
        print(f"  Total stable: {len(stable)}")
        for ref, pid, pt, eta, phi, e, mass in stable[:15]:
            print(f"  {ref:<6} {pid:<10} {pt:<8.2f} {eta:<8.3f} {phi:<8.3f} {e:<8.2f} {mass:<8.3f}")
        if len(stable) > 15:
            print(f"  ... ({len(stable)-15} more)")
 
    # --- Multiplicity across 10 events ---
    print(f"\n  Stable constituent counts per FatJet (10 events):")
    all_stable_counts = []
    for ev in range(10):
        pid_ev    = ak.to_numpy(gen_pid[ev])
        status_ev = ak.to_numpy(gen_status[ev])
        for j_idx, refs in enumerate(fj_parts[ev]):
            refs_np = ak.to_numpy(refs)
            n = sum(1 for r in refs_np if r < len(pid_ev) and status_ev[r] == 1)
            all_stable_counts.append(n)
            print(f"  ev={ev} jet={j_idx}  total_refs={len(refs_np)}  stable={n}")
 
    all_stable_counts = np.array(all_stable_counts)
    print(f"\n  Summary: min={all_stable_counts.min()}  "
          f"mean={all_stable_counts.mean():.1f}  "
          f"max={all_stable_counts.max()}")
    print(f"  MAX_PARTICLES=240: "
          f"{'OK' if all_stable_counts.max() <= 240 else f'EXCEEDS — raise to {all_stable_counts.max()}'}")
 
except Exception as e:
    print(f"  ERROR: {e}")
 

