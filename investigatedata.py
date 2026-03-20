"""
investigate_pfcands.py
Usage: python3 investigate_pfcands.py /path/to/events.root
Requires: pip install uproot awkward numpy --upgrade
"""
import sys
import numpy as np
import uproot
import awkward as ak

ROOT_FILE = sys.argv[1] if len(sys.argv) > 1 else "/lustre/research/hep/akshriva/SVJ_RandD/TrainingDatamaker/training_data/SVJ_Training_2D_20260316_2324/s-channel_mmed-2000_Nc-2_Nf-2_scale-35.1539_mq-10_mpi-10.08_mrho-88.96_pvector-0.75_spectrum-cms_gq-0.25_gchi-0.5_rinv-0.3/events.root"

with uproot.open(ROOT_FILE) as f:

    # find main tree
    tree = None
    for k in f.keys():
        try:
            obj = f[k]
            if hasattr(obj, "keys") and len(obj.keys()) > 5:
                tree = obj; tree_name = k; break
        except: pass

    print(f"\nTREE: '{tree_name}'  ({tree.num_entries} events)")
    branches = tree.keys()

    # all branches
    print("\n--- ALL BRANCHES ---")
    prefixes = {}
    for b in branches:
        pfx = b.split(".")[0]
        prefixes.setdefault(pfx, []).append(b)
    for pfx, blist in sorted(prefixes.items()):
        print(f"\n[{pfx}]")
        for b in blist:
            print(f"  {b}")

    # PF multiplicity + pT stats
    N = min(500, tree.num_entries)
    print(f"\n--- PF STATS (first {N} events) ---")
    for label in ["EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron", "Jet"]:
        pt_key = next((b for b in branches if label in b and ".PT" in b), None)
        if not pt_key:
            print(f"{label:30s}  NOT FOUND"); continue
        arr = tree[pt_key].array(entry_stop=N, library="ak")
        counts = ak.to_numpy(ak.num(arr))
        pts = ak.to_numpy(ak.flatten(arr))
        print(f"{label:30s}  mult: min={counts.min()} mean={counts.mean():.1f} max={counts.max()} | "
              f"pT: min={pts.min():.2f} mean={pts.mean():.2f} max={pts.max():.2f} GeV")

    # total PF per event
    total = np.zeros(N, dtype=int)
    for label in ["EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron"]:
        pt_key = next((b for b in branches if label in b and ".PT" in b), None)
        if pt_key:
            total += ak.to_numpy(ak.num(tree[pt_key].array(entry_stop=N, library="ak")))
    print(f"\nTOTAL PF per event:  min={total.min()}  mean={total.mean():.1f}  max={total.max()}")
    print(f"MAX_PARTICLES=240: {'OK' if total.max() <= 240 else 'EXCEEDS — need truncation'}")

    # GenParticle PDG census
    pid_key = next((b for b in branches if "GenParticle" in b and "PID" in b), None)
    if pid_key:
        print(f"\n--- PDG ID CENSUS (from {pid_key}) ---")
        pid_data = tree[pid_key].array(entry_stop=min(2000, tree.num_entries), library="ak")
        pid_flat = ak.to_numpy(ak.flatten(pid_data))
        unique, counts = np.unique(np.abs(pid_flat), return_counts=True)
        known = {
            1:"d", 2:"u", 3:"s", 4:"c", 5:"b", 11:"e", 13:"mu", 15:"tau",
            21:"gluon", 22:"photon", 23:"Z", 24:"W", 25:"H",
            111:"pi0", 211:"pi±", 2212:"proton", 2112:"neutron",
            4900001:"dark_q", 4900021:"dark_g",
            4900111:"dark_pi0", 4900113:"dark_rho0",
            4900211:"dark_pi±", 4900213:"dark_rho±",
        }
        for i in np.argsort(-counts)[:25]:
            print(f"  PDG {int(unique[i]):>10}  {known.get(int(unique[i]),''):<12}  n={counts[i]}")

    # EFlowTrack detail
    print("\n--- EFLOWTRACK BRANCH DETAIL ---")
    tk_branches = [b for b in branches if "EFlowTrack" in b]
    for b in tk_branches:
        print(f"  {b}")

    # sample 3 events
    print("\n--- SAMPLE 3 EVENTS (EFlowTrack PT, Eta, Phi) ---")
    tk_pt  = next((b for b in branches if "EFlowTrack" in b and ".PT"  in b), None)
    tk_eta = next((b for b in branches if "EFlowTrack" in b and ".Eta" in b), None)
    tk_phi = next((b for b in branches if "EFlowTrack" in b and ".Phi" in b), None)
    if tk_pt and tk_eta and tk_phi:
        data = tree.arrays([tk_pt, tk_eta, tk_phi], entry_stop=3, library="ak")
        for ev in range(3):
            pts  = ak.to_list(data[tk_pt][ev])
            etas = ak.to_list(data[tk_eta][ev])
            phis = ak.to_list(data[tk_phi][ev])
            print(f"\n  Event {ev}: {len(pts)} tracks")
            for i, (pt, eta, phi) in enumerate(zip(pts[:5], etas[:5], phis[:5])):
                print(f"    track {i}: pT={pt:.2f}  eta={eta:.3f}  phi={phi:.3f}")
            if len(pts) > 5:
                print(f"    ... ({len(pts)-5} more)")