"""
Python translation of the C++ dihadron correlation generator.

Generates trigger-associate hadron pairs from PYTHIA 8 pp events in a
given pTHat bin, writes the pair-level CSV (with trailing statistics
comments) plus a companion particle-counts text file, mirroring the
original C++ output format exactly.

Requires the PYTHIA 8 Python bindings on PYTHONPATH/LD_LIBRARY_PATH
(e.g. `module load software/pythia-8317b`).

Usage:
    python dihadron_generator.py <COM_energy> <num_events> <power> <seed> <bin_index>
"""

import math
import os
import sys
from collections import defaultdict

import pythia8

# ── pTHat bin edges — must match run.sh ─────────────────────────────
PT_LIMITS = [18.0, 600.0]

# ── Dihadron parameters (unchanged from the original) ────────────────
TRIG_RANGES = [(19.2, 24.0)]
TRIG_PT_MIN, TRIG_PT_MAX = TRIG_RANGES[0][0], TRIG_RANGES[-1][1]
ASSOC_PT_MIN, ASSOC_PT_MAX = 2.0, 3.0
TRIG_ETA_MIN, TRIG_ETA_MAX = -2.0, 2.0
ASSOC_ETA_MIN, ASSOC_ETA_MAX = -2.0, 2.0


def make_filename(com, power, seed, pt_hat_min, pt_hat_max):
    return (
        f"pythiaData/{int(com)}/cms/vacuumOnly/"
        f"dihadron_pow{int(power)}_pT{int(pt_hat_min)}to{int(pt_hat_max)}"
        f"_seed{int(seed)}.csv"
    )


def make_count_filename(data_filename):
    """Derive the particle-counts filename from the data filename.

    e.g. pythiaData/200/cms/dihadron_pow2_pT10to20_seed1.csv
      -> pythiaData/200/cms/dihadron_pow2_pT10to20_seed1_particle_counts.txt
    """
    out = data_filename
    out = out.removesuffix(".csv")
    return out + "_particle_counts.txt"


def sorted_entries(counts):
    """Sort a name->count dict by descending count, then alphabetically."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def main():
    if len(sys.argv) != 6:
        print(
            f"Usage: {sys.argv[0]} <COM_energy> <num_events> <power> <seed> <bin_index>",
            file=sys.stderr,
        )
        return 1

    com = float(sys.argv[1])
    nevents = int(sys.argv[2])
    power = float(sys.argv[3])
    seed = int(sys.argv[4])
    ibin = int(sys.argv[5])

    nbin = len(PT_LIMITS) - 1
    if ibin < 0 or ibin >= nbin:
        print(f"bin_index {ibin} out of range [0, {nbin - 1}]", file=sys.stderr)
        return 1

    pt_min = PT_LIMITS[ibin]
    pt_max = PT_LIMITS[ibin + 1]  # -1 would mean "no upper limit"

    pt_max_display = math.inf if pt_max < 0 else pt_max
    print(f"Bin {ibin}: pTHat = [{pt_min}, {pt_max_display}] GeV  |  events={nevents}")

    # ── Pythia setup ──────────────────────────────────────────────────
    pythia = pythia8.Pythia()

    pythia.readString("Tune:pp = 14")
    pythia.readString(f"Beams:eCM = {com}")
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")

    pythia.readString("Init:showProcesses = off")
    pythia.readString("Init:showMultipartonInteractions = off")
    pythia.readString("Init:showChangedSettings = off")
    pythia.readString("Init:showChangedParticleData = off")
    pythia.readString("Next:numberCount = 1000")
    pythia.readString("Next:numberShowInfo = 0")
    pythia.readString("Next:numberShowProcess = 0")
    pythia.readString("Next:numberShowEvent = 0")

    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {seed}")

    pythia.readString("SoftQCD:all = off")
    pythia.readString("HardQCD:all = on")
    pythia.readString("PartonLevel:MPI = on")
    pythia.readString("PartonLevel:ISR = on")
    pythia.readString("PartonLevel:FSR = on")
    pythia.readString("HadronLevel:Hadronize = on")
    pythia.readString("HadronLevel:Decay = on")

    # pythia.readString("ParticleDecays:limitTau0 = on")
    # pythia.readString("ParticleDecays:tau0Max = 10")

    pythia.readString(f"PhaseSpace:pTHatMin = {pt_min}")
    if pt_max > 0:
        pythia.readString(f"PhaseSpace:pTHatMax = {pt_max}")

    pythia.readString("PhaseSpace:bias2Selection = on")
    pythia.readString(f"PhaseSpace:bias2SelectionPow = {power}")
    pythia.readString("PhaseSpace:bias2SelectionRef = 5")

    print(
        f"Seed check: {pythia.settings.mode('Random:seed')}"
        f"  setSeed: {pythia.settings.flag('Random:setSeed')}"
    )

    if not pythia.init():
        print(f"Pythia init failed for bin {ibin}", file=sys.stderr)
        return 1

    # ── Output file ───────────────────────────────────────────────────
    fname = make_filename(com, power, seed, pt_min, 999 if pt_max < 0 else pt_max)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    try:
        out = open(fname, "w", buffering=64 * 1024 * 1024)  # noqa: SIM115
    except OSError as e:
        print(f"Cannot open output file: {fname} ({e})", file=sys.stderr)
        return 1

    # ── Header ────────────────────────────────────────────────────────
    out.write(f"# BIN: {ibin} of {nbin}\n")
    out.write(f"# NEVENTS: {nevents}\n")
    out.write(f"# POWER: {power}\n")
    out.write("# PREF: 5\n")
    out.write(f"# PTHAT_RANGE: {pt_min} - {pt_max}\n")
    out.write(f"# TRIG_PT_RANGE: {TRIG_PT_MIN} - {TRIG_PT_MAX}\n")
    out.write(f"# TRIG_ETA_RANGE: {TRIG_ETA_MIN} - {TRIG_ETA_MAX}\n")
    out.write(f"# ASSOC_PT_RANGE: {ASSOC_PT_MIN} - pTtrig (dynamic)\n")
    out.write(f"# ASSOC_ETA_RANGE: {ASSOC_ETA_MIN} - {ASSOC_ETA_MAX}\n")
    out.write("\n# Dihadron correlation data\n")
    out.write(
        "event,weight,trigger_id,trigger_pT,trigger_eta,trigger_phi,"
        "assoc_id,assoc_pT,assoc_eta,assoc_phi\n"
    )

    # ── Event loop ────────────────────────────────────────────────────
    sum_weights = 0.0
    trigger_weight_sum = 0.0
    range_weight_sums = [0.0] * len(TRIG_RANGES)
    global_event = 0
    trigger_count = 0
    pair_count = 0

    trigger_particle_counts = defaultdict(int)
    assoc_particle_counts = defaultdict(int)

    for _ in range(nevents):
        if not pythia.next():
            continue
        global_event += 1

        event_weight = pythia.infoPython().weight()
        sum_weights += event_weight

        trigger_indices = []
        event = pythia.event
        for i in range(event.size()):
            p = event[i]
            if not (p.isFinal() and p.isCharged() and p.isHadron()):
                continue
            pt, eta = p.pT(), p.eta()
            if TRIG_PT_MIN <= pt <= TRIG_PT_MAX and TRIG_ETA_MIN <= eta <= TRIG_ETA_MAX:
                trigger_weight_sum += event_weight
                trigger_indices.append(i)

                pname = pythia.particleData.name(p.id())
                trigger_particle_counts[pname] += 1

                for r, (lo, hi) in enumerate(TRIG_RANGES):
                    if lo <= pt < hi:
                        range_weight_sums[r] += event_weight
                        break

        trigger_count += len(trigger_indices)

        for i_trig in trigger_indices:
            trigger = event[i_trig]
            for i in range(event.size()):
                if i == i_trig:
                    continue
                assoc = event[i]
                if not (assoc.isFinal() and assoc.isCharged() and assoc.isHadron()):
                    continue

                pt, eta = assoc.pT(), assoc.eta()
                if (
                    ASSOC_PT_MIN <= pt < ASSOC_PT_MAX
                    and ASSOC_ETA_MIN <= eta <= ASSOC_ETA_MAX
                ):
                    out.write(
                        f"{global_event:d},{event_weight:.6e},{i_trig:d},"
                        f"{trigger.pT():.6e},{trigger.eta():.6e},{trigger.phi():.6e},"
                        f"{i:d},{assoc.pT():.6e},{assoc.eta():.6e},{assoc.phi():.6e}\n"
                    )
                    pair_count += 1

                    pname = pythia.particleData.name(assoc.id())
                    assoc_particle_counts[pname] += 1

    # ── Trailing statistics ───────────────────────────────────────────
    out.write(f"# sigmaGEN: {pythia.infoPython().sigmaGen():.6e}\n")
    out.write(f"# weightSum: {sum_weights:.6e}\n")
    out.write(f"# nEvents  : {global_event}\n")
    out.write(f"# nTriggers: {trigger_count}\n")
    out.write(f"# nPairs   : {pair_count}\n")
    out.write(f"# triggerWeightSum: {trigger_weight_sum:.6e}\n")
    for r, (lo, hi) in enumerate(TRIG_RANGES):
        out.write(f"# triggerWeightSum_{lo}to{hi}: {range_weight_sums[r]:.6e}\n")
    out.close()

    # ── Write particle-count file ────────────────────────────────────
    cnt_fname = make_count_filename(fname)
    try:
        cnt_out = open(cnt_fname, "w")  # noqa: SIM115
    except OSError as e:
        print(
            f"Warning: cannot open particle-count file: {cnt_fname} ({e})",
            file=sys.stderr,
        )
    else:
        cnt_out.write(f"# Particle counts for bin {ibin} of {nbin}\n")
        cnt_out.write(f"# PTHAT_RANGE: {pt_min} - {pt_max}\n")
        cnt_out.write(f"# NEVENTS_GENERATED: {global_event}\n")
        cnt_out.write("#\n")
        cnt_out.write("# Trigger condition:\n")
        cnt_out.write(f"#   pT in [{TRIG_PT_MIN}, {TRIG_PT_MAX}] GeV\n")
        cnt_out.write(f"#   |eta| in [{TRIG_ETA_MIN}, {TRIG_ETA_MAX}]\n")
        cnt_out.write("#   final-state charged particles only\n")
        cnt_out.write("#\n")
        cnt_out.write("# Associate condition:\n")
        cnt_out.write(
            f"#   pT in [{ASSOC_PT_MIN}, pTtrig] GeV  (dynamic upper bound)\n"
        )
        cnt_out.write(f"#   |eta| in [{ASSOC_ETA_MIN}, {ASSOC_ETA_MAX}]\n")
        cnt_out.write("#   final-state charged particles only\n")
        cnt_out.write("#   (counted once per trigger-assoc pair)\n")
        cnt_out.write("\n")

        # ── Trigger table ────────────────────────────────────────────
        cnt_out.write(f"=== TRIGGER particles (total: {trigger_count}) ===\n")
        cnt_out.write(f"{'particle_name':<30}{'count':>12}\n")
        cnt_out.write("-" * 44 + "\n")
        cnt_out.writelines(
            f"{name:<30}{cnt:>12}\n"
            for name, cnt in sorted_entries(trigger_particle_counts)
        )
        cnt_out.write("\n")

        # ── Associate table ───────────────────────────────────────────
        total_assoc = sum(assoc_particle_counts.values())
        cnt_out.write(f"=== ASSOCIATE particles (total: {total_assoc}) ===\n")
        cnt_out.write(f"{'particle_name':<30}{'count':>12}\n")
        cnt_out.write("-" * 44 + "\n")
        cnt_out.writelines(
            f"{name:<30}{cnt:>12}\n"
            for name, cnt in sorted_entries(assoc_particle_counts)
        )

        cnt_out.close()
        print(f"  Particle counts  : {cnt_fname}")

    print(
        f"=== Bin {ibin} done ===\n"
        f"  Events generated : {global_event}\n"
        f"  Triggers found   : {trigger_count}\n"
        f"  Pairs written    : {pair_count}\n"
        f"  Output file      : {fname}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
