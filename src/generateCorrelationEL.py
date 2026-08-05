"""
Python dihadron correlation generator with pQCD energy loss (DGLV) applied 
to hard partons before hadronization using UserHooks.

For a given pTHat bin, this script runs TWO passes sharing the same seed
and pTHat window:

  - vacuum  : plain PYTHIA 8 pp events (ISR + FSR + MPI, no quenching)
  - quenched: same events, but each final-state parton above the pT
              threshold has its momentum rescaled by (1 - eps), with eps
              sampled from the DGLV energy-loss-fraction distribution for
              that parton's flavour, before colour reconnection and
              hadronization run.

Both passes then apply the same trigger/associate hadron selection and
write pair-level CSVs (with trailing statistics comments), one per pass,
so the quenched output can be directly compared to the vacuum baseline
(I_AA-style ratios).

Usage:
    python generateCorrelationEL.py <COM_energy> <num_events> <seed> <bin_index>
"""

import sys
import os
import math
import numpy as np
import pythia8

from energyLossSampler import EnergyLossSampler

# ── pTHat bin edges — must match run.sh ─────────────────────────────
PT_LIMITS = [18.0, 600.0]

# ── Dihadron parameters (unchanged from the original) ────────────────
TRIG_RANGES = [(19.2, 24.0)]
TRIG_PT_MIN, TRIG_PT_MAX = TRIG_RANGES[0][0], TRIG_RANGES[-1][1]
ASSOC_PT_MIN, ASSOC_PT_MAX = 2.0, 3.0
TRIG_ETA_MIN, TRIG_ETA_MAX = -2.0, 2.0
ASSOC_ETA_MIN, ASSOC_ETA_MAX = -2.0, 2.0

# Minimum parton pT for quenching to be applied (matches jetQuenching.py)
QUENCH_PT_MIN = 5.0

# Eta acceptance applied when printing debug parton info (matches hadron cuts)
PARTON_ETA_MIN, PARTON_ETA_MAX = -2.0, 2.0

# Debug printout controls
DEBUG_MAX_EVENTS = 3
DEBUG_MAX_PARTONS = 10

# ── Flavour mapping ────────────────────────────
def pdgToFlavour(pid: int):
    pid = abs(pid)
    if pid in (1, 2, 3):
        return "light"
    if pid == 4:
        return "charm"
    if pid == 5:
        return "bottom"
    if pid == 21:
        return "gluon"
    return None

# ── Energy-loss hook ─────────────────────────────────────────────────────────
class EnergyLossHook(pythia8.UserHooks):
    def __init__(self, sampler: EnergyLossSampler, quench: bool = True):
        pythia8.UserHooks.__init__(self)
        self.sampler = sampler
        self.quench  = quench
        self.lastEvent = []

    def canVetoPartonLevel(self):
        return True

    def doVetoPartonLevel(self, event):
        self.lastEvent = []

        for i in range(event.size()):
            p = event[i]

            if not p.isFinal():
                continue

            flavour = pdgToFlavour(p.id())
            if flavour is None:
                continue

            pTIn = p.pT()
            if pTIn < 5:
                continue

            etaIn = p.eta()

            mu  = float(self.sampler.mean(flavour, pTIn))
            sig = float(self.sampler.sigma(flavour, pTIn))

            if self.quench:
                eps = float(self.sampler.sample(flavour, pTIn))

                px = p.px() * (1.0 - eps)
                py = p.py() * (1.0 - eps)
                pz = p.pz()
                m  = p.m()
                e  = math.sqrt(px*px + py*py + pz*pz + m*m)

                if e <= 0.0:
                    e  = m if m > 0.0 else 1e-3
                    px = 0.0
                    py = 0.0

                p.px(px)
                p.py(py)
                p.e(e)

                pTOut = math.sqrt(px*px + py*py)
            else:
                eps   = 0.0
                px    = p.px()
                py    = p.py()
                pz    = p.pz()
                m     = p.m()
                e     = p.e()
                pTOut = pTIn

            self.lastEvent.append({
                "idx":     i,
                "id":      p.id(),
                "flavour": flavour,
                "px":      px,
                "py":      py,
                "pz":      pz,
                "e":       e,
                "m":       m,
                "pTIn":    pTIn,
                "eta":     etaIn,
                "mu":      mu,
                "sig":     sig,
                "eps":     eps,
                "pT":      pTOut,
            })

        return False

def make_filename(com, power, seed, pt_hat_min, pt_hat_max, label):
    return (
        f"pythiaData/{int(com)}/cms/"
        f"dihadron_pow{int(power)}_pT{int(pt_hat_min)}to{int(pt_hat_max)}"
        f"_{label}_seed{int(seed)}.csv"
    )

def build_pythia(com, seed, pt_min, pt_max, power, hook):
    """Construct and initialise a Pythia instance for one pass (vacuum or
    quenched). The hook is attached before init() so the UserHooks pointer
    is live from the first event, matching jetQuenching.py's pattern."""
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
    pythia.readString("HadronLevel:all = on")

    pythia.readString("Check:event = off")

    pythia.readString("ParticleDecays:limitTau0 = on")
    pythia.readString("ParticleDecays:tau0Max = 10")

    pythia.readString(f"PhaseSpace:pTHatMin = {pt_min}")
    if pt_max > 0:
        pythia.readString(f"PhaseSpace:pTHatMax = {pt_max}")

    pythia.readString("PhaseSpace:bias2Selection = on")
    pythia.readString(f"PhaseSpace:bias2SelectionPow = {power}")
    pythia.readString("PhaseSpace:bias2SelectionRef = 5")

    pythia.setUserHooksPtr(hook)

    return pythia

def run_pass(pythia, hook, com, power, seed, pt_min, pt_max, ibin, nbin, nevents, label):
    """Run one full pass (vacuum or quenched) through the same
    trigger/associate dihadron selection as the original script, and write
    the pair-level CSV for this pass."""

    fname = make_filename(com, power, seed, pt_min, 999 if pt_max < 0 else pt_max, label)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    try:
        out = open(fname, "w", buffering=64 * 1024 * 1024)
    except OSError as e:
        print(f"Cannot open output file: {fname} ({e})", file=sys.stderr)
        return None

    # ── Header ────────────────────────────────────────────────────────
    out.write(f"# BIN: {ibin} of {nbin}\n")
    out.write(f"# LABEL: {label}\n")
    out.write(f"# NEVENTS: {nevents}\n")
    out.write(f"# POWER: {power}\n")
    out.write(f"# PREF: 5\n")
    out.write(f"# PTHAT_RANGE: {pt_min} - {pt_max}\n")
    out.write(f"# QUENCH_PT_MIN: {QUENCH_PT_MIN}\n")
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

    for _ in range(nevents):
        if not pythia.next():
            continue
        global_event += 1

        if global_event <= DEBUG_MAX_EVENTS:
            printQuenchingInfo(hook, global_event)

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
                if ASSOC_PT_MIN <= pt < ASSOC_PT_MAX and ASSOC_ETA_MIN <= eta <= ASSOC_ETA_MAX:
                    out.write(
                        f"{global_event:d},{event_weight:.6e},{i_trig:d},"
                        f"{trigger.pT():.6e},{trigger.eta():.6e},{trigger.phi():.6e},"
                        f"{i:d},{assoc.pT():.6e},{assoc.eta():.6e},{assoc.phi():.6e}\n"
                    )
                    pair_count += 1

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

    print(
        f"=== Bin {ibin} [{label}] done ===\n"
        f"  Events generated : {global_event}\n"
        f"  Triggers found   : {trigger_count}\n"
        f"  Pairs written    : {pair_count}\n"
        f"  Output file      : {fname}"
    )

    return fname

# ── Debug printer ─────────────────────────────────────────────────────────────
def printQuenchingInfo(hook, evNum, maxPartons=DEBUG_MAX_PARTONS):
    rows = [
        row for row in hook.lastEvent
        if PARTON_ETA_MIN <= row["eta"] <= PARTON_ETA_MAX
    ]

    print(f"\n=== Event {evNum} — parton quenching (|eta| cut: "
          f"{PARTON_ETA_MIN} to {PARTON_ETA_MAX}) ===")
    print(f"{'idx':>5}  {'id':>5}  {'flavour':>7}  {'eta':>7}  "
          f"{'pTIn':>9}  {'pz':>9}  {'mu':>8}  {'sigma':>8}  {'eps':>8}  {'pT':>9}  {'loss%':>7}")
    print("-" * 105)

    for n, row in enumerate(rows):
        if n >= maxPartons:
            print(f"  ... (showing first {maxPartons} QCD partons only)")
            break

        lossPct = 100.0 * (1.0 - row["pT"] / row["pTIn"]) if row["pTIn"] > 0 else 0.0

        print(f"{row['idx']:>5}  {row['id']:>5}  {row['flavour']:>7}  {row['eta']:>7.3f}  "
              f"{row['pTIn']:>9.3f}  {row['pz']:>9.3f}  {row['mu']:>8.4f}  {row['sig']:>8.4f}  "
              f"{row['eps']:>8.4f}  {row['pT']:>9.3f}  {lossPct:>6.1f}%")

    if not rows:
        print("  (no QCD partons within eta cut this event)")

    print()

def main():
    if len(sys.argv) != 5:
        print(
            f"Usage: {sys.argv[0]} <COM_energy> <num_events> <power> <seed> <bin_index>",
            file=sys.stderr,
        )
        return 1

    com = float(sys.argv[1])
    nevents = int(sys.argv[2])
    seed = int(sys.argv[3])
    ibin = int(sys.argv[4])
    power = 3

    nbin = len(PT_LIMITS) - 1
    if ibin < 0 or ibin >= nbin:
        print(f"bin_index {ibin} out of range [0, {nbin - 1}]", file=sys.stderr)
        return 1

    pt_min = PT_LIMITS[ibin]
    pt_max = PT_LIMITS[ibin + 1]  # -1 would mean "no upper limit"

    pt_max_display = math.inf if pt_max < 0 else pt_max
    print(f"Bin {ibin}: pTHat = [{pt_min}, {pt_max_display}] GeV  |  events={nevents}")

    # Sampler is shared across both passes (fixed rng seed, matching
    # jetQuenching.py, so the eps distribution is reproducible independent
    # of the PYTHIA seed used to drive event generation).
    sampler = EnergyLossSampler(
        os.path.join("energyLoss", "energyLossFraction"),
        rng=np.random.default_rng(seed=42),
    )
    print(sampler)

    # ── Run both passes, same pTHat bin and PYTHIA seed ─────────────────
    for quench in (False, True):
        label = "quenched" if quench else "vacuum"

        hook = EnergyLossHook(sampler, quench=quench)
        pythia = build_pythia(com, seed, pt_min, pt_max, power, hook)

        print(
            f"Seed check [{label}]: {pythia.settings.mode('Random:seed')}"
            f" setSeed: {pythia.settings.flag('Random:setSeed')}"
        )

        if not pythia.init():
            print(f"Pythia init failed for {label} pass, bin {ibin}", file=sys.stderr)
            return 1

        run_pass(pythia, hook, com, power, seed, pt_min, pt_max, ibin, nbin, nevents, label)

    return 0

if __name__ == "__main__":
    sys.exit(main())