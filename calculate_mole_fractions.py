"""Combine NRLMSIS 2.1 number densities (msis_avg.csv) and the WACCM-X/
diffusion-derived trace gas mole fractions (waccm_gases.csv) into one
mole-fraction table covering every gas tracked by either source, plus the
resulting mean molar mass of air at each altitude.

The non-NRLMSIS gases (CO2, CH4, N2O, H2, Ne, Kr, Xe) keep their mole
fractions exactly as given in waccm_gases.csv. The NRLMSIS gases (N2, O2,
O, He, H, Ar, N) keep their relative proportions to each other (from their
number densities), but are rescaled so that collectively they make up
1 minus the non-NRLMSIS total -- i.e. everything sums to 1.

Columns are ordered by sea-level mole fraction, descending.
"""

import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOLAR_MASSES_CSV = os.path.join(SCRIPT_DIR, "molar_masses.csv")
MSIS_AVG_CSV = os.path.join(SCRIPT_DIR, "msis_avg.csv")
WACCM_GASES_CSV = os.path.join(SCRIPT_DIR, "waccm_gases.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "mole_fractions.csv")

SIG_FIGS = 10


# ---------------------------------------------------------------------------
# Molar masses, from molar_masses.csv -- these reflect the actual isotopic
# composition of each gas as found in Earth's atmosphere (e.g. atmospheric
# CO2's carbon/oxygen isotope ratios), which is more precise than summing
# conventional IUPAC atomic weights for this purpose.
# ---------------------------------------------------------------------------

def _load_molar_masses():
    masses = {}
    with open(MOLAR_MASSES_CSV) as f:
        next(f)  # skip header (gas,molar_mass,source)
        for line in f:
            gas, mass, *_ = line.strip().split(",", 2)
            masses[gas] = float(mass)
    return masses


MOLAR_MASSES = _load_molar_masses()


# ---------------------------------------------------------------------------
# Load and combine
# ---------------------------------------------------------------------------

def read_csv(filename):
    with open(filename) as f:
        header = f.readline().strip().split(",")
        alts, columns = [], {name: [] for name in header[1:]}
        for line in f:
            fields = line.strip().split(",")
            alts.append(float(fields[0]))
            for name, value in zip(header[1:], fields[1:]):
                columns[name].append(float(value))
    return np.array(alts), {name: np.array(v) for name, v in columns.items()}


msis_alts, msis_data = read_csv(MSIS_AVG_CSV)
waccm_alts, waccm_data = read_csv(WACCM_GASES_CSV)

if not np.array_equal(msis_alts, waccm_alts):
    raise ValueError("msis_avg.csv and waccm_gases.csv altitude grids don't match")
altitudes = msis_alts

NRLMSIS_SPECIES = [name for name in msis_data if name != "temp"]
WACCM_SPECIES = list(waccm_data)

nrlmsis_total = sum(msis_data[name] for name in NRLMSIS_SPECIES)
non_nrlmsis_total = sum(waccm_data[name] for name in WACCM_SPECIES)
nrlmsis_remaining = 1.0 - non_nrlmsis_total

mole_fractions = {}
for name in NRLMSIS_SPECIES:
    mole_fractions[name] = (msis_data[name] / nrlmsis_total) * nrlmsis_remaining
for name in WACCM_SPECIES:
    mole_fractions[name] = waccm_data[name]

# He floor: NRLMSIS's well-mixed-region He (~5.1999e-6) sits below the
# well-established atmospheric constant (5.24e-6, Glueckauf) -- unlike CO2,
# He doesn't meaningfully change over human timescales, so 5.24e-6 is simply
# the right sea-level/well-mixed value, not a "current vs. outdated" question.
# NRLMSIS has no real low-altitude compositional data informing this (its
# measurements are thermospheric), so it's floored up to 5.24e-6 wherever
# it would otherwise fall below that -- which only binds in the well-mixed
# layer (checked: NRLMSIS's own He rises smoothly above 5.24e-6 by ~79 km,
# so the handoff to real NRLMSIS thermospheric data is continuous, not a
# hard cutoff). Raising He requires shrinking the other NRLMSIS species
# proportionally, to keep their total (and thus every row's total) at 1.
HE_FLOOR = 5.24e-6
he_deficit = np.maximum(HE_FLOOR - mole_fractions["He"], 0.0)
other_nrlmsis = [name for name in NRLMSIS_SPECIES if name != "He"]
other_total = sum(mole_fractions[name] for name in other_nrlmsis)
shrink_factor = np.where(other_total > 0, 1.0 - he_deficit / other_total, 1.0)
for name in other_nrlmsis:
    mole_fractions[name] = mole_fractions[name] * shrink_factor
mole_fractions["He"] = mole_fractions["He"] + he_deficit

# Columns ordered by sea-level (altitude=0) mole fraction, descending.
sea_level_index = int(np.where(altitudes == 0)[0][0])
ALL_SPECIES = sorted(
    list(NRLMSIS_SPECIES) + list(WACCM_SPECIES),
    key=lambda name: mole_fractions[name][sea_level_index],
    reverse=True,
)

# Treat anything below 1e-20 as zero -- at that level it's well past the
# point of representing any physically meaningful concentration.
ZERO_THRESHOLD = 1e-20
for name in ALL_SPECIES:
    mole_fractions[name][mole_fractions[name] < ZERO_THRESHOLD] = 0.0

mean_molar_mass = sum(
    mole_fractions[name] * MOLAR_MASSES[name] for name in ALL_SPECIES
)


# ---------------------------------------------------------------------------
# Write the CSV
# ---------------------------------------------------------------------------

def format_fixed_sigfigs(value, sig_figs):
    """Fixed-point (non-scientific) string with exactly sig_figs significant
    figures, padded with trailing zeros as needed.
    """
    magnitude = int(np.floor(np.log10(abs(value))))
    decimals = max(sig_figs - 1 - magnitude, 0)
    return f"{value:.{decimals}f}"


with open(OUTPUT_CSV, "w") as f:
    f.write("altitude,molar_mass," + ",".join(ALL_SPECIES) + "\n")
    for i, alt_m in enumerate(altitudes):
        molar_mass_str = format_fixed_sigfigs(mean_molar_mass[i], SIG_FIGS)
        vals = ",".join(f"{mole_fractions[name][i]:.{SIG_FIGS - 1}e}" for name in ALL_SPECIES)
        f.write(f"{alt_m:.0f},{molar_mass_str},{vals}\n")

print(f"Wrote {len(altitudes)} altitude levels to {OUTPUT_CSV}")
print("Column order (by sea-level mole fraction): " + ", ".join(ALL_SPECIES))


# ---------------------------------------------------------------------------
# Verify: reread the CSV and confirm every row's gas mole fractions
# (excluding molar_mass) sum to 1 within 10^-9
# ---------------------------------------------------------------------------

tolerance = 10 ** (1 - SIG_FIGS)
failures = []
max_deviation = 0.0
worst_row = None
n_rows = 0
with open(OUTPUT_CSV) as f:
    next(f)  # skip header
    for line in f:
        n_rows += 1
        fields = line.strip().split(",")
        row_sum = sum(float(v) for v in fields[2:])  # skip altitude, molar_mass
        deviation = abs(row_sum - 1.0)
        if deviation > max_deviation:
            max_deviation = deviation
            worst_row = fields[0]
        if deviation > tolerance:
            failures.append((fields[0], row_sum, deviation))

print(f"Verification: tolerance is 1 +/- {tolerance:.3e} ({SIG_FIGS} sig figs)")
if not failures:
    print(f"  PASSED: all {n_rows} rows sum to 1 within tolerance "
          f"(max |sum - 1| = {max_deviation:.3e}, at altitude={worst_row} m)")
else:
    print(f"  FAILED: {len(failures)}/{n_rows} rows outside tolerance "
          f"(max |sum - 1| = {max_deviation:.3e}, at altitude={worst_row} m)")
    for alt_m, row_sum, deviation in failures[:10]:
        print(f"    altitude={alt_m} m  sum={row_sum:.{SIG_FIGS}g}  |sum-1|={deviation:.3e}")
    if len(failures) > 10:
        print(f"    ... and {len(failures) - 10} more")
