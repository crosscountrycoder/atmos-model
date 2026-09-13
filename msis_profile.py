"""Compute atmospheric number-density and temperature profiles using
NRLMSIS 2.1 (via pymsis).

Samples 4 seasonal dates (solstices/equinoxes) at both noon and midnight
solar time (8 samples total, to capture diurnal composition variation in
the thermosphere), writes one CSV per sample into msis_samples/, and
writes msis_avg.csv averaging all 8 together.
"""

import os
from datetime import datetime

import numpy as np
import pymsis
from pymsis import Variable

# ---------------------------------------------------------------------------
# Input parameters
# ---------------------------------------------------------------------------

YEAR = 2026
SEASONAL_DATES = {
    "mar20": datetime(YEAR, 3, 20),   # spring equinox
    "jun21": datetime(YEAR, 6, 21),   # summer solstice
    "sep22": datetime(YEAR, 9, 22),   # fall equinox
    "dec21": datetime(YEAR, 12, 21),  # winter solstice
}
HOURS = [0, 12]  # midnight and noon UTC (= local solar time at LONGITUDE = 0)

LATITUDE = 45.0    # degrees N
LONGITUDE = 0.0    # degrees E

ALT_MIN, ALT_MAX, ALT_STEP = 0, 1000, 1  # km, inclusive of ALT_MAX

# Solar/geomagnetic activity. Set to None to use historical values for each
# date (auto-downloaded and cached by pymsis) instead of these fixed values.
# 150/150/4 = moderate solar flux, quiet geomagnetic conditions.
F107 = 150.0     # daily F10.7 solar flux (sfu) of the previous day
F107A = 150.0    # 81-day centered average F10.7 (sfu)
AP = 4.0         # daily Ap geomagnetic index

SIG_FIGS = 10  # significant figures for values written to the CSVs

# NO's number density never gets large and only becomes non-negligible above
# ~100 km, so for partial-pressure purposes it's often not worth tracking.
EXCLUDE_NITRIC_OXIDE = True

SAMPLES_DIR = "msis_samples"
AVG_CSV = "msis_avg.csv"

altitudes = np.arange(ALT_MIN, ALT_MAX + ALT_STEP, ALT_STEP)


# ---------------------------------------------------------------------------
# Run MSIS for one date/time -> (number densities per species, temperature)
# ---------------------------------------------------------------------------

def compute_profile(date):
    result = pymsis.calculate(
        dates=[date],
        lons=[LONGITUDE],
        lats=[LATITUDE],
        alts=altitudes,
        f107s=F107,
        f107as=F107A,
        aps=[[AP] * 7] if AP is not None else None,
    )
    # Grid mode output shape: (ndates, nlons, nlats, nalts, 11) -> (nalts, 11)
    # pymsis returns float32; upcast to float64 for full precision downstream.
    profile = result[0, 0, 0, :, :].astype(np.float64)

    # "Anomalous oxygen" (hot/energetic O not in diffusive equilibrium) is
    # folded into regular atomic O so the output has one simple O column.
    o_total = np.nansum(
        np.stack([profile[:, Variable.O], profile[:, Variable.ANOMALOUS_O]]), axis=0
    )
    o_total[np.isnan(profile[:, Variable.O]) & np.isnan(profile[:, Variable.ANOMALOUS_O])] = np.nan

    # Species covered by NRLMSIS 2.1, with anomalous O folded into O above.
    # Missing/unmodeled values (nan) below the species' onset altitude are 0.
    species_densities = {
        "N2": np.nan_to_num(profile[:, Variable.N2]),
        "O2": np.nan_to_num(profile[:, Variable.O2]),
        "O": np.nan_to_num(o_total),
        "He": np.nan_to_num(profile[:, Variable.HE]),
        "H": np.nan_to_num(profile[:, Variable.H]),
        "Ar": np.nan_to_num(profile[:, Variable.AR]),
        "N": np.nan_to_num(profile[:, Variable.N]),
    }
    if not EXCLUDE_NITRIC_OXIDE:
        species_densities["NO"] = np.nan_to_num(profile[:, Variable.NO])

    return species_densities, profile[:, Variable.TEMPERATURE]


SPECIES_NAMES = ["N2", "O2", "O", "He", "H", "Ar", "N"]
if not EXCLUDE_NITRIC_OXIDE:
    SPECIES_NAMES.append("NO")


# ---------------------------------------------------------------------------
# Write a number-density + temperature CSV
# ---------------------------------------------------------------------------

def write_csv(filename, species_densities, temperature):
    header = "altitude,temp," + ",".join(SPECIES_NAMES)
    with open(filename, "w") as f:
        f.write(header + "\n")
        for i, alt_km in enumerate(altitudes):
            alt_m = alt_km * 1000
            temp_str = f"{temperature[i]:.{SIG_FIGS}g}"
            dens_str = ",".join(f"{species_densities[name][i]:.{SIG_FIGS - 1}e}"
                                 for name in SPECIES_NAMES)
            f.write(f"{alt_m},{temp_str},{dens_str}\n")


# ---------------------------------------------------------------------------
# Compute all 8 samples (4 dates x noon/midnight), write them, and average
# ---------------------------------------------------------------------------

os.makedirs(SAMPLES_DIR, exist_ok=True)

densities = {}
temperatures = {}
for date_label, date in SEASONAL_DATES.items():
    for hour in HOURS:
        sample_label = f"{date_label}_{hour:02d}"
        dt = date.replace(hour=hour)
        densities[sample_label], temperatures[sample_label] = compute_profile(dt)
        filename = os.path.join(SAMPLES_DIR, f"msis_{sample_label}.csv")
        write_csv(filename, densities[sample_label], temperatures[sample_label])
        print(f"Wrote {len(altitudes)} altitude levels to {filename} "
              f"({dt.isoformat()} UTC, {LATITUDE}N {LONGITUDE}E)")

avg_densities = {
    name: np.mean([densities[label][name] for label in densities], axis=0)
    for name in SPECIES_NAMES
}
avg_temperature = np.mean(list(temperatures.values()), axis=0)
write_csv(AVG_CSV, avg_densities, avg_temperature)
print(f"Wrote {len(altitudes)} altitude levels to {AVG_CSV} "
      f"(average of {len(densities)} samples: {', '.join(densities)})")
