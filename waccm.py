"""Mole fractions of CO2, CH4, N2O, H2, Ne, Kr, and Xe at 1 km increments
from 0 to 1000 km.

CO2, CH4, N2O, and H2 all use real WACCM-X output: the altitude-dependent
ratio-to-surface-value of each gas (averaged from June 2016 and December
2016, 45N, zonal mean), applied to the current NOAA-sourced sea-level mole
fraction. This captures real photochemistry (CH4/N2O stratospheric
destruction, CO2 mesospheric "freeze-out", H2's mesospheric buildup) that a
pure diffusion formula can't. WACCM-X's grid only extends to ~500 km, so
above that each gas's ratio is held flat at its last available value
(np.interp's default extrapolation) rather than continued with a diffusive-
equilibrium formula -- that was tried for H2 and rejected: diffusive
equilibrium assumes zero net vertical flux, which is wrong for a species
this light (close to atomic H's mass, so real Jeans escape matters), and it
produced an even less physical number than just holding WACCM's last value.

Ne/Kr/Xe (WACCM doesn't track noble gases -- they're chemically inert, so
there's nothing for it to model, and pure diffusion is already the
physically correct treatment for them) use diffusive equilibrium referenced
against NRLMSIS's bulk density and temperature (msis_avg.csv), the same
formula used elsewhere in this project.
"""

import os

import numpy as np
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATOMIC_MASSES_CSV = os.path.join(SCRIPT_DIR, "atomic_masses.csv")
MSIS_AVG_CSV = os.path.join(SCRIPT_DIR, "msis_avg.csv")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "waccm_gases.csv")

LATITUDE = 45.0
SIG_FIGS = 10

WACCM_BASE_URL = (
    "https://tds.gdex.ucar.edu/thredds/dodsC/files/d651034/SD-WACCM-X_v2.1/"
    "atm/cam.h0/f.e20.FXSD.f19_f19.001.cam.h0.{month}.nc"
)
WACCM_MONTHS = ["2016-06", "2016-12"]
WACCM_GASES = ["H2", "CO2", "CH4", "N2O"]

# Current (2025-2026) well-mixed sea-level mole fractions.
SEA_LEVEL_MOLE_FRACTIONS = {
    "CO2": 4.260e-4,
    "CH4": 1.936e-6,
    "N2O": 3.390e-7,
    "H2": 5.530e-7,
    "Ne": 1.818e-5,
    "Kr": 1.140e-6,
    "Xe": 8.700e-8,
}

HOMOPAUSE_ALT = 86000  # m
G0 = 9.80665            # standard gravity at sea level, m/s^2
R_GAS = 8.314462618     # universal gas constant, J/(mol*K)
EARTH_RADIUS_M = 6356766.0  # USSA76 effective Earth radius (r0)

ALT_MIN, ALT_MAX, ALT_STEP = 0, 1000, 1  # km, output grid
FINE_STEP = 10  # m, resolution for the internal diffusive-equilibrium integration

output_altitudes_m = np.arange(ALT_MIN, ALT_MAX + ALT_STEP, ALT_STEP) * 1000


# ---------------------------------------------------------------------------
# Molecular masses (needed for the diffusive-equilibrium species)
# ---------------------------------------------------------------------------

def _load_atomic_masses():
    masses = {}
    with open(ATOMIC_MASSES_CSV) as f:
        for line in f:
            symbol, atomic_number, mass = line.strip().split(",")
            masses[symbol] = float(mass)
    return masses


ATOMIC_MASSES = _load_atomic_masses()
# Simple formulas only (element symbol, optional integer count) -- sufficient
# for He/Ne/Kr/Xe/H2.
MOLAR_MASS = {
    "H2": 2 * ATOMIC_MASSES["H"],
    "Ne": ATOMIC_MASSES["Ne"],
    "Kr": ATOMIC_MASSES["Kr"],
    "Xe": ATOMIC_MASSES["Xe"],
}


# ---------------------------------------------------------------------------
# NRLMSIS 2.1 background (temperature + bulk density), from msis_avg.csv
# ---------------------------------------------------------------------------

BULK_SPECIES = ["N2", "O2", "O", "He", "H", "Ar", "N"]


def _load_msis_avg():
    alts, temps, bulk_total = [], [], []
    with open(MSIS_AVG_CSV) as f:
        header = f.readline().strip().split(",")
        idx = {name: i for i, name in enumerate(header)}
        for line in f:
            fields = line.strip().split(",")
            alts.append(float(fields[idx["altitude"]]))
            temps.append(float(fields[idx["temp"]]))
            bulk_total.append(sum(float(fields[idx[s]]) for s in BULK_SPECIES))
    return np.array(alts), np.array(temps), np.array(bulk_total)


MSIS_ALTS, MSIS_TEMPS, MSIS_BULK_TOTAL = _load_msis_avg()


def msis_temperature(z):
    return np.interp(z, MSIS_ALTS, MSIS_TEMPS)


def msis_bulk_total_density(z):
    return np.interp(z, MSIS_ALTS, MSIS_BULK_TOTAL)


def gravity(z):
    return G0 * (EARTH_RADIUS_M / (EARTH_RADIUS_M + z)) ** 2


# ---------------------------------------------------------------------------
# WACCM-X: real ratio-to-surface profiles for H2, CO2, CH4, N2O
# ---------------------------------------------------------------------------

def fetch_waccm_ratio_profiles():
    """{gas: (altitudes_m, ratio_to_surface)} averaged over WACCM_MONTHS,
    at LATITUDE, zonal (longitude) mean.
    """
    per_month = {gas: [] for gas in WACCM_GASES}
    z_per_month = []

    for month in WACCM_MONTHS:
        ds = xr.open_dataset(WACCM_BASE_URL.format(month=month))
        z3 = ds["Z3"].isel(time=0).sel(lat=LATITUDE, method="nearest").mean(dim="lon").values
        order = np.argsort(z3)
        z_per_month.append(z3[order])
        for gas in WACCM_GASES:
            vals = ds[gas].isel(time=0).sel(lat=LATITUDE, method="nearest").mean(dim="lon").values
            vals = vals[order]
            per_month[gas].append(vals / vals[0])  # ratio to surface

    # Average the two months on a common altitude grid (June's grid).
    z_ref = z_per_month[0]
    profiles = {}
    for gas in WACCM_GASES:
        ratios_interp = [
            np.interp(z_ref, z_per_month[i], per_month[gas][i]) for i in range(len(WACCM_MONTHS))
        ]
        profiles[gas] = (z_ref, np.mean(ratios_interp, axis=0))
    return profiles


# ---------------------------------------------------------------------------
# Diffusive equilibrium (for Ne/Kr/Xe, and H2 above H2_WACCM_TRUST_ALT)
# ---------------------------------------------------------------------------

def diffusive_equilibrium_mole_fraction(molar_mass_g_mol, sea_level_fraction,
                                         start_alt_m=HOMOPAUSE_ALT, start_fraction=None):
    """Mole fraction on the fine altitude grid via well-mixed-below /
    diffusive-equilibrium-above the given start altitude.
    """
    molar_mass = molar_mass_g_mol / 1000  # kg/mol
    if start_fraction is None:
        start_fraction = sea_level_fraction

    fine_z = np.arange(0, ALT_MAX * 1000 + FINE_STEP, FINE_STEP, dtype=np.float64)
    fine_T = msis_temperature(fine_z)
    fine_g = gravity(fine_z)
    fine_bulk_total = msis_bulk_total_density(fine_z)

    density = np.empty_like(fine_z)
    below = fine_z <= start_alt_m
    density[below] = start_fraction * fine_bulk_total[below]

    above = fine_z >= start_alt_m
    z_above = fine_z[above]
    T_above = fine_T[above]
    g_above = fine_g[above]

    T_start = msis_temperature(start_alt_m)
    n_start = start_fraction * msis_bulk_total_density(start_alt_m)

    integrand = molar_mass * g_above / (R_GAS * T_above)
    cum = np.concatenate(([0.0], np.cumsum(
        (integrand[1:] + integrand[:-1]) / 2 * np.diff(z_above)
    )))
    density[above] = n_start * (T_start / T_above) * np.exp(-cum)

    return fine_z, density / fine_bulk_total


# ---------------------------------------------------------------------------
# Assemble mole fractions for all 7 gases on the output grid
# ---------------------------------------------------------------------------

def compute_mole_fractions():
    result = {}
    waccm_profiles = fetch_waccm_ratio_profiles()

    for gas in WACCM_GASES:  # CO2, CH4, N2O, H2
        z, ratio = waccm_profiles[gas]
        fraction = SEA_LEVEL_MOLE_FRACTIONS[gas] * ratio
        result[gas] = np.interp(output_altitudes_m, z, fraction)

    # CO2, CH4, N2O are already below 1e-9 by WACCM-X's own grid top (~500 km,
    # taken from the data itself rather than hardcoded), in an environment
    # that's already a near-vacuum -- treat them as zero above that altitude
    # rather than holding flat at a statistically meaningless value. H2 is
    # deliberately left flat (see module docstring for why).
    waccm_grid_top = waccm_profiles["CO2"][0][-1]
    for gas in ["CO2", "CH4", "N2O"]:
        result[gas][output_altitudes_m > waccm_grid_top] = 0.0

    for gas in ["Ne", "Kr", "Xe"]:
        fine_z, fraction = diffusive_equilibrium_mole_fraction(
            MOLAR_MASS[gas], SEA_LEVEL_MOLE_FRACTIONS[gas]
        )
        result[gas] = np.interp(output_altitudes_m, fine_z, fraction)

    # Sea level (altitude=0) is a boundary condition, not a derived value --
    # pin it to the exact spec instead of leaving it subject to floating-
    # point noise from interpolating the WACCM ratio curve.
    for gas, fraction in SEA_LEVEL_MOLE_FRACTIONS.items():
        result[gas][0] = fraction

    return result


def write_csv(filename, mole_fractions):
    gases = list(SEA_LEVEL_MOLE_FRACTIONS)
    with open(filename, "w") as f:
        f.write("altitude," + ",".join(gases) + "\n")
        for i, alt_m in enumerate(output_altitudes_m):
            vals = ",".join(f"{mole_fractions[g][i]:.{SIG_FIGS - 1}e}" for g in gases)
            f.write(f"{alt_m},{vals}\n")


if __name__ == "__main__":
    mole_fractions = compute_mole_fractions()
    write_csv(OUTPUT_CSV, mole_fractions)

    gases = list(SEA_LEVEL_MOLE_FRACTIONS)
    print(f"{'alt(km)':>8} " + " ".join(f"{g:>12}" for g in gases))
    for i, alt_m in enumerate(output_altitudes_m):
        if alt_m % 100000 == 0:
            vals = [mole_fractions[g][i] for g in gases]
            print(f"{alt_m // 1000:>8} " + " ".join(f"{v:>12.4e}" for v in vals))
    print(f"\nWrote {len(output_altitudes_m)} altitude levels to {OUTPUT_CSV}")
