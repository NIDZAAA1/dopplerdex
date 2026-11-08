"""Atmospheric acoustics: speed of sound and ISO 9613-1 absorption."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

C_STANDARD = 343.0  # m/s at ~20 C; used only as a default
T0_K = 293.15  # ISO 9613-1 reference temperature
T01_K = 273.16  # triple point of water
P_REF_KPA = 101.325


def speed_of_sound(temp_c: ArrayLike, rel_humidity: ArrayLike = 0.0) -> NDArray[np.float64]:
    """Speed of sound in air (m/s).

    Uses c = 331.3 * sqrt(1 + T/273.15) with a small humidity correction
    (~0.6 m/s at 20 C, 50 % RH), adequate to ~0.1 % for our purposes.
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.asarray(rel_humidity, dtype=np.float64)
    c_dry = 331.3 * np.sqrt(1.0 + t / 273.15)
    # Humidity raises c slightly (lower molar mass); linearised Cramer-style term.
    h = _molar_water_pct(t + 273.15, rh, P_REF_KPA)
    return c_dry * (1.0 + 0.0016 * h)


def _molar_water_pct(temp_k: ArrayLike, rel_humidity: ArrayLike, pa_kpa: ArrayLike) -> NDArray[np.float64]:
    """Molar concentration of water vapour h (%) per ISO 9613-1 Annex B."""
    t = np.asarray(temp_k, dtype=np.float64)
    c = -6.8346 * (T01_K / t) ** 1.261 + 4.6151
    psat_ratio = 10.0**c
    return np.asarray(rel_humidity, dtype=np.float64) * psat_ratio / (np.asarray(pa_kpa) / P_REF_KPA)


def absorption_db_per_m(
    freq_hz: ArrayLike,
    temp_c: float = 20.0,
    rel_humidity: float = 70.0,
    pressure_kpa: float = P_REF_KPA,
) -> NDArray[np.float64]:
    """Pure-tone atmospheric absorption coefficient alpha (dB/m), ISO 9613-1 eq. (5)."""
    f = np.asarray(freq_hz, dtype=np.float64)
    t = temp_c + 273.15
    pr = pressure_kpa / P_REF_KPA
    h = _molar_water_pct(t, rel_humidity, pressure_kpa)
    fr_o = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pr * (t / T0_K) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((t / T0_K) ** (-1.0 / 3.0) - 1.0)))
    term_classical = 1.84e-11 * (1.0 / pr) * (t / T0_K) ** 0.5
    term_o = 0.01275 * np.exp(-2239.1 / t) / (fr_o + f * f / fr_o)
    term_n = 0.1068 * np.exp(-3352.0 / t) / (fr_n + f * f / fr_n)
    return 8.686 * f * f * (term_classical + (t / T0_K) ** -2.5 * (term_o + term_n))
