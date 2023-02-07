#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import logging
from datetime import datetime
from pathlib import Path
import json
from json import JSONEncoder
import sys
from astropy.time import Time

import numpy as np
from typing import List, Any, Dict, Tuple

import scipy
from scipy import signal
from scipy.optimize import curve_fit
from rich.logging import RichHandler
from mako.template import Template
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from striptease import Tag, DataStorage, polarimeter_iterator


def get_acquisition_tags(ds: DataStorage, date_range: Tuple[str, str], pol_name: str) -> Tag:
    tags = ds.get_tags(date_range)
    return [x for x in tags if "STABLE_ACQUISITION" in x.name and x.name.endswith(pol_name)][0]


def load_sci_data(ds: DataStorage, data_type, pol_name, tag: Tag):

    times, values = ds.load_sci((tag.mjd_start + 500 / 86400, tag.mjd_start + 7200 / 86400), polarimeter=pol_name,
                                data_type=f"{data_type}")

    times = Time(times, format="mjd").to_value(format="unix")  # time values are converted for the calculations

    return times, values


def second_demodulation(times, values, data_type, channel):

    times_values = (times[1:] + times[:-1]) / 2
    times_even = times_values[::2]

    if data_type == "PWR":

        pwr_values = values[f"{data_type}{channel}"][1:] + values[f"{data_type}{channel}"][:-1]
        pwr_values_even = pwr_values[::2]

        return times_even, pwr_values_even

    else:

        dem_values = values[f"{data_type}{channel}"][1:] - values[f"{data_type}{channel}"][:-1]
        dem_values_even = dem_values[::2]

        return times_even, dem_values_even


def noise_interpolation(f, sigma, alpha, f_knee):
    return sigma * ((f_knee / f)**alpha + 1)


def delta_method(values):

    delta_values = values[1:] - values[:-1]
    delta_values_even = delta_values[::2]
    mean_abs_dev = np.median(np.abs(np.median(delta_values_even) - delta_values_even))

    return mean_abs_dev / (0.67449 * np.sqrt(2))


def fourier_data_trasform(times, values, f_max, f_min):

    freq, fft = scipy.signal.welch(values, 1 / np.median(times[1:] - times[:-1]),
                                   nperseg=400 * 50)

    mask = (freq > f_min) & (freq <= f_max)
    freq = freq[mask]
    fft = fft[mask]

    return freq, fft


def error_propagation_corr(cov, params, wn_level):

    one_alpha = 1 / params[1]
    sig_c = params[0] ** one_alpha / wn_level ** one_alpha
    der_f = sig_c
    der_sigma = one_alpha * sig_c * params[2] / wn_level
    der_alpha = params[2] * sig_c * np.log(params[0] / wn_level) * one_alpha ** 2

    return np.sqrt(
        der_f ** 2 * cov[2, 2] + der_sigma ** 2 * cov[0, 0] + der_alpha ** 2 * cov[1, 1] + 2 * der_f * der_sigma * cov[
            2, 0] + 2 * der_f * der_alpha * cov[2, 1] + 2 * der_alpha * der_sigma * cov[0, 1])


def noise_characterisation(ds: DataStorage, output_dir: Path, tag: Tag, data_type):

    result = []

    for board_name, pol_idx, pol_name in polarimeter_iterator():

        times, values = load_sci_data(ds, data_type, pol_name, tag)

        var = {"pol": {}}
        var["pol"]["pol_name"] = pol_name

        for channel in ("Q1", "Q2", "U1", "U2"):

            time_stream = second_demodulation(times, values, data_type, channel)

            # two options of cut in data depending on data_type:

            if data_type == "PWR":

                freq, fft = fourier_data_trasform(time_stream[0], time_stream[1], 10, 0)

            else:

                freq, fft = fourier_data_trasform(time_stream[0], time_stream[1], 1, 0)

            # fitting section:

            params, cov = curve_fit(noise_interpolation, freq, fft, maxfev=10000)

            # return interpolated parameters:

            var["pol"][f"{channel}_wn_delta"] = delta_method(time_stream[1])
            wn_level_pwr = var["pol"][f"{channel}_wn_delta"] ** 2 * 2 / 50
            var["pol"][f"{channel}_wn_interpol"] = np.sqrt(params[0] * 50) / np.sqrt(2)
            var["pol"][f"{channel}_fknee_delta"] = params[2] / (wn_level_pwr / params[0]) ** (1 / params[1])
            var["pol"][f"{channel}_fknee_interp"] = params[2]
            var["pol"][f"sigma_fknee_interpol_{channel}"] = np.sqrt(cov[2, 2])
            var["pol"][f"sigma_fknee_delta_{channel}"] = error_propagation_corr(cov, params, wn_level_pwr)
            var["pol"][f"alpha{channel}_f"] = params[1]
            var["pol"][f"sigma_alpha_interpol_{channel}"] = np.sqrt(cov[1, 1])
            var["pol"][f"sigma_sigma_interpol_{channel}"] = np.sqrt(cov[0, 0])

        result.append(var)

    return result


def main():
    if len(sys.argv) != 7:
        print("Wrong number of parameters")
        sys.exit(1)

    logging.basicConfig(level="INFO", format='%(message)s', datefmt="[%X]", handlers=[RichHandler()])

    storage_path = sys.argv[1]
    start_time = sys.argv[2]
    end_time = sys.argv[3]
    pol_name = sys.argv[4]
    output_dir = Path(sys.argv[5])
    data_type = sys.argv[6]

    logging.info(f"Going to load data from directory'{storage_path}'")
    ds = DataStorage(storage_path)
    tags = get_acquisition_tags(ds, (start_time, end_time), pol_name)

    output_dir.mkdir(exist_ok=True, parents=True)

    cur_result = noise_characterisation(ds, output_dir, tags, data_type)

    with open(output_dir / "results.json", "wt") as outf:
        json.dump(cur_result, outf)


if __name__ == "__main__":
    main()
