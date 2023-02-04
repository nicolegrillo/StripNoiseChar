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


def load_sci_data(ds: DataStorage, data_type, pol_name, start_time):

    mjd_start = Time(start_time[0], format="iso").to_value(format="mjd")
    mjd_end = Time(start_time[1], format="iso").to_value(format="mjd")

    times, values = ds.load_sci((mjd_start, end), polarimeter=pol_name,
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
    sig_c = (params[0] / wn_level) ** one_alpha
    der_f = sig_c
    der_sigma = one_alpha * sig_c * params[2] / wn_level
    der_alpha = params[2] * sig_c * np.log(params[0] / wn_level) * one_alpha ** 2

    return np.sqrt(
        der_f ** 2 * cov[2, 2] + der_sigma ** 2 * cov[0, 0] + der_alpha ** 2 * cov[1, 1] + 2 * der_f * der_sigma * cov[
            2, 0] + 2 * der_f * der_alpha * cov[2, 1] + 2 * der_alpha * der_sigma * cov[0, 1])


def results_rounding(value, error):

    if np.isnan(error) or error > 100 or value > 3000 or np.isnan(value):  # nonsense error associated to an ADU measurement

        return 1

    result = round(value, int(np.ceil(-np.log10(error))) + 1)

    return result


def all_pol_names(board_name): 

    B = ["B0","B1", "B2", "B3", "B4", "B5", "B6"]
    G = ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]
    I = ["I0", "I1", "I2", "I3", "I4", "I5", "I6"]
    O = ["O0", "O1", "O2", "O3", "O4", "O5", "O6"]
    R = ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]
    V = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
    W = ["W1", "W2", "W3", "W4", "W5", "W6"]
    Y = ["Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6"]

    if board_name == "B":
        return B

    if board_name == "G":
        return G

    if board_name == "I":
        return I

    if board_name == "O":
        return O

    if board_name == "R":
        return R

    if board_name == "V":
        return V

    if board_name == "Y":
        return Y

    return W  # in case of no valid choice, return the W elements


def noise_characterisation(ds: DataStorage, output_dir: Path, start_time, data_type, board_name):

    result = []
    board_pol_names = all_pol_names(board_name)
    boardname_xtick = 1

    # preparing to set the size of plots:

    # plt.figure(figsize=(12, 7))

    for pol_name in board_pol_names:

        channel_xtick = 0  # x tick for plot

        for channel in ("Q1", "Q2", "U1", "U2"):

            times, values = load_sci_data(ds, data_type, pol_name, start_time)

            var = {f"{pol_name}": {}}
            var[f"{pol_name}"]["pol_name"] = pol_name

            time_stream = second_demodulation(times, values, data_type, channel)

            # two options of cut in data depending on data_type:

            if data_type == "PWR":

                freq, fft = fourier_data_trasform(time_stream[0], time_stream[1], 50, 0)

            else:

                freq, fft = fourier_data_trasform(time_stream[0], time_stream[1], 50, 0)

            # fitting section:

            params, cov = curve_fit(noise_interpolation, freq, fft, maxfev=10000)

            # return interpolated parameters:

            wn_delta = delta_method(time_stream[1])
            var[f"{pol_name}"][f"err_sigma_interpol_{channel}"] = results_rounding(np.sqrt(cov[0][0]), np.sqrt(cov[0][0]))
            var[f"{pol_name}"][f"err_alpha_interpol_{channel}"] = results_rounding(np.sqrt(cov[1][1]), np.sqrt(cov[1][1]))
            var[f"{pol_name}"][f"err_fknee_interpol_{channel}"] = results_rounding(np.sqrt(cov[2][2]), np.sqrt(cov[2][2]))
            wn_level_pwr = wn_delta ** 2 * 2 / 50
            var[f"{pol_name}"][f"sigma_fknee_delta_{channel}"] = results_rounding(error_propagation_corr(cov, params, wn_level_pwr), error_propagation_corr(cov, params, wn_level_pwr))
            var[f"{pol_name}"][f"{channel}_fknee_delta"] = results_rounding(params[2] / (wn_level_pwr / params[0])**(1 / params[1]), var[f"{pol_name}"][f"sigma_fknee_delta_{channel}"])
            var[f"{pol_name}"][f"{channel}_fknee_interp"] = results_rounding(params[2], var[f"{pol_name}"][f"err_fknee_interpol_{channel}"])
            var[f"{pol_name}"][f"{channel}_wn"] = np.sqrt(params[0] * 50) / np.sqrt(2)
            var[f"{pol_name}"][f"alpha{channel}_f"] = results_rounding(params[1], var[f"{pol_name}"][f"err_alpha_interpol_{channel}"])
            var[f"{pol_name}"][f"{channel}_wn_delta"] = results_rounding(wn_delta, var[f"{pol_name}"][f"sigma_fknee_delta_{channel}"])

            # plots section:

            plt.errorbar(channel_xtick + boardname_xtick, np.array(var[f"{pol_name}"][f"{channel}_fknee_delta"]), yerr=var[f"{pol_name}"][f"sigma_fknee_delta_{channel}"], fmt='.', markersize='3.5', ecolor='royalblue',capsize=4, elinewidth=1.4, c='royalblue')
            plt.errorbar(channel_xtick + boardname_xtick, np.array(var[f"{pol_name}"][f"{channel}_fknee_interp"]), yerr=var[f"{pol_name}"][f"err_fknee_interpol_{channel}"], fmt='.', markersize='3.5', ecolor='red',capsize=4, elinewidth=1.4, c='red')

            channel_xtick = channel_xtick + 0.1  # increment every time of 0.05 to see separated lines

        boardname_xtick = boardname_xtick + 1

    plt.xticks(np.arange(1, len(board_pol_names), 1.0), board_pol_names)  # prepare ticks for graphs to plot the values
    plt.legend(['f_knee delta', 'f_knee interpol'])

    plot_file_name = output_dir / f"f_knee_{data_type}_plots.png"
    plt.savefig(plot_file_name, bbox_inches="tight")
    var[f"{pol_name}"][f"{board_name}{data_type}_plots_file_name"] = str(plot_file_name)

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
    # tags = get_acquisition_tags(ds, (start_time, end_time), pol_name)
    times = [start_time, end_time]

    output_dir.mkdir(exist_ok=True, parents=True)

    board_name = input("Choose the board (B, G, I, O, R, V, W, Y):")

    cur_result = noise_characterisation(ds, output_dir, times, data_type, board_name)


if __name__ == "__main__":
    main()

