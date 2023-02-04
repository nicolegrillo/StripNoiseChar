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

    times, values = ds.load_sci((mjd_start, mjd_end), polarimeter=pol_name,
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


def cross_correlation(ds: DataStorage, output_dir: Path, start_time, data_type):
    result = []

    for board_name, pol_idx, pol_name in polarimeter_iterator():  # this is set for al polarimeters that must be on to work

        times, values = load_sci_data(ds, data_type, pol_name, start_time)

        var = {"polarimeter": {}}
        var["polarimeter"]["pol_name"] = pol_name

        for channel in ("Q1", "Q2", "U1", "U2"):

            var["polarimeter"][f"{pol_name}_{channel}_data"] = second_demodulation(times, values, data_type, channel)

        fs = 1 / np.median(var["polarimeter"][f"{pol_name}_{channel}_data"][0][1:] - var["polarimeter"][f"{pol_name}_{channel}_data"][0][:-1])

        f_q, power_q = signal.csd(var["polarimeter"][f"{pol_name}_Q1_data"][1], var["polarimeter"][f"{pol_name}_Q2_data"][1], fs, nperseg=400 * 50)
        f_u, power_u = signal.csd(var["polarimeter"][f"{pol_name}_U1_data"][1], var["polarimeter"][f"{pol_name}_U2_data"][1], fs, nperseg=400 * 50)

        # plt.suptitle(f'{pol_name} cross-correlation plot')
        plt.plot(f_q[f_q > 0], np.abs(power_q[f_q > 0]), label="Q1, Q2 corr", c="dodgerblue")
        plt.plot(f_u[f_u > 0], np.abs(power_u[f_u > 0]), label="U1, U2 corr", c="cyan")
        plt.xlabel('frequency [Hz]')
        plt.ylabel('CSD [ADU**2/Hz]')
        plt.legend()

        plot_file_name = output_dir / f"{pol_name}_{data_type}_corr_plots.png"
        plt.savefig(plot_file_name, bbox_inches="tight")
        var["polarimeter"][f"{data_type}{channel}_plots_file_name"] = str(plot_file_name)

        plt.clf()

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
    times = [start_time, end_time]

    output_dir.mkdir(exist_ok=True, parents=True)

    cur_result = cross_correlation(ds, output_dir, times, data_type)


if __name__ == "__main__":
    main()
