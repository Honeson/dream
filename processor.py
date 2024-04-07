# -*- coding: utf-8 -*-
import argparse
import os
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
#from utils import get_all_files_include_sub

def get_all_files_include_sub(path, file_type):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_type in file[-len(file_type):]:
                files.append(os.path.join(os.path.abspath(r), file))
    return files


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MESA",
                        help='the data to process')
    parser.add_argument('--parallel', type=int, default=0, help='parallel processing')
    parser.add_argument('--debug', type=int, default=1, help='debug mode')
    return parser.parse_args(argv)


def extract_dataset_part(file, pos=-1):
    return str(file.split(os.path.sep)[-1].split('.')[0].split('-')[pos])


def extract_PSG_annotation(profusion_file):
    try:
        from xml.etree import ElementTree as ET
        with open(profusion_file, "r") as f:
            contents = f.read()
            root = ET.fromstring(contents)
            sleep_stages = root.findall('.//SleepStage')
            sleep_stage_values = [elem.text for elem in sleep_stages]
        return sleep_stage_values
    except Exception as e:
        print(e)
        return None


# def downsampling_signal(time_val, signal_val, target_sampling_rate=35):
#     total_duration = time_val.max()
#     num_samples = len(time_val)
#     num_new_samples = int(total_duration * target_sampling_rate)
#     new_time_points = np.linspace(0, total_duration, num_new_samples, endpoint=False)
#     interpolation_function = interp1d(time_val, signal_val, kind='linear')
#     downsampled_values = interpolation_function(new_time_points)

#     print("Length of time index:", num_samples)
#     print("Length of downsampled values:", len(downsampled_values))
#     return downsampled_values

def downsampling_signal(time_val, signal_val, num_samples_desired):
    total_duration = time_val.max()
    new_time_points = np.linspace(0, total_duration, num_samples_desired, endpoint=False)
    interpolation_function = interp1d(time_val, signal_val, kind='linear')
    downsampled_values = interpolation_function(new_time_points)

    print("Length of time index:", len(time_val))
    print("Length of downsampled values:", len(downsampled_values))
    return downsampled_values




def extractChannel(edf_file, annotation_files, output_dir, target_hz=35):
    mesaid = extract_dataset_part(edf_file, -1)
    annotation_file = [prof for prof in annotation_files if mesaid in prof][0]
    raw = mne.io.read_raw_edf(edf_file)
    pleth_df = raw.to_data_frame(picks='Pleth')
    pleth_df = pleth_df.rename(columns={'Pleth': 'Pleth'})
    pleth_sampling_rate = 1000  # Assuming Pleth signal is sampled at 1000 Hz

    print("Length of time index:", len(pleth_df.index))
    print("Length of Pleth values:", len(pleth_df['Pleth']))
    num_sample_desired = len(pleth_df.index)

    pleth_df['Pleth'] = downsampling_signal(pleth_df.index.values, pleth_df['Pleth'].values, num_sample_desired)

    # Extract PSG annotation from profusion XML file
    stages = extract_PSG_annotation(annotation_file)
    if stages is None:
        return None, f"{mesaid}$The stages are None."
    if len(stages) > 1920:
        return None, f"{mesaid}$PSG recording was longer than 16 hours."

    # Save the data to an HDF5 file
    signal_output_dir = os.path.join(output_dir, "psg_signals")
    Path(signal_output_dir).mkdir(parents=True, exist_ok=True)
    h5_file = os.path.join(signal_output_dir, mesaid + ".h5")
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('signals', data=pleth_df.values)
        f.attrs['columns'] = ['Pleth']

    # Save the stages to a CSV file
    label_output_dir = os.path.join(output_dir, "psg_labels")
    Path(label_output_dir).mkdir(parents=True, exist_ok=True)
    label_file = os.path.join(label_output_dir, mesaid + ".csv")
    pd.DataFrame(stages).to_csv(label_file, index=False)

    return f"{mesaid}$PSG recording length is {len(stages)}"


def main(args):
    root = 'data'  # Specify your data root directory here
    dataset = args.dataset
    output_dir = os.path.join('data/results', dataset)  # Specify your output directory here
    all_edfs = get_all_files_include_sub(os.path.join(root, "edfs"), '.edf')
    all_profs = get_all_files_include_sub(os.path.join(root, "annotations-events-profusion"), '.xml')

    if args.parallel == 0:
        results = [extractChannel(edf_file, all_profs, output_dir) for edf_file in all_edfs]
    else:
        if args.debug == 1:
            all_edfs = all_edfs[:2]
        results = Parallel(n_jobs=-1)(delayed(extractChannel)(edf_file, all_profs, output_dir) for edf_file in all_edfs)

    # Save the processing result to a CSV file
    processing_result_file = os.path.join(output_dir, "processing_result.csv")
    processing_result_df = pd.DataFrame(results)
    processing_result_df = processing_result_df[0].str.split('$', expand=True)
    processing_result_df.columns = ['mesaid', 'details']
    processing_result_df['length'] = processing_result_df['details'].str.extract(r'PSG recording length is (\d+)')
    processing_result_df.to_csv(processing_result_file, index=False)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
