"""Preprocess the SISSLE data for training.

Expects the folder structure of the ANU Data Commons record (see scripts/download_data.py):
    <data-dir>/seismic_data/*.mseed
    <data-dir>/distributed_acoustic_sensing_data/{index0185_0205, <event id>, ...}/*.h5
and the station coordinates in metadata/stations_2B_SISSLE.csv (exported from AusPass).

Writes to <data-dir>/preprocessed/:
    SIS-rotated_{train,test}_50Hz.npy           simulated strain rates from nodal station pairs
    traffic_{train,test}_50Hz_XX_{inc,dec}.npy  manually picked traffic signals
    real_{train,test}/<event id>.h5             real DAS recordings of earthquakes
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

# station metadata of the SISSLE nodal array, exported from AusPass (network 2B, 2023)
STATIONS_CSV = Path(__file__).resolve().parents[1] / 'metadata' / 'stations_2B_SISSLE.csv'

# manually selected traffic picks: (chunk index, first channel, sample, slowness in s/m)
TRAFFIC_TRAIN_PICKS = [(2, 190, 3200, -0.034), (3, 440, 2000, -0.035), (4, 870, 2000, -0.033), (5, 1230, 1500, -0.03),
                       (7, 2230, 2000, -0.039), (15, 3050, 7500, 0.045), (21, 1320, 7500, 0.048), (23, 750, 7000, 0.05),
                       (23, 1880, 7500, 0.057), (25, 1285, 7500, 0.051), (27, 740, 7500, 0.054), (29, 220, 7000, 0.052)]
TRAFFIC_TEST_PICKS = [(1, 625, 1500, -0.047), (2, 935, 1500, -0.04), (3, 1290, 1500, -0.037), (5, 2070, 1500, -0.048),
                      (5, 4000, 6500, 0.042), (10, 2440, 7000, 0.048), (12, 1720, 7000, 0.040), (13, 1270, 7500, 0.040)]

# earthquakes used as real DAS samples
# test events:
#   2023p202703  M4.7, heavy traffic, distance 175 km
#   2023p215452  M2.7, no traffic, distance 76 km
#   2023p273854  M1.9, minimal traffic, distance 10 km
#   2023p181489  M3.3, some traffic, distance 92 km
#   2023p197461  M3.6, some traffic, distance 159 km
#   2023p215753  M3.1, some traffic, distance 191 km
REAL_TRAIN_EVENTS = ['2023p214660', '2023p300579', '2023p303853', '2023p218553', '2023p326201', '2023p313442',
                     '2023p265881', '2023p171245', '2023p291534', '2023p293493', '2023p278798', '2023p321451',
                     '2023p166738', '2023p307984', '2023p273765', '2023p209788', '2023p326200', '2023p299899',
                     '2023p212693', '2023p303138', '2023p252684', '2023p167136', '2023p202039', '2023p235224',
                     '2023p272380', '2023p172434', '2023p152354', '2023p316959', '2023p178129']
REAL_TEST_EVENTS = ['2023p202703', '2023p215452', '2023p273854', '2023p181489', '2023p197461', '2023p215753']


def simulate_strain_rates(seismic_dir, out_dir):
    import obspy
    import pyproj
    from obspy.geodetics import base
    from obspy.signal.rotate import rotate_ne_rt

    # (code, latitude, longitude) in the order of the traces in the mseed files
    with open(STATIONS_CSV) as f:
        stations = [(r['station'], float(r['latitude']), float(r['longitude'])) for r in csv.DictReader(f)]
    locations = {code: (lat, lon) for code, lat, lon in stations}
    codes = list(locations.keys())

    # all station pairs closer than 150 m
    station_pairs = []
    for i, (code_i, lat_i, lon_i) in enumerate(stations):
        if code_i in ['SIS01', 'SIS05']:
            continue
        for code_j, lat_j, lon_j in stations[i + 1:]:
            d = base.gps2dist_azimuth(lat_i, lon_i, lat_j, lon_j)[0]
            if d <= 150:
                station_pairs.append((code_i, code_j))
    print(f'{len(station_pairs)} station pairs')

    files = sorted(glob.glob(os.path.join(seismic_dir, '*.mseed')))
    n_train = int(len(files) * 0.8)
    geodesic = pyproj.Geod(ellps='WGS84')

    # the traces of each station are stored in the order E, N, Z
    def strain_rates(file_list):
        out = []
        for f in tqdm(file_list):
            st = obspy.read(f)
            for station1, station2 in station_pairs:
                lat1, lon1 = locations[station1]
                idx1 = codes.index(station1)
                assert st[3 * idx1].stats.station == station1
                north1 = st[3 * idx1 + 1].data[:9000]
                east1 = st[3 * idx1].data[:9000]

                lat2, lon2 = locations[station2]
                idx2 = codes.index(station2)
                assert st[3 * idx2].stats.station == station2
                north2 = st[3 * idx2 + 1].data[:9000]
                east2 = st[3 * idx2].data[:9000]

                _, back_azimuth, distance_in_m = geodesic.inv(lon1, lat1, lon2, lat2)

                # rotate to the radial component and take the finite difference
                radial1, _ = rotate_ne_rt(north1, east1, back_azimuth % 360)
                radial2, _ = rotate_ne_rt(north2, east2, back_azimuth % 360)
                out.append((radial1 - radial2) / distance_in_m)
        return np.stack(out)

    np.save(out_dir / 'SIS-rotated_train_50Hz.npy', strain_rates(files[:n_train]))
    np.save(out_dir / 'SIS-rotated_test_50Hz.npy', strain_rates(files[n_train:]))


def extract_traffic(das_dir, out_dir, dx=4, fs=50):
    dir1 = das_dir / 'index0185_0205'
    files1 = [f'{m // 60:02d}{m % 60:02d}01' for m in range(7 * 60 + 55, 8 * 60 + 30)]
    prefix = 'south30_50Hz_UTC_20230401_'
    suffix = '.001.h5'

    def get_chunk(idx):
        return np.hstack([h5py.File(dir1 / (prefix + files1[idx + k] + suffix), 'r')['DAS'][:].T[81:4978] for k in range(3)])

    for split, picks in [('train', TRAFFIC_TRAIN_PICKS), ('test', TRAFFIC_TEST_PICKS)]:
        for n, (chunk_idx, i, j, slowness) in enumerate(tqdm(picks, desc=f'traffic {split}')):
            shift = dx * fs * slowness
            data = get_chunk(chunk_idx)
            strain_rates = []
            for x in range(512):
                jj = j - int(x * shift)
                strain_rates.append(data[i + x, jj - 30 * fs:jj + 30 * fs])
            strain_rates = np.stack(strain_rates)
            direction = 'inc' if slowness < 0 else 'dec'
            np.save(out_dir / 'traffic_{}_50Hz_{:02d}_{}.npy'.format(split, n, direction), strain_rates)


def extract_real(das_dir, out_dir):
    for split, events in [('train', REAL_TRAIN_EVENTS), ('test', REAL_TEST_EVENTS)]:
        os.makedirs(out_dir / f'real_{split}', exist_ok=True)
        for event in tqdm(events, desc=f'real {split}'):
            files = sorted(os.listdir(das_dir / event))
            data = np.hstack([h5py.File(das_dir / event / f, 'r')['DAS'][:].T for f in files])
            with h5py.File(out_dir / f'real_{split}' / f'{event}.h5', 'w') as hf:
                hf.create_dataset('DAS', data=data)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=None, help='default: <data-dir>/preprocessed')
    p.add_argument('--steps', nargs='+', default=['strain', 'traffic', 'real'], choices=['strain', 'traffic', 'real'])
    args = p.parse_args()

    out = args.out or args.data_dir / 'preprocessed'
    out.mkdir(parents=True, exist_ok=True)
    das_dir = args.data_dir / 'distributed_acoustic_sensing_data'

    if 'strain' in args.steps:
        simulate_strain_rates(args.data_dir / 'seismic_data', out)
    if 'traffic' in args.steps:
        extract_traffic(das_dir, out)
    if 'real' in args.steps:
        extract_real(das_dir, out)


if __name__ == '__main__':
    sys.exit(main())
