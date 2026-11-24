"""Train CN2S or N2S on synthetic DAS data or fine-tune on real DAS data. 
Checkpoints are written as plain state dicts to <ckpt-dir>/<name>_<epoch>.pt 
every 10 epochs, with <name> as in the paper, e.g.

    CN2S-synthetic4-complement-128x512-16x180    python scripts/train.py --model cn2s --stage synthetic
    CN2S-finetuned4-complement-128x512-16x180    python scripts/train.py --model cn2s --stage finetune
    CN2S-real4-complement-128x512-16x180         python scripts/train.py --model cn2s --stage real
    N2S-synthetic4-44x2048-4x2048                python scripts/train.py --model n2s --stage synthetic
    N2S-finetuned4-44x2048-4x2048                python scripts/train.py --model n2s --stage finetune

Fine-tuning loads <ckpt-dir>/<synthetic name>_200.pt unless --init is given.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use('Agg')
matplotlib.rcParams['image.interpolation_stage'] = 'data'
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data  # noqa: E402
import inference  # noqa: E402
import masks  # noqa: E402
import models  # noqa: E402

# test windows (sample index of the earthquake) in the sorted real_test files
TEST_INDICES = [6250, 6750, 6250, 5250, 7250, 5500]
TEST_ORDER = [2, 3, 5, 0, 1, 4]
FIRST_CHANNEL = 81


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['cn2s', 'n2s'], default='cn2s')
    p.add_argument('--stage', choices=['synthetic', 'finetune', 'real'], default='synthetic',
                   help='real: train on real data from a random initialization (Fig. A3)')
    p.add_argument('--dx', type=int, choices=[4, 20], default=4, help='channel spacing in m')
    p.add_argument('--target', choices=['complement', 'center'], default='complement', help='CN2S loss target')
    p.add_argument('--data-dir', type=Path, default=Path('data/preprocessed'))
    p.add_argument('--ckpt-dir', type=Path, default=Path('ckpt'))
    p.add_argument('--out-dir', type=Path, default=Path('runs'), help='logs and test plots')
    p.add_argument('--init', type=Path, default=None, help='checkpoint to fine-tune from')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--samples-per-epoch', type=int, default=10_000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden-ch', type=int, default=32, help='CN2S width')
    p.add_argument('--n-layers', type=int, default=4, help='CN2S depth')
    p.add_argument('--num-workers', type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--amp', action='store_true', help='bfloat16 autocast (off in the paper)')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--val-every', type=int, default=10, help='epochs between reconstructions of the synthetic validation section')
    p.add_argument('--plot-every', type=int, default=50, help='epochs between test plots, 0 disables them')
    p.add_argument('--no-train-bandpass', action='store_true',
                   help='fine-tune on the unfiltered real data like the original notebooks did (see README)')
    p.add_argument('--name', default=None, help='overrides the checkpoint name')
    return p.parse_args()


def patch_config(args, fs):
    if args.model == 'cn2s':
        nx, nx_width, nt = (128, 16, 512) if args.dx == 4 else (32, 4, 512)
        nt_width = int(np.round((nx_width - 1) / (60 * 10 / 36 / fs / args.dx)))
    else:
        nx, nx_width, nt = (44, 4, 2048) if args.dx == 4 else (11, 1, 2048)
        nt_width = 2048
    return nx, nx_width, nt, nt_width


def run_name(args, nx, nx_width, nt, nt_width):
    stage = {'synthetic': 'synthetic', 'finetune': 'finetuned', 'real': 'real'}[args.stage]
    if args.model == 'cn2s':
        return f'CN2S-{stage}{args.dx}-{args.target}-{nx:d}x{nt:d}-{nx_width:d}x{nt_width:d}'
    return f'N2S-{stage}{args.dx}-{nx:d}x{nt:d}-{nx_width:d}x{nt_width:d}'


def load_test_windows(data_dir, dx):
    """Six 2048 sample windows of the real test events, bandpass filtered and normalized per channel."""
    test_dir = data_dir / 'real_test'
    paths = sorted(test_dir / f for f in os.listdir(test_dir))
    windows = []
    for p, idx in zip(paths, TEST_INDICES):
        with h5py.File(p, 'r') as hf:
            x = hf['DAS'][FIRST_CHANNEL:, idx - 1024:idx + 1024]
        windows.append(x[::5] if dx == 20 else x)
    x = np.stack(windows)[TEST_ORDER]
    gutter = 100
    x = np.pad(x, ((0, 0), (0, 0), (gutter, gutter)), mode='constant', constant_values=0)
    x = data.bandpass(x, low=1.0, high=10.0, fs=50, gutter=gutter)
    x /= x.std(axis=-1, keepdims=True)
    return torch.from_numpy(x.copy())


def load_real(split_dir, dx, fs, n_chunks, bandpass=True):
    """Full real recordings, bandpass filtered from 1 to 10 Hz and normalized per channel, as float32."""
    paths = sorted(split_dir / f for f in os.listdir(split_dir))
    x = []
    for p in paths:
        with h5py.File(p, 'r') as hf:
            s = hf['DAS'][FIRST_CHANNEL:]
        x.append(s[::5] if dx == 20 else s)
    x = np.stack(x)
    gutter = 1000
    x = np.pad(x, ((0, 0), (0, 0), (gutter, gutter)), mode='constant', constant_values=0)
    if bandpass:
        # in chunks to bound the memory of filtfilt
        x = np.concatenate([data.bandpass(c, low=1.0, high=10.0, fs=fs, gutter=gutter) for c in np.array_split(x, n_chunks)], axis=0)
    x /= x.std(axis=-1, keepdims=True)
    return x.astype(np.float32)


def kms2slowness(kms):
    return 1 / (kms * 1000)


def kmh2slowness(kmh):
    return 1 / (kmh * 1000 / 3600)


def synthetic_validation_section(eq_test, traffic_inc_test, traffic_dec_test, dx, fs):
    """Fixed 500 channel synthetic section (earthquake at 3 km/s, cars at 80 and 100 km/h, SNR 0.5)."""
    np.random.seed(42)
    eq = data.generate_synthetic_das(eq_test[4], dx, fs, kms2slowness(3), nx=500)[:, 4300:7300]
    eq /= eq.std(dim=-1, keepdim=True)

    traffic = data.shift_traffic_rates(traffic_dec_test[0], dx, fs, -kmh2slowness(80))[:500, 100:3300]
    traffic += data.shift_traffic_rates(traffic_inc_test[2], dx, fs, kmh2slowness(100))[:500, 1900:5100]
    traffic = torch.from_numpy(data.bandpass(traffic, 1.0, 10.0, fs, 100).copy())
    traffic /= traffic.std(dim=-1, keepdim=True)

    amp = np.sqrt(0.5)
    sample = amp * eq + traffic
    scale = sample.std(dim=-1, keepdim=True)
    sample /= scale
    # clean target in the units of the normalized sample
    return sample, amp * eq / scale


def load_synthetic_sources(data_dir):
    def load(name):
        x = torch.from_numpy(np.load(data_dir / name))
        return x / x.std(dim=-1, keepdim=True)

    def stack(fmt, idx):
        x = torch.from_numpy(np.stack([np.load(data_dir / fmt.format(i)) for i in idx]))
        return x / x.std(dim=-1, keepdim=True)

    train = (load('SIS-rotated_train_50Hz.npy'),
             stack('traffic_train_50Hz_{:02d}_inc.npy', range(5)),
             stack('traffic_train_50Hz_{:02d}_dec.npy', range(5, 12)))
    test = (load('SIS-rotated_test_50Hz.npy'),
            stack('traffic_test_50Hz_{:02d}_inc.npy', range(4)),
            stack('traffic_test_50Hz_{:02d}_dec.npy', range(4, 8)))
    return train, test


def main():
    args = parse_args()
    fs = 50.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    nx, nx_width, nt, nt_width = patch_config(args, fs)
    name = args.name or run_name(args, nx, nx_width, nt, nt_width)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.out_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                  | dict(name=name, nx=nx, nt=nt, nx_width=nx_width, nt_width=nt_width), f, indent=2)
    print(f'{name} on {device}', flush=True)

    # data
    test_windows = load_test_windows(args.data_dir, args.dx) if args.plot_every > 0 else None
    valsection = None
    if args.stage == 'synthetic':
        (eq_train, inc_train, dec_train), (eq_test, inc_test, dec_test) = load_synthetic_sources(args.data_dir)
        if args.model == 'cn2s':
            kw = dict(nx=nx, nt=nt, eq_slowness=(1e-4, 5e-3), traffic_slowness=(3e-2, 6e-2), dx=args.dx, fs=fs)
            trainset = data.SyntheticTrafficDAS(eq_train, inc_train, dec_train, size=args.samples_per_epoch, **kw)
            testset = data.SyntheticTrafficDAS(eq_test, inc_test, dec_test, size=args.samples_per_epoch // 10, **kw)
        else:
            trainset = data.SyntheticNoiseDAS(eq_train, nx=nx, nt=nt, dx=args.dx, size=args.samples_per_epoch)
            testset = data.SyntheticNoiseDAS(eq_test, nx=nx, nt=nt, dx=args.dx, size=args.samples_per_epoch // 10)
    else:
        bp = not args.no_train_bandpass
        trainset = data.RealDAS(load_real(args.data_dir / 'real_train', args.dx, fs, 10, bp), nx=nx, nt=nt, size=args.samples_per_epoch)
        testset = data.RealDAS(load_real(args.data_dir / 'real_test', args.dx, fs, 5), nx=nx, nt=nt, size=args.samples_per_epoch // 10)
        if args.model == 'cn2s':
            _, (eq_test, inc_test, dec_test) = load_synthetic_sources(args.data_dir)
    if args.model == 'cn2s':
        valsection = synthetic_validation_section(eq_test, inc_test, dec_test, args.dx, fs)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                     persistent_workers=args.num_workers > 0)
    dloader = {'train': DataLoader(trainset, shuffle=True, drop_last=False, **loader_kw),
               'test': DataLoader(testset, shuffle=False, **loader_kw)}

    # model
    if args.model == 'cn2s':
        model = models.CN2SUNet(1, 1, hidden_ch=args.hidden_ch, n_layers=args.n_layers).to(device)
    else:
        model = models.N2SUNet(1, 1, 4).to(device)
    if args.stage == 'finetune' or args.init is not None:
        init = args.init or args.ckpt_dir / (run_name(argparse.Namespace(**(vars(args) | {'stage': 'synthetic'})), nx, nx_width, nt, nt_width) + '_200.pt')
        print(f'initializing from {init}', flush=True)
        model.load_state_dict(torch.load(init, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def loss_fn(x):
        if args.model == 'cn2s':
            bs = x.shape[0]
            center = torch.cat([torch.randint(0, nx, (bs, 1)), torch.randint(0, nt, (bs, 1))], dim=1)
            input_mask, center_mask = masks.xmask(x, center, nx_width=nx_width, dx=args.dx, fs=fs)
            target_mask = center_mask if args.target == 'center' else 1 - input_mask
            out = model(input_mask * x)
            return torch.mean(torch.sum((target_mask * (out - x)) ** 2, dim=(1, 2, 3)) / torch.sum(target_mask, dim=(1, 2, 3)))
        mask = masks.channelwise_mask(x, nx_width)
        out = model(x * mask)
        return torch.mean(torch.mean((1 - mask) * (out - x) ** 2, dim=-1))

    def to_input(batch):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return x.to(device, non_blocking=True).float()

    def reconstruct(x):
        if args.model == 'cn2s':
            return inference.xreconstruct(model, x.to(device), nx, nt, nx_width, batch_size=128, dx=args.dx, fs=fs)
        if args.dx == 4:
            return inference.xreconstruct(model, x.to(device), nx, nt, nx_width, batch_size=128, dx=args.dx, fs=fs, ende=True)
        return inference.channelwise_reconstruct(model, x.to(device), nx, nt, batch_size=128)

    log_path = run_dir / 'log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_section_mse', 'time_s'])

    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp)
    pbar = tqdm(range(args.epochs), mininterval=10)
    for epoch in pbar:
        t0 = time.time()
        model.train()
        train_loss = torch.zeros((), device=device)
        for batch in dloader['train']:
            x = to_input(batch)
            with autocast:
                loss = loss_fn(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # accumulate on the device to avoid a sync per step
            train_loss += loss.detach()
        train_loss = train_loss.item() / len(dloader['train'])

        model.eval()
        with torch.no_grad(), autocast:
            val_loss = sum(loss_fn(to_input(b)).item() for b in dloader['test']) / len(dloader['test'])
            val_mse = float('nan')
            if valsection is not None and ((epoch + 1) % args.val_every == 0 or epoch == 0):
                rec = reconstruct(valsection[0]).float().cpu()
                val_mse = torch.mean((rec - valsection[1]) ** 2).item()

        dt = time.time() - t0
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_mse:.6f}', f'{dt:.1f}'])
        print(f'epoch {epoch + 1:3d}  train {train_loss:.6f}  val {val_loss:.6f}  section mse {val_mse:.6f}  {dt:.0f}s', flush=True)

        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), args.ckpt_dir / f'{name}_{epoch + 1:03d}.pt')

        if test_windows is not None and (epoch + 1) % args.plot_every == 0:
            fig, axs = plt.subplots(len(test_windows), 2, sharex=True, figsize=(8, 12))
            with torch.no_grad(), autocast:
                for i, w in enumerate(test_windows):
                    out = reconstruct(w).float().cpu()
                    kw = dict(origin='lower', interpolation='antialiased', cmap='seismic', aspect='auto', vmin=-1, vmax=1)
                    axs[i, 0].imshow(w.numpy(), **kw)
                    axs[i, 1].imshow(out.numpy(), **kw)
            fig.tight_layout()
            fig.savefig(run_dir / f'test_{epoch + 1:03d}.png', dpi=100)
            plt.close(fig)


if __name__ == '__main__':
    main()
