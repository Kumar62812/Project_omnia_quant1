"""Project Omnia entropy-driven mixed-precision PTQ entry point.

Important protocol rule: tau is selected on a dedicated selection set. The final
held-out test/validation set is evaluated only after tau and the bit strategy are
frozen. Hardware values are labeled analytical unless actual measurements are
supplied.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import models as customized_models
from lib.utils.data_utils import get_calibration_loader, get_dataset
from lib.utils.entropy_utils import apply_strategy, estimate_compression, run_calibration_and_get_entropy, sweep_tau
from lib.utils.model_registry import build_model_registry
from lib.utils.quantize_utils import QConv2d, QLinear, calibrate
from lib.utils.utils import AverageMeter, Logger, accuracy

try:
    from lib.simulator.lookup_tables import LatencyEstimator
    HAS_ANALYTICAL_HW = True
except Exception:
    LatencyEstimator = None
    HAS_ANALYTICAL_HW = False

model_names, models = build_model_registry(customized_models)


def get_args():
    parser = argparse.ArgumentParser(description="Project Omnia: entropy-driven mixed-precision PTQ")
    parser.add_argument('--dataset', default='imagenet')
    parser.add_argument('--dataset_root', default='data/imagenet')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--calib_size', type=int, default=100)
    parser.add_argument('--calib_batch', type=int, default=25)
    parser.add_argument('--selection_subset', type=int, default=5000)
    parser.add_argument('--eval_batch', type=int, default=256)
    parser.add_argument('--arch', '-a', default='qmobilenetv2', choices=model_names)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--num_bins', type=int, default=256)
    parser.add_argument('--clip_percentile', type=float, default=99.9)
    parser.add_argument('--entropy_mode', choices=['per_tensor', 'per_channel'], default='per_tensor')
    parser.add_argument('--tau_steps', type=int, default=25)
    parser.add_argument('--tau_min', type=float, default=None)
    parser.add_argument('--tau_max', type=float, default=None)
    parser.add_argument('--w_bit_low', type=int, default=4)
    parser.add_argument('--a_bit_low', type=int, default=4)
    parser.add_argument('--w_bit_high', type=int, default=8)
    parser.add_argument('--a_bit_high', type=int, default=8)
    parser.add_argument('--max_acc_drop', type=float, default=0.8)
    parser.add_argument('--min_low_bit_frac', type=float, default=0.6)
    parser.add_argument('--gpu_id', default='0')
    parser.add_argument('--output', default='save/omnia')
    parser.add_argument('--seed', type=int, default=234)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build_quantizable_index(model):
    return [i for i, module in enumerate(model.modules()) if type(module) in (QConv2d, QLinear)]


def load_fp32_weights(model, checkpoint_path, min_load_fraction=0.95):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    state = {key.replace('module.', '', 1): value for key, value in state.items()}
    target = model.state_dict()
    loaded = 0
    skipped = []
    for key, value in state.items():
        if key in target and target[key].shape == value.shape:
            target[key].copy_(value)
            loaded += 1
        else:
            skipped.append(key)
    if not state:
        raise RuntimeError(f'Checkpoint {checkpoint_path!r} contains no state_dict tensors')
    fraction = loaded / len(state)
    if fraction < min_load_fraction:
        raise RuntimeError(f'Only {loaded}/{len(state)} checkpoint tensors matched ({fraction:.3f}); refusing to evaluate a partial load')
    model.load_state_dict(target, strict=False)
    return model, {'loaded_tensors': loaded, 'checkpoint_tensors': len(state), 'skipped_tensors': len(skipped), 'match_fraction': fraction}


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    top1 = AverageMeter()
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output = model(images)
            top1_acc, _ = accuracy(output.data, targets.data, topk=(1, 5))
            top1.update(float(top1_acc.item()), images.size(0))
    return float(top1.avg)


def make_loader_subset(loader, limit):
    # Validation is already deterministic. We use the first `limit` samples
    # only for tau selection; final evaluation remains full-set and separate.
    if limit is None:
        return loader, None
    from torch.utils.data import DataLoader, Subset
    limit = min(int(limit), len(loader.dataset))
    subset = Subset(loader.dataset, range(limit))
    return DataLoader(subset, batch_size=loader.batch_size, shuffle=False,
                      num_workers=loader.num_workers, pin_memory=True), limit


def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    logger = Logger(str(out / 'log.txt'), title='project-omnia-' + args.arch)
    logger.set_names(['tau', 'selection_acc', 'acc_drop', 'low_bit_frac', 'feasible'])

    train_loader, full_eval_loader, n_class = get_dataset(
        dataset_name=args.dataset, batch_size=args.eval_batch,
        n_worker=args.workers, data_root=args.dataset_root)
    selection_loader, selection_count = make_loader_subset(full_eval_loader, args.selection_subset)
    calib_loader = get_calibration_loader(
        dataset_name=args.dataset, data_root=args.dataset_root,
        calib_size=args.calib_size, batch_size=args.calib_batch,
        n_worker=args.workers, seed=args.seed)

    model = models[args.arch](pretrained=False, num_classes=n_class).to(device)
    model, checkpoint_info = load_fp32_weights(model, args.resume)
    quantizable_idx = build_quantizable_index(model)
    if not quantizable_idx:
        raise RuntimeError('No QConv2d/QLinear layers found in the selected architecture')

    fp32_selection_acc = evaluate(model, selection_loader, device)
    fp32_final_acc = evaluate(model, full_eval_loader, device)

    entropy_start = time.perf_counter()
    entropy_dict = run_calibration_and_get_entropy(
        model, calib_loader, quantizable_idx,
        num_bins=args.num_bins, use_cuda=(device.type == 'cuda'),
        entropy_mode=args.entropy_mode)
    entropy_time = time.perf_counter() - entropy_start
    entropies = list(entropy_dict.values())
    tau_min = min(entropies) if args.tau_min is None else args.tau_min
    tau_max = max(entropies) if args.tau_max is None else args.tau_max
    if args.tau_steps < 2 or tau_max < tau_min:
        raise ValueError('tau_steps must be >=2 and tau_max must be >= tau_min')
    tau_candidates = [tau_min + i * (tau_max - tau_min) / (args.tau_steps - 1) for i in range(args.tau_steps)]

    def calibrate_fn(current_model, strategy):
        apply_strategy(current_model, quantizable_idx, strategy)
        return calibrate(current_model, calib_loader)

    def selection_eval_fn(current_model):
        return evaluate(current_model, selection_loader, device)

    best_tau, sweep_results = sweep_tau(
        model, quantizable_idx, entropy_dict, tau_candidates,
        calibrate_fn=calibrate_fn, eval_fn=selection_eval_fn,
        fp32_acc=fp32_selection_acc,
        low_bits=(args.w_bit_low, args.a_bit_low),
        high_bits=(args.w_bit_high, args.a_bit_high),
        max_acc_drop=args.max_acc_drop,
        min_low_bit_frac=args.min_low_bit_frac)

    best_result = next(item for item in sweep_results if item['tau'] == best_tau)
    if not best_result['satisfies_constraints']:
        raise RuntimeError('No tau candidate satisfied both accuracy-drop and low-bit-fraction constraints; publication run is FAIL')

    # Freeze the selected strategy before touching the final evaluation set.
    best_strategy = best_result['strategy']
    calibrate_fn(model, best_strategy)
    final_acc = evaluate(model, full_eval_loader, device)
    final_drop = fp32_final_acc - final_acc
    compression = estimate_compression(model, quantizable_idx, best_strategy)

    hw_report = {'source': 'not_run'}
    if HAS_ANALYTICAL_HW:
        hw_report = {
            'source': 'analytical_model_requires_feature_shapes',
            'hardware_profile': 'fpga_edge_custom_hls_analytic',
            'status': 'not_reported_as_measured',
        }

    report = {
        'protocol_version': 'omnia-publication-v1',
        'arch': args.arch,
        'dataset': args.dataset,
        'seed': args.seed,
        'checkpoint': {'path': os.path.abspath(args.resume), 'sha256': file_sha256(args.resume), **checkpoint_info},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device)},
        'calibration': {'images': args.calib_size, 'batch': args.calib_batch, 'entropy_time_sec': entropy_time},
        'entropy': {'mode': args.entropy_mode, 'bins': args.num_bins, 'clip_percentile': args.clip_percentile, 'layer_values': entropy_dict},
        'selection': {'images': selection_count, 'fp32_top1': fp32_selection_acc, 'tau_candidates': tau_candidates, 'chosen_tau': best_tau, 'sweep': sweep_results},
        'final_test': {'fp32_top1': fp32_final_acc, 'quantized_top1': final_acc, 'top1_drop': final_drop},
        'strategy': {str(k): list(v) for k, v in best_strategy.items()},
        'low_bit_layer_fraction': best_result['low_bit_frac'],
        'compression': compression,
        'hardware': hw_report,
        'targets': {
            'max_top1_drop': args.max_acc_drop,
            'top1_target_met': final_drop <= args.max_acc_drop,
            'min_low_bit_fraction': args.min_low_bit_frac,
            'low_bit_target_met': best_result['low_bit_frac'] >= args.min_low_bit_frac,
            'theoretical_max_compression_vs_int8_for_INT4_INT8_only': 2.0,
            'calibration_time_target_sec': 180,
            'calibration_time_target_met': entropy_time <= 180,
        },
        'publication_note': 'No experimental result in this report should be described as measured hardware latency unless a real hardware measurement is supplied.'
    }

    with open(out / 'omnia_report.json', 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    with open(out / 'experiment_manifest.json', 'w', encoding='utf-8') as handle:
        json.dump({'protocol_version': 'omnia-publication-v1', 'args': vars(args), 'report_path': str(out / 'omnia_report.json')}, handle, indent=2)

    torch.save({'state_dict': model.state_dict(), 'strategy': best_strategy, 'tau': best_tau}, out / 'omnia_quantized.pth.tar')
    for result in sweep_results:
        logger.append([result['tau'], result['acc'], result['acc_drop'], result['low_bit_frac'], result['satisfies_constraints']])
    logger.close()

    print('\n===== Project Omnia Publication-Safe Report =====')
    print(f'FP32 final Top-1:      {fp32_final_acc:.4f}')
    print(f'Quantized final Top-1: {final_acc:.4f}')
    print(f'Top-1 drop:            {final_drop:.4f} pp')
    print(f'Chosen tau:            {best_tau:.6f}')
    print(f'INT4 layer fraction:   {best_result["low_bit_frac"]:.4f}')
    print(f'Compression vs INT8:   {compression["compression_vs_int8"]:.4f}x')
    print(f'Entropy calibration:   {entropy_time:.2f}s')
    print(f'Report:                {out / "omnia_report.json"}')


if __name__ == '__main__':
    main()
