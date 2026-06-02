# Advection-Aware Temporal Attention: First Todo List

This is the first implementation slice for the temporal-context downscaling
plan. The goal is to de-risk the data harness and baseline ladder before adding
the windowed attention module.

## M0 - Harness and Baseline Parity

- [ ] Reproduce the single-frame CorrDiff regression baseline on Scandinavia.
- [ ] Confirm train/validation/evaluation splits are contiguous time blocks.
- [ ] Run the existing temporal input smoke test config end to end.
- [ ] Evaluate deterministic regression with extreme-focused metrics already in
      `helpers/metrics.py`: tail RMSE/CRPS, twCRPS, HRRE, MPPE, rank histogram,
      spread-skill diagnostics, and spectra/FSS where available.
- [ ] Record all runs with explicit temporal context labels passed as Hydra
      overrides instead of one-off config files: `t0`, `past_3h`, `sym_3h`,
      `past_6h`, `sym_6h`.

## M1 - Make-or-Break Baselines

- [ ] Train/evaluate parameter-matched channel-stacking baselines:
      `t0`, `[-3h,0]`, `[-3h,0,+3h]`, `[-6h,-3h,0]`,
      `[-6h,-3h,0,+3h,+6h]`.
- [ ] Add a lightweight temporal-conv conditioning baseline before STVD.
- [ ] Define the minimal STVD-style baseline needed for a fair comparison.
- [ ] Gate decision: if channel stacking or STVD saturates the tail metrics,
      re-center the project around advection verification and horizon analysis.

## M2 - Windowed Advection Module

- [ ] Implement single-head local temporal attention on the conditioning path.
- [ ] Use physical-hour metadata to reshape temporal channel stacks back to
      `(frame, channel, y, x)`.
- [ ] Return attention weights indexed by `(frame, dy, dx)`.
- [ ] Start with `[-3h,0,+3h]`, one coarse-resolution attention level, and a
      fixed physical window radius.

## M3 - Advection Verification

- [ ] Convert attention weights into expected displacement vectors.
- [ ] Compare attention displacement direction and magnitude against low-res
      winds, stratified by wind speed.
- [ ] Add perturbation tests: corrupt upwind attended region versus random
      region.
- [ ] Add negative controls: shuffled time order, wrong wind, shifted wind, and
      undersized attention window.

## Immediate Next Runs

1. `python -m pytest tests/test_temporal_inputs.py`
2. `python train.py --config-name=tr_reg_norway_sym3h_smoke`
3. Launch or schedule the Norway baseline ladder using explicit configs once data
   paths are available on the target machine:
   - `t0`: `python train.py --config-name=tr_reg_norway_t0`
   - `past_3h`: `python train.py --config-name=tr_reg_norway_past3h`
   - `sym_3h`: `python train.py --config-name=tr_reg_norway_sym3h`
   - `past_6h`: `python train.py --config-name=tr_reg_norway_past6h`
   - `sym_6h`: `python train.py --config-name=tr_reg_norway_sym6h`
