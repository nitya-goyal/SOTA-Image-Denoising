# Two-person work plan and supervisor update

## Roles

| Work | Repository owner | Friend with the GPU PC |
| --- | --- | --- |
| Source and configs | Review changes and maintain revision/experiment records | Work on feature branches, push source/config fixes and request review |
| Dataset | Agree protocol and inspect scene splits together | Store existing SIDD locally, validate pairs, retain manifests and metadata |
| Experiments | Maintain experiment register and check fair comparisons | Verify the Windows 11 RTX 4050 installation/free VRAM; run checks, training, evaluation and backups |
| Papers and understanding | Lead literature notes and proposed hypotheses | Study implementations and connect observed behavior to the papers |
| Analysis and reporting | Combine results, figures and presentation | Supply complete run evidence and help interpret successes/failures |

Both members should understand the pipeline and be able to explain their work.
Do not treat the GPU machine as the only research contribution. The owner need
not train locally; timing comparisons should use the same GPU PC.

## First working session

1. Accept collaborator access, clone current code and install a CUDA-enabled
   PyTorch build on the Windows 11 RTX 4050 laptop.
2. Save `check_environment.py --require-gpu --output runs/hardware.json` output.
   Verify actual free VRAM and device execution rather than assuming them.
3. Point preparation at the existing SIDD Medium sRGB folder. Agree on internal
   scene holdout versus official validation and save the split metadata.
4. Run the real-data memory check for both standard models. Keep batch/patch/
   precision consistent; record any necessary change in new configs.
5. Run each model to 100 steps, sequentially. Inspect loss/history/checkpoints.
6. Evaluate those pilots on validation, retain noisy/prediction/GT images and
   label all metrics **preliminary, 100-step pilot**. Do not inspect the internal
   test set to select training settings.
7. Record the commit, environment, config, protocol and actual elapsed times.
   Estimate remaining runtime only after observing the real pilot.

## Next milestones (progress-based, not promised dates)

| Milestone | Deliverable |
| --- | --- |
| Environment/data ready | Hardware JSON, verified pair counts, fixed split metadata |
| Pilot completed | Two 100-step histories, checkpoints and preliminary validation outputs |
| Reference runs completed | Matching planned budgets, best checkpoints, complete histories |
| Final reference evaluation | Untouched internal test scores OR explicitly labeled official validation scores, visual crops and timings |
| Research direction selected | Specific failure case, relevant papers, testable improvement hypothesis |
| Improvement established | Separate modified method, matched comparisons, ablations and repeated seeds |

The initial 20,000-step common recipe is a starting controlled experiment.
If the goal is published-number reproduction, first audit and match upstream
training/inference settings rather than assuming more GPU time alone will match
published scores. Keep initial reference runs unchanged while trying improvements.

## What to show sir now

A short 5–7 minute progress discussion can use the following sequence. A working
pipeline and an honest plan are sufficient for an implementation-stage update;
results should be included only if the real experiment was actually run.

1. **Problem:** remove real camera noise while preserving useful detail.
   Working title: *Real-World Image Denoising*. State the two-person work split.
2. **Dataset:** SIDD sRGB noisy/GT examples from the actual dataset. Explain
   pairing, normalization and scene-disjoint splitting; show split pair counts.
3. **Method understanding:** explain U-Net encoder/decoder + skip connections
   and NAFNet's SimpleGate and channel attention. Cite the original papers.
4. **Implementation:** show the clean repo, shared train/eval pipeline, saved
   configs, checkpoint/resume, and actual local GPU report if available.
5. **Evidence:** software checks are already recorded in VERIFICATION.md. If
   you have a real pilot, show its loss history and fixed validation crops with
   steps/config clearly labeled. Otherwise say training is pending.
6. **Next work:** complete reference training; inspect residual noise, texture
   loss and runtime; choose one improvement with evidence, then test ablations.

Suggested wording (adapt to the actual status):

> Our project is real-world image denoising. We have implemented a paired SIDD
> pipeline with U-Net and NAFNet as reference methods. The current stage is to
> understand and validate those methods on our GPU machine. We will use measured
> failure cases and recent literature to develop and evaluate an improvement.

Do not say the project is complete, that an improvement already exists, or that
published/current SOTA scores have been reproduced. A test passing means software
behavior was checked, not that a model learned to denoise real images.

## Evidence to keep and share

Maintain an experiment register with: run name, commit, model/config, protocol,
seed, batch/patch size, precision, planned/completed steps, hardware and status.
Save a copy of configs/manifests, hardware and environment reports, history.csv,
run_info.json, best.pt/last.pt, evaluation JSON/CSV and representative images.
Back up completed checkpoints to another disk or shared storage after stopping
the run. A checkpoint being actively written is not a safe backup snapshot.

Git-ignored data/results are not uploaded with source. Share a small evidence
bundle (metadata, histories, metrics and fixed crops) for reviews, with weights
separately when needed. Agree data/checkpoint sharing locations between you.


## Contributor changes

Use an individual feature/experiment branch and a pull request for code changes.
The owner and contributor review each other's work. Pull before starting a new
run, not during one. Keep the commit, config and split fixed for continuation.
If a run exposes a bug, stop and record its status before reviewing a fix; do not
silently combine measurements from different implementations.
