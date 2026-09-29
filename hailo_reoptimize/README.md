# Rebuilding yolox-m + efficientnet-lite3 for Hailo-8, from checkpoint to HEF

**Flow:** PyTorch checkpoint → ONNX → parsed HAR → quantized HAR → HEF → verify on the Hailo host.

**Two problems this rebuild fixes:**
1. **Detector color bug.** The old detector alls has `input_conversion(bgr_to_rgb)`, but YOLOX
   trains on OpenCV BGR, so the chip fed the model RGB. Measured on the Hailo (conf 0.001):
   AP 0.571 with that bug, 0.662 when the host compensates. The pipeline now compensates by
   default (`--detector-bgr-to-rgb`). The rebuilt detector drops the line, so it doesn't need that.
2. **Classifier quantization loss.** 82.7 % top-1 on `test/` crops on the Hailo, against about
   97 % in PyTorch. The first build was calibrated on only about 100 crops.

**Targets.** Real-world settings (conf 0.3), on the zeus-cropped test set:

| | PyTorch | Hailo, current HEFs (color fix on) | goal |
|---|---|---|---|
| yolox-m only, AP@[.5:.95] | 0.682 | 0.659 | ≈ 0.68 |
| yolox-m + efficientnet-lite3, AP@[.5:.95] | 0.719 | 0.629 | ≈ 0.71 |
| classifier top-1 on test crops | ≈ 0.97 | 0.827 | ≥ 0.95 |

None of these scripts have been run yet (no DFC/GPU on the Hailo host). Steps 2, 3 and 6
are checks that stop you early if a setup error sneaks in.

---

## Scripts (this folder)

| step | script | env |
|---|---|---|
| 2 | `export_yolox_onnx.py`: checkpoint → ONNX, finds the 9 head output nodes, ORT check | training env |
| 3 | `export_efficientnet_onnx.py`: checkpoint → ONNX, ONNX float accuracy, prints the alls normalization line | training env |
| 4 | `parse_onnx.py`: ONNX → parsed HAR (+ NMS config json for the detector) | `hailo_dfc` |
| 5 | `prepare_calib.py`: calibration `.npy`, preprocessed exactly like the runtime | either |
| 6–8 | `optimize.py`: parsed HAR → quantized HAR, with stronger settings | `hailo_dfc` + GPU |
| 6–7 | `eval_emulated_classifier.py`: classifier accuracy in the emulator, float vs int8 | `hailo_dfc` |
| 7, 8 | `compile_hef.py`: quantized HAR → HEF | `hailo_dfc` |
| – | `alls/*.alls`: fallback model scripts, if the originals are lost | |

On the Hailo host (`/data-ext/amo`): `eval_classifier_hailo.py` and `run_all_hailo.py` (step 9).

---

## 0. Inputs needed on the GPU machine

- **Checkpoints:**
  - `YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth`
  - the **exp file** used for training it (`yolox_m_leaky_zeus.py`, with `act="lrelu"`)
  - `output/best_model.pt` (classifier)
- **Datasets:**
  - `zeus-cropped/images/train` (detector calibration)
  - `zeus-cropped-classification/{train,test}`
- **Classifier training script** `train_efficientnet_lite3.py`, to check its preprocessing (step 3).
- **Original alls and NMS config, if you still have them.** They're under
  `hailo_model_zoo/hailo_model_zoo/cfg/alls/generic/` (`*.alls`), plus
  `nms_config_yolox_m_leaky_zeus_zeuscropped.json`. If they're gone, use `alls/` and step 4's
  `--write-nms-config`.

## 1. Setup

Copy over from the Hailo host:
```bash
scp -r odas@compas:/data-ext/amo/hailo_reoptimize odas@compas:/data-ext/amo/classes.json ~/zeus-training/
```

Set paths in every shell (adjust to your layout):
```bash
cd ~/zeus-training/hailo_reoptimize
export MT=~/zeus-training/master_thesis
export DET_DS=$MT/data/zeus-cropped
export CLS_DS=$MT/data/zeus-cropped-classification
export YOLOX_EXP=<path>/yolox_m_leaky_zeus.py
export YOLOX_CKPT=$DET_DS/YOLOX_outputs/yolox_m_leaky_zeus/best_ckpt.pth
export CLS_CKPT=~/zeus-training/output/best_model.pt
export CLS_ALLS=<original efficientnet alls, or alls/efficientnet_lite3_zeus_cropped.alls>
export DET_ALLS=<original yolox alls, or alls/yolox_m_leaky_zeus_zeuscropped.alls>
mkdir -p onnx har calib out
```

Install the export tools into the **training** env (the one with yolox, torch and timm):
```bash
conda activate master_thesis
pip install onnx onnxsim onnxruntime
```

Sanity check the **DFC** env:
```bash
conda activate hailo_dfc
nvidia-smi && python -c "import hailo_sdk_client as h; print(h.__version__)"
```

## 2. YOLOX → ONNX (training env)

```bash
conda activate master_thesis
python export_yolox_onnx.py -f $YOLOX_EXP -c $YOLOX_CKPT --out onnx/yolox_m_leaky_zeus.onnx
```

✅ **Check:**
- "ONNX vs PyTorch max abs diff" is small (about 1e-3 or less).
- It prints 3 strides with `reg/obj/cls` nodes each, also written to `onnx/yolox_m_leaky_zeus.end_nodes.json`.

If it finds fewer than 3 levels, open the ONNX in netron. Pass the 9 nodes feeding the
per-level Concats to `parse_onnx.py` as `--end-node`s, in the order reg, obj, cls per stride.

## 3. EfficientNet → ONNX, and pin down its preprocessing (training env)

```bash
python export_efficientnet_onnx.py --ckpt $CLS_CKPT --classes ../classes.json \
    --out onnx/efficientnet_lite3_zeus_cropped.onnx --test-dir $CLS_DS/test
```

✅ **Check:** "ONNX float top-1" should be about 0.97.
- **If it's lower**, the assumed preprocessing is wrong. Look at the val/test transforms in
  `train_efficientnet_lite3.py` and re-run with the matching `--mean/--std` (0–1 scale),
  `--color` and `--interp`, until you get about 0.97.
  - e.g. ImageNet stats: `--mean 0.485 0.456 0.406 --std 0.229 0.224 0.225`
- **Then** put the printed `normalization1 = normalization([...], [...])` line into `$CLS_ALLS`,
  replacing its normalization line. If the model trained on RGB (`--color rgb`), the alls keeps
  `input_conversion(bgr_to_rgb)`; for `--color bgr`, remove that line.

## 4. ONNX → parsed HAR (DFC env)

Keep the net names; the runtime and the alls layer names depend on them.
```bash
conda activate hailo_dfc
python parse_onnx.py --onnx onnx/yolox_m_leaky_zeus.onnx --net-name yolox_m_leaky_zeus_zeuscropped \
    --nodes onnx/yolox_m_leaky_zeus.end_nodes.json --out har/yolox_m_parsed.har \
    --write-nms-config nms_config_yolox_m_leaky_zeus_zeuscropped.generated.json
python parse_onnx.py --onnx onnx/efficientnet_lite3_zeus_cropped.onnx \
    --net-name efficientnet_lite3_zeus_cropped --out har/efficientnet_parsed.har
```

✅ **Check:**
- The detector reports **9** output layers; the classifier reports 1.
- **NMS config:**
  - **If you have the original json**, keep using it. Compare its `reg/obj/cls_layer` names
    with the generated file: they must be the same layers.
  - **If it's lost**, use the generated one: rename it to
    `nms_config_yolox_m_leaky_zeus_zeuscropped.json`, the name the alls points at. Keep
    `nms_scores_th` 0.2 for deployment. Set 0.01 if you want benchmark-comparable mAP (then
    filter at conf 0.3 on the host).
- Make sure the `nms_postprocess(...)` path in `$DET_ALLS` points at that json.

## 5. Calibration sets

```bash
python prepare_calib.py detector   --images-dir $DET_DS/images/train --n 1024 --color bgr --out calib/det_bgr.npy
python prepare_calib.py classifier --crops-dir $CLS_DS/train --n 2048 --color bgr --out calib/cls_bgr.npy
python prepare_calib.py classifier --crops-dir $CLS_DS/train --n 2048 --color rgb --out calib/cls_rgb.npy
```
The detector set is about 1.5 GB on disk and takes about 6 GB of RAM during optimize.

## 6. Classifier: find the calibration color, then optimize (GPU)

**Quick probe** (minutes): does the DFC want the calibration data before or after `input_conversion`?
```bash
python optimize.py --har har/efficientnet_parsed.har --base-alls $CLS_ALLS --calib calib/cls_bgr.npy \
    --calib-size 64 --opt-level 0 --out out/cls_probe.har
for c in bgr rgb; do
  python eval_emulated_classifier.py --har out/cls_probe.har --test-dir $CLS_DS/test \
      --classes ../classes.json --context fp_optimized --color $c
done
export CCOL=<the color that gave ≈ 0.97>
```
If neither gives about 0.97 (i.e. step 3's number), the alls normalization or conversion is
wrong. Go back to step 3.

**Full run** (about 0.5–2 h):
```bash
python optimize.py --har har/efficientnet_parsed.har --base-alls $CLS_ALLS --calib calib/cls_$CCOL.npy \
    --opt-level 2 --finetune --ft-epochs 4 \
    --a16-layer efficientnet_lite3_zeus_cropped/fc1 \
    --extra-line "performance_param(compiler_optimization_level=max)" \
    --out out/cls_q.har
python eval_emulated_classifier.py --har out/cls_q.har --test-dir $CLS_DS/test \
    --classes ../classes.json --context fp_optimized --context quantized --color $CCOL
```
Use `optimize.py --har har/efficientnet_parsed.har --list-layers` to confirm `fc1`'s exact name.

✅ **Check:** the `quantized` accuracy should be ≥ 0.95. If not, try in order:
1. `--opt-level 4`, or `--ft-epochs 8`
2. `--n 4096` in step 5
3. `hailo analyze-noise out/cls_q.har --data-path calib/cls_$CCOL.npy` (see `--help`), then add
   the noisiest layers as extra `--a16-layer`s
4. If compile fails on a16 layers, drop them and rely on finetuning

## 7. Classifier → HEF

```bash
python compile_hef.py --har out/cls_q.har --out out/efficientnet_lite3_zeus_cropped_v2.hef
```

## 8. Detector: optimize + compile (GPU, several hours)

The `--drop-line bgr_to_rgb` removes the color bug from an original alls. The template in
`alls/` already lacks that line.
```bash
python optimize.py --har har/yolox_m_parsed.har --list-layers | grep -i conv | tail -12
python optimize.py --har har/yolox_m_parsed.har --base-alls $DET_ALLS --drop-line bgr_to_rgb \
    --calib calib/det_bgr.npy --opt-level 2 --finetune --ft-epochs 4 \
    --a16-layer <the 9 output convs, e.g. yolox_m_leaky_zeus_zeuscropped/conv97> \
    --extra-line "performance_param(compiler_optimization_level=max)" \
    --out out/det_q.har
cat out/det_q.alls        # confirm: no bgr_to_rgb, nms_postprocess present
python compile_hef.py --har out/det_q.har --out out/yolox_m_leaky_zeus_zeuscropped_v2.hef
```
The output convs are the layers named in the NMS config's `reg/obj/cls_layer`. There's no
emulator check for the detector, so it's verified on hardware in step 9.

## 9. Verify on the Hailo host

```bash
scp out/*_v2.hef odas@compas:/data-ext/amo/
# on compas:
cd /data-ext/amo && source venv/bin/activate
python eval_classifier_hailo.py --classifier-hef efficientnet_lite3_zeus_cropped_v2.hef
python run_all_hailo.py --output-dir ./eval_output_hailo_v2 --no-detector-bgr-to-rgb \
    --detector-hef yolox_m_leaky_zeus_zeuscropped_v2.hef \
    --classifier-hef efficientnet_lite3_zeus_cropped_v2.hef
```
- **`--no-detector-bgr-to-rgb` is required with the rebuilt detector**, because it no longer has
  the on-chip conversion. Keep the default when using the old detector HEF.
- If you used the fallback classifier alls without a softmax line, add `--classifier-raw-logits`.
- Compare `eval_output_hailo_v2/report.json` with `eval_output_hailo_det_rgb/report.json`
  (the current HEFs with the color fix). Swap in one new HEF at a time to see what each gains.

---

## Reference: alls lines used

| line | effect |
|---|---|
| `normalization1 = normalization(mean*255, std*255)` | on-chip input normalization; must match training |
| `input_conversion1 = input_conversion(bgr_to_rgb)` | on-chip channel swap; only for models trained on RGB (the classifier, not YOLOX) |
| `nms_postprocess("<json>", meta_arch=yolox, engine=cpu)` | YOLOX box decoding + NMS inside HailoRT |
| `model_optimization_flavor(optimization_level=2, compression_level=0)` | 2 = equalization + bias correction + finetune; 4 = heaviest. `compression_level=0` means no 4-bit weights |
| `model_optimization_config(calibration, batch_size=8, calibset_size=N)` | use all N calibration images (default is only 64) |
| `post_quantization_optimization(finetune, policy=enabled, ...)` | QFT: short distillation of the int8 model toward the float one |
| `quantization_param(<layer>, precision_mode=a16_w16)` | 16-bit for that layer (output layers) |
| `performance_param(compiler_optimization_level=max)` | compile only: longer search for a faster allocation |
