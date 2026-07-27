# Magnum DQGS=128 B70 Report

Date: 2026-07-26

Device: Intel Arc Pro B70 at PCI `0000:08:00.0` (`GPU.2`)

Runtime: Wondernuttz OpenVINO GPU fork, prefix cache disabled for measurement

Method: fresh process per setting, exact tokenized prompts, one same-shape
warmup, two timed PP runs, deterministic generation, and 128 forced decode
tokens. PP values below are the mean of the two timed runs.

## Results

| Model | Precision | DQGS | 512 PP | 2K PP | 6K PP | Decode |
|---|---:|---:|---:|---:|---:|---:|
| Magnum v4 12B Anthracite | INT4 | 0 | 5,095 | 4,481 | 3,897 | 72.05 |
| Magnum v4 12B Anthracite | INT4 | 128 | 5,894 | 6,268 | 5,134 | 72.37 |
| Lumimaid-Magnum v4 12B | INT4 | 0 | 5,131 | 4,488 | 3,907 | 71.89 |
| Lumimaid-Magnum v4 12B | INT4 | 128 | 5,912 | 6,267 | 5,146 | 71.99 |
| Magnum Diamond 24B | INT4 | 0 | 2,590 | 2,324 | 2,134 | 42.04 |
| Magnum Diamond 24B | INT4 | 128 | 3,062 | 3,202 | 2,974 | 41.97 |
| Lumimaid-Magnum v4 12B | INT8 | 0 | 5,076 | 4,545 | 4,075 | 43.28 |
| Lumimaid-Magnum v4 12B | INT8 | 128 | 7,138 | 7,105 | 5,533 | 43.27 |
| Magnum Diamond 24B | INT8 | 0 | 2,680 | 2,472 | 2,237 | 22.20 |
| Magnum Diamond 24B | INT8 | 128 | 4,017 | 3,755 | 3,225 | 22.19 |

## Gain From DQGS=128

| Model | Precision | 512 | 2K | 6K |
|---|---:|---:|---:|---:|
| Magnum v4 12B Anthracite | INT4 | +15.7% | +39.9% | +31.8% |
| Lumimaid-Magnum v4 12B | INT4 | +15.2% | +39.7% | +31.7% |
| Magnum Diamond 24B | INT4 | +18.2% | +37.8% | +39.4% |
| Lumimaid-Magnum v4 12B | INT8 | +40.6% | +56.3% | +35.8% |
| Magnum Diamond 24B | INT8 | +49.9% | +51.9% | +44.2% |

## Output Gate

The 512, 2K, 6K, and 128-token decode hashes match `DQGS=0` for each model.

The longer coherence case produced 6,610 input tokens and 128 output tokens:

| DQGS | PP tok/s | Decode tok/s | Token SHA-256 |
|---:|---:|---:|---|
| 0 | 3,360.2 | 62.53 | `38477c679dadad9d019ccdccb5fbd863ac3c461fa8278cc65ded946522fe26a2` |
| 128 | 4,739.6 | 62.61 | `38477c679dadad9d019ccdccb5fbd863ac3c461fa8278cc65ded946522fe26a2` |

Both runs passed the three exact-detail checks and the Entry 121 style check.

## Deployment

`OV_DQGS=128` is set on all five Magnum entries in
`/home/wondernutts/ovhub/models.json`. It changes runtime configuration only.
The model weights are unchanged.
