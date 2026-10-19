# STARR-Net

STARR-Net estimates a Beta distribution over relative fatal-crash risk from three satellite-image scales, a frozen GeoCLIP location embedding, and cyclic hour/weekday features.

## Code structure

- `starr_train.py` trains one factorial variant, selects the checkpoint by validation Brier score, and writes the headline test metrics.
- `starr_predict.py` evaluates one three-scale image query at a latitude, longitude, weekday, and hour.
- `export_thesis_eval.py` exports aligned per-sample predictions through the same evaluation path used for the headline metrics.
- `compare_results.py` collects Brier, AUC, ECE, selective AUSC, F1, and recall across the five variants.
- `utils/starr_model.py` defines the shared ResNet-50 image stream, optional location and temporal streams, fusion head, and Beta or deterministic output.
- `utils/mydatasets.py` loads zoom levels 17, 18, and 19 and constructs the Beta training targets.
- `utils/mydatasets_temporal.py` attaches recorded positive crash times and seeded uniform negative query times.
- `utils/wasserstein_beta_loss.py` implements the two-moment Wasserstein-2 surrogate.
- `geoclip/model/location_encoder.py` provides the frozen GeoCLIP location encoder.
- `launch_variants.slurm`, `launch_eval.slurm`, and `launch_starr.slurm` are the cluster job scripts used for training and evaluation.

The five variant names are `mscm_ms`, `betarisk`, `starr_loc`, `starr_time`, and `starr_full`.

## Weights and data

The MSCM-pretrained ResNet-50 checkpoint and GeoCLIP location-encoder weights are not included because of their size. Place them at:

```text
ckpts/pre-trained_weights/pre-trained_weights_mscm.pth
geoclip/model/weights/location_encoder_weights.pth
```

The imagery and metadata remain in the MSCM data directory supplied with `--data-root`.
