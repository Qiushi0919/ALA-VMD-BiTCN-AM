# Reproduction record and method correspondence

## Reused artifacts

Original model source, capacity CSVs, decomposition CSVs and ten prediction-result CSVs were copied unchanged. The release process reran only the standard-library verification/metric calculation and website build. It did not retrain neural networks, rerun ALA search, smooth predictions, shift targets or extrapolate curves.

The video and numerical figures use the NASA B5 saved capacity, eight saved ALA–VMD components and saved prediction output. The Visio-derived topology is the paper's method schematic. Diagram variables and attention nodes are symbolic structure, not measured activations or exported attention weights. No synthetic optimization history or attention values are presented.

## Pipeline-to-code mapping

| Stage | Available artifact / implementation |
| --- | --- |
| Capacity input | `data/capacity/B5.csv`, `CS2_38.csv` |
| ALA parameter selection → VMD | Saved decomposition files; search source, final α logs and iterative optimization history were absent |
| Components → two model inputs | Supplied ALA script uses u1 and the sum of remaining IMFs; raw capacity is the final column and excluded from that sum |
| Scaling and one-step input | MinMaxScaler and `create_dataset`; NASA split at 60, look_back=1; targets start at index 61 |
| Forward / reverse TCN | Reverse input along time, process both branches, reverse backward output and concatenate features |
| Attention | Permute → Dense(T, softmax) → Permute → Multiply |
| Output | GlobalAveragePooling1D → dense layers → predicted capacity of each of two independent models |
| Aggregation | Sum the two inverse-transformed outputs; compare with the observed final CSV column |
| End of life | First capacity target strictly below the failure threshold in the saved test interval |

## Important implementation details

The paper schematic uses weight-normalization blocks, residual 1×1 convolutions, plus-symbol feature fusion and a multi-position attention illustration. The supplied saved Python model concatenates forward/backward features and has additional dense output layers; the extra 1×1 convolution is commented out. The video preserves the requested paper drawing rather than presenting it as a literal graph trace of every saved-code layer.

The supplied look_back is **1**. Its temporal softmax length is consequently 1, so the corresponding normalized temporal weight is identically 1. The diagram's h/e/α nodes remain symbols; they do not claim an observed nontrivial distribution of temporal attention weights.

The original script fits MinMaxScaler separately on train and test portions, reusing the scaler reference. This protocol is preserved in the source and recorded here, rather than silently corrected while claiming reproduction of the saved experiment.

No model checkpoints, ALA optimizer source, complete search logs or environment lockfile were present. Therefore checkpoint inference and exact end-to-end retraining were not performed or validated. Numerical reproducibility in this release means alignment and metrics computed from the saved CSV outputs. The verified report is `results/verified_metrics.json`.
