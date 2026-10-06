# Original saved model scripts

These five files are byte-for-byte copies of the supplied Python training scripts. Their hashes are recorded in the repository manifest. They are preserved for inspection; automatic verification uses `../evaluate_saved_results.py` and does not run them.

The scripts import TensorFlow/Keras, keras-tcn, NumPy, pandas, scikit-learn and Matplotlib. The original environment lockfile was not available. They include historical APIs such as `error_bad_lines=False`, `keras.layers.recurrent`, and mixed standalone Keras / tf.keras imports, so installing current packages is not a validated way to recreate the training environment.

They expect relative CSV paths in the working directory. Copy the relevant dataset into an isolated scratch directory before adapting a script. The supplied ALA–VMD script is configured for NASA B5, eight IMFs, a split at 60 and look_back=1. CALCE CSVs and saved outputs are included, but a separate recorded CALCE training configuration was not available.

Execution starts training (100 epochs in the supplied ALA–VMD model) and may write or plot outputs. Do not run a legacy script as a quick smoke test or assume that a newly trained model will exactly match the saved CSV results.
