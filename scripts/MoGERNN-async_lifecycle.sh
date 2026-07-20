# Paired Lifecycle Event Study (PLES) inference.
# Keep training and testing separate: this command loads the checkpoint identified
# by the training configuration and evaluates it with a dataset-specific manifest.

python main.py \
  -c configs/MoGERNN/METRLA_async_lifecycle_100.yaml \
  --mode test_async

# Final ALFNet model under the same PLES protocol.
python main.py \
  -c configs/ALFNet/rf_alf_sd.yaml \
  --mode test_async
