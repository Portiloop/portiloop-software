# TinySleepNet sleep-staging model

Used by `portiloop/src/custom/sleep_staging.py` (pipeline "Sleep staging (TinySleepNet)").

- Network: TinySleepNet ("model-mod-8": CNN + 1-layer peephole LSTM, 128 units), akaraspt/tinysleepnet @ 3e0a19d.
- Training: Sleep-EDF-20 (sleep-cassette subjects 0-19), Fpz-Cz, 100 Hz, 30 s epochs, 20-fold leave-one-subject-out.
  This is fold 7 (test subject 7; 17 training and 2 validation subjects), best checkpoint chosen on the validation set
  (step 50348). Fold-7 test accuracy 89.6 %, macro-F1 84.0 (training log and an unmodified re-run agree).
- Why fold 7: the 20 fold models and their ensemble were compared on sleep-cassette subjects 20-82, which none of
  them was trained on, split into a selection half and a report half (fixed seed). Fold 7 had the highest Cohen's
  kappa on the selection half (0.699; ensemble 0.697, fold 0 0.671); on the report half: accuracy 0.734,
  macro-F1 0.656, kappa 0.625.
- Conversion: weights read from the TensorFlow 1.13 checkpoint, single-step Keras model with the LSTM state as
  input/output, float32 TFLite (TensorFlow 2.5). Same class as the original model on all 976 epochs of SC4071E0
  (logits within 1.2e-5 on a PC).
- `probe.npz`: one input epoch and the LSTM state the original model reaches on it from zero, used to tell the
  two state outputs apart.

| file | SHA256 |
|------|--------|
| tinysleepnet_float.tflite | 2d814fb2aa55d3c85bba2ba67b2ddbca14879452ed81c99d4a4edb41cb3d04b7 |
| probe.npz | 1d8e2c7e86afbe590af873f40de70be1dd1701b3a433dba46ad7bfba511ac565 |

Not validated on Portiloop recordings: no expert-scored Portiloop data was available.
