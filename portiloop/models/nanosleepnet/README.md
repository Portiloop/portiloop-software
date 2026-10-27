# NanoSleepNet-TCN sleep-staging model

Used by `portiloop/src/custom/sleep_staging.py` (pipeline "Sleep staging (NanoSleepNet)"), run with NumPy.

- Network: NanoSleepNet-TCN, causal mode (Liu, Zhang & Wei, manuscript submitted to IEEE TBME), code from
  guisongliu/Nano-SleepNet @ 0420482 (Apache-2.0 for the original code), unchanged: an epoch encoder makes a
  64-value token of each 30 s epoch, a causal temporal convolutional network stages each epoch from its token and
  those of the 9 epochs before it. 17,098 parameters.
- Training: no trained weights are published; trained in this project (2026-10-26) on Sleep-EDF-20
  (sleep-cassette subjects 0-19), Fpz-Cz, 100 Hz, 30 s epochs, 20-fold leave-one-subject-out with the same folds
  and validation subjects as the TinySleepNet model. The authors do not publish their training procedure; used here:
  sequences of 10 epochs at a random offset in each pass, Adam (learning rate 1e-3, 32 sequences per step),
  unweighted cross-entropy, the pass with the best Cohen's kappa on the validation subjects, stop after 30 passes
  without improvement. This is fold 13 (test subject 13; validation subjects 2 and 17; best pass 18).
- 20-fold result (each subject staged by the fold that never saw it, 42,307 epochs): accuracy 0.844, macro-F1 0.776,
  kappa 0.785 (TinySleepNet on the same epochs: 0.836, 0.789, 0.777).
- Why fold 13: the 20 fold models and their ensemble were compared on sleep-cassette subjects 20-82, which none of
  them was trained on, split into a selection half and a report half (same split and rule as the TinySleepNet
  model). Fold 13 had the highest kappa on the selection half (0.644; ensemble 0.630); on the report half:
  accuracy 0.717, macro-F1 0.640, kappa 0.607 (TinySleepNet fold 7: 0.734, 0.656, 0.625).
- `nanosleepnet.npz`: the weights of the eval-mode PyTorch model, every BatchNorm reduced to a scale and a shift.
  The NumPy implementation gives the same class as PyTorch (float64) on all 841 epochs of SC4001E0, logits within
  1.1e-5, for all 20 folds.

| file | SHA256 |
|------|--------|
| nanosleepnet.npz | 79ca1cf8b52d42b2211ec43c6c38f0bcd0fe53cfc43f6975bd43aaec7282726d |

Not validated on Portiloop recordings: no expert-scored Portiloop data was available.
