# Model comparison (test split, threshold tuned on val)

| model       |   test_roc_auc |   test_pr_auc |   test_precision |   test_recall |   test_f1 |   test_fpr |   holdout_detection_rate |   holdout_roc_auc |   val_roc_auc |   latency_ms |   size_mb |
|:------------|---------------:|--------------:|-----------------:|--------------:|----------:|-----------:|-------------------------:|------------------:|--------------:|-------------:|----------:|
| baseline    |         0.9762 |        0.9758 |           0.9156 |        0.9208 |    0.9181 |     0.0747 |                   0.793  |            0.9435 |        0.9767 |       0.2299 |     4.17  |
| charlm      |         0.8969 |        0.8968 |           0.8556 |        0.7742 |    0.8129 |     0.115  |                   0.5257 |            0.8138 |        0.8967 |       0.9773 |     2.447 |
| canine_lora |         0.9291 |        0.9376 |           0.9056 |        0.8172 |    0.8591 |     0.075  |                   0.6783 |            0.9223 |        0.9287 |      42.9384 |     1.396 |
| hybrid      |         0.9346 |        0.9434 |           0.8966 |        0.8405 |    0.8677 |     0.0853 |                   0.7175 |            0.9276 |        0.9339 |     nan      |   nan     |

Hybrid = LR stack of char-LM log-perplexity + canine_lora logit (coef=[1.168, 0.585]).
holdout_* = unseen DGA families (matsnu, pykspa, symmi).
