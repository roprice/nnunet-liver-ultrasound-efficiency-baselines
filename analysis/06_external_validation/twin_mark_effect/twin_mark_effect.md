# Effect of caliper marks on predictions

Prediction snapshot `195716b4a6ec`. 380 hemangioma Clean frames (190 patients) with usable twin-derived proxies, each paired with its caliper twin: the same pixels plus the caliper marks. Both frames go through the same 5 final nnU-Net default-augmentation models (588-image AUL training pool, seed 42, 150 epochs, folds 0–4), so any difference between the two predictions is caused by the marks.

Rates are mean ± sample SD across the five fold models on the same pairs; differences are marked twin minus Clean frame, as proportions, computed per fold. The study's relevance threshold is 0.03. Overlap and centroid detection use the same circular proxy for both frames of a pair. These are hemangioma marks only; HCC frames come from several scanners with other mark styles and are not tested here.

## Detection with and without marks

| metric | frame | value |
|---|---|---|
| presence | Clean | 0.940 ± 0.006 |
| presence | marked twin | 0.888 ± 0.017 |
| presence | difference | -0.052 ± 0.016 |
| overlap_iou_00 | Clean | 0.758 ± 0.017 |
| overlap_iou_00 | marked twin | 0.546 ± 0.051 |
| overlap_iou_00 | difference | -0.212 ± 0.044 |
| centroid_deq_100 | Clean | 0.662 ± 0.023 |
| centroid_deq_100 | marked twin | 0.468 ± 0.039 |
| centroid_deq_100 | difference | -0.193 ± 0.042 |

## Pair-level agreement

Components are retained predicted components (same size filter as everywhere else). A component is only in one frame's prediction when it has no pixel overlap with the other frame's retained mask.

| measure | value |
|---|---|
| pairs with identical retained masks | 0.052 ± 0.006 |
| components only in the marked twin, per pair | 0.217 ± 0.085 |
| components only in the Clean frame, per pair | 0.244 ± 0.043 |
| presence only on the Clean frame (pairs) | 22.8 ± 5.1 |
| presence only on the marked twin (pairs) | 3.0 ± 1.9 |
