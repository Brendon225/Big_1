# Stage4 F1 Summary

Metrics are computed on the test split. Precision, recall, Micro-F1, and Macro-F1 exclude `NO_RELATION`.

| Dataset | Model | P | R | Micro-F1 | Macro-F1 | mapped exact | invalid rate |
|---|---|---:|---:|---:|---:|---:|---:|
| CDRIntra | Stage2 B2-coarse | 0.6460 ± 0.0985 | 0.4883 ± 0.2807 | 0.5111 ± 0.1186 | 0.5111 ± 0.1186 | 0.6889 ± 0.0200 | 0.0000 |
| CDRIntra | **B4 semantics-only** | 0.6859 ± 0.0087 | 0.6782 ± 0.0211 | 0.6818 ± 0.0078 | 0.6818 ± 0.0078 | 0.7661 ± 0.0031 | 0.0000 |
| CDRIntra | B5 dual-view concat | 0.6942 ± 0.0078 | 0.6595 ± 0.0124 | 0.6764 ± 0.0102 | 0.6764 ± 0.0102 | 0.7667 ± 0.0065 | 0.0000 |
| CDRIntra | B7 gated dual-view | 0.6760 ± 0.0120 | 0.6738 ± 0.0576 | 0.6737 ± 0.0248 | 0.6737 ± 0.0248 | 0.7596 ± 0.0068 | 0.0000 |
| ChemProtSent | Stage2 B2-coarse | 0.7345 ± 0.0651 | 0.5890 ± 0.0893 | 0.6476 ± 0.0285 | 0.6057 ± 0.0326 | 0.8711 ± 0.0045 | 0.0000 |
| ChemProtSent | B4 semantics-only | 0.7578 ± 0.0231 | 0.6687 ± 0.0340 | 0.7097 ± 0.0124 | 0.6872 ± 0.0196 | 0.8904 ± 0.0027 | 0.0000 |
| ChemProtSent | B5 dual-view concat | 0.7590 ± 0.0256 | 0.6740 ± 0.0343 | 0.7133 ± 0.0136 | 0.6933 ± 0.0164 | 0.8910 ± 0.0044 | 0.0000 |
| ChemProtSent | **B7 gated dual-view** | 0.7685 ± 0.0178 | 0.6857 ± 0.0139 | 0.7246 ± 0.0106 | 0.7070 ± 0.0144 | 0.8956 ± 0.0049 | 0.0000 |
