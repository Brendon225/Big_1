# Stage4 Final Evaluation Report

## Scope

This report summarizes test-split F1 evaluation for Stage2 B2-coarse and Stage3 B4/B5/B7 on ChemProtSent and CDRIntra. Main F1 metrics exclude `NO_RELATION`.

## Key Findings

1. On ChemProtSent, B7 achieves the best Micro-F1 (0.7246 ± 0.0106), improving over Stage2 B2-coarse by 0.0771.
2. On CDRIntra, B4 achieves the highest test Micro-F1 (0.6818 ± 0.0078), improving over Stage2 B2-coarse by 0.1707.
3. B7 remains useful as the unified gated model, but on CDRIntra its test Micro-F1 (0.6737 ± 0.0248) is slightly below B5 (0.6764 ± 0.0102) and B4.
4. All evaluated runs have invalid_rate = 0.0000, confirming that label-rerank produces valid relation labels.

## Main Table

| Dataset | Model | P | R | Micro-F1 | Macro-F1 | mapped exact | invalid rate |
|---|---|---:|---:|---:|---:|---:|---:|
| CDRIntra | Stage2 B2-coarse | 0.6460 ± 0.0985 | 0.4883 ± 0.2807 | 0.5111 ± 0.1186 | 0.5111 ± 0.1186 | 0.6889 ± 0.0200 | 0.0000 |
| CDRIntra | B4 semantics-only | 0.6859 ± 0.0087 | 0.6782 ± 0.0211 | 0.6818 ± 0.0078 | 0.6818 ± 0.0078 | 0.7661 ± 0.0031 | 0.0000 |
| CDRIntra | B5 dual-view concat | 0.6942 ± 0.0078 | 0.6595 ± 0.0124 | 0.6764 ± 0.0102 | 0.6764 ± 0.0102 | 0.7667 ± 0.0065 | 0.0000 |
| CDRIntra | B7 gated dual-view | 0.6760 ± 0.0120 | 0.6738 ± 0.0576 | 0.6737 ± 0.0248 | 0.6737 ± 0.0248 | 0.7596 ± 0.0068 | 0.0000 |
| ChemProtSent | Stage2 B2-coarse | 0.7345 ± 0.0651 | 0.5890 ± 0.0893 | 0.6476 ± 0.0285 | 0.6057 ± 0.0326 | 0.8711 ± 0.0045 | 0.0000 |
| ChemProtSent | B4 semantics-only | 0.7578 ± 0.0231 | 0.6687 ± 0.0340 | 0.7097 ± 0.0124 | 0.6872 ± 0.0196 | 0.8904 ± 0.0027 | 0.0000 |
| ChemProtSent | B5 dual-view concat | 0.7590 ± 0.0256 | 0.6740 ± 0.0343 | 0.7133 ± 0.0136 | 0.6933 ± 0.0164 | 0.8910 ± 0.0044 | 0.0000 |
| ChemProtSent | B7 gated dual-view | 0.7685 ± 0.0178 | 0.6857 ± 0.0139 | 0.7246 ± 0.0106 | 0.7070 ± 0.0144 | 0.8956 ± 0.0049 | 0.0000 |

## Interpretation

The test F1 results confirm the central Stage3 claim that explicit semantic/syntactic modeling substantially improves over dependency linearization. However, the strongest variant is dataset-dependent: gated fusion is best on ChemProtSent, while the semantics-only B4 model is strongest on CDRIntra under test Micro-F1. This suggests that CDRIntra may benefit more from robust biomedical semantic representations than from additional fusion complexity, whereas ChemProtSent still gains from the adaptive dual-view gate.

A conservative paper narrative should therefore present B7 as the final unified model with the strongest ChemProtSent performance and stable gated behavior, while acknowledging that CDRIntra favors simpler semantic or dual-view variants in F1.
