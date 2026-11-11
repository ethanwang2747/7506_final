# Experiment summary

## E1 clean behavior and input dependence

| task | split | group | AUC | delta | img | question | gray | gen yes | gen no | margin yes | margin no |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spatial | dev | woman/to the left of/man | 0.785 | +1.69 | +1.64 | +3.65 | +2.22 | 0.90 | 0.25 | 0.90 | 0.25 |
| spatial | dev | man/to the right of/woman | 0.735 | +1.17 | +1.18 | +2.44 | +1.02 | 0.75 | 0.75 | 0.75 | 0.75 |
| spatial | dev | man/to the left of/person | 0.757 | +0.71 | +0.71 | +1.72 | +0.77 | 0.65 | 0.75 | 0.65 | 0.75 |
| spatial | dev | person/to the right of/man | 0.772 | +0.76 | +0.76 | +1.41 | -0.04 | 0.35 | 0.85 | 0.35 | 0.85 |
| spatial | dev | tree/to the left of/building | 0.650 | +0.45 | +0.34 | +1.25 | +0.42 | 0.35 | 0.85 | 0.40 | 0.80 |
| spatial | dev | chair/to the left of/table | 0.816 | +0.52 | +0.52 | -0.00 | -0.25 | 0.00 | 1.00 | 0.00 | 1.00 |
| spatial | test | woman/to the left of/man | 0.605 | +0.58 | +0.63 | +2.63 | +1.31 | 0.85 | 0.15 | 0.85 | 0.15 |
| spatial | test | man/to the right of/woman | 0.618 | +0.54 | +0.46 | +1.59 | +0.27 | 0.50 | 0.80 | 0.50 | 0.80 |
| spatial | test | man/to the left of/person | 0.405 | -0.21 | -0.24 | +1.10 | +0.30 | 0.25 | 0.55 | 0.25 | 0.55 |
| spatial | test | person/to the right of/man | 0.400 | -0.27 | -0.37 | +0.68 | -0.61 | 0.25 | 0.75 | 0.25 | 0.75 |
| spatial | test | tree/to the left of/building | 0.455 | +0.01 | +0.04 | +1.27 | +0.50 | 0.40 | 0.60 | 0.50 | 0.60 |
| spatial | test | chair/to the left of/table | 0.725 | +0.47 | +0.49 | +0.00 | -0.11 | 0.10 | 1.00 | 0.10 | 1.00 |
| semantic | dev | fence/material/wood | 0.907 | +2.08 | +2.11 | +0.57 | +1.59 | 0.25 | 1.00 | 0.25 | 1.00 |
| semantic | dev | table/color/brown | 0.830 | +1.02 | +1.01 | +1.03 | +0.72 | 0.35 | 0.90 | 0.40 | 0.90 |
| semantic | dev | jacket/color/black | 0.769 | +1.19 | +1.19 | +1.07 | +0.69 | 0.33 | 0.83 | 0.33 | 0.83 |
| semantic | dev | building/color/white | 0.828 | +1.53 | +1.53 | +0.00 | +0.42 | 0.25 | 1.00 | 0.30 | 1.00 |
| semantic | dev | door/color/white | 0.815 | +1.31 | +1.32 | +0.34 | +1.06 | 0.25 | 0.95 | 0.25 | 0.95 |
| semantic | test | fence/material/wood | 0.890 | +2.40 | +2.53 | +1.04 | +1.88 | 0.40 | 0.95 | 0.40 | 0.95 |
| semantic | test | table/color/brown | 0.810 | +1.11 | +1.09 | +1.08 | +0.89 | 0.45 | 0.95 | 0.55 | 0.95 |
| semantic | test | jacket/color/black | 0.690 | +0.81 | +0.83 | +0.83 | +0.37 | 0.30 | 0.90 | 0.30 | 0.90 |
| semantic | test | building/color/white | 0.750 | +1.19 | +1.25 | +0.00 | +0.71 | 0.45 | 0.90 | 0.50 | 0.90 |
| semantic | test | door/color/white | 0.805 | +1.31 | +1.28 | +0.50 | +1.11 | 0.30 | 0.95 | 0.35 | 0.95 |

## E1 bootstrap confidence intervals

| task | split | group | AUC | AUC 95% CI | delta | delta 95% CI | n |
|---|---|---|---|---|---|---|---|
| spatial | dev | chair/to the left of/table | 0.809 | [0.551, 1.000] | +0.53 | [-0.06, +1.10] | 7 |
| spatial | dev | man/to the left of/person | 0.757 | [0.607, 0.898] | +0.71 | [+0.31, +1.12] | 20 |
| spatial | dev | man/to the right of/woman | 0.734 | [0.550, 0.900] | +1.18 | [+0.58, +1.75] | 20 |
| spatial | dev | person/to the right of/man | 0.772 | [0.625, 0.895] | +0.77 | [+0.43, +1.15] | 20 |
| spatial | dev | tree/to the left of/building | 0.651 | [0.482, 0.802] | +0.45 | [+0.02, +0.94] | 20 |
| spatial | dev | woman/to the left of/man | 0.786 | [0.642, 0.902] | +1.69 | [+0.96, +2.43] | 20 |
| spatial | test | chair/to the left of/table | 0.724 | [0.552, 0.883] | +0.47 | [+0.19, +0.71] | 20 |
| spatial | test | man/to the left of/person | 0.407 | [0.242, 0.585] | -0.22 | [-0.56, +0.10] | 20 |
| spatial | test | man/to the right of/woman | 0.619 | [0.420, 0.818] | +0.53 | [-0.16, +1.24] | 20 |
| spatial | test | person/to the right of/man | 0.394 | [0.215, 0.575] | -0.27 | [-0.65, +0.13] | 20 |
| spatial | test | tree/to the left of/building | 0.447 | [0.268, 0.627] | -0.00 | [-0.79, +0.84] | 20 |
| spatial | test | woman/to the left of/man | 0.599 | [0.400, 0.775] | +0.56 | [-0.22, +1.37] | 20 |
| semantic | dev | building/color/white | 0.824 | [0.662, 0.943] | +1.52 | [+0.73, +2.45] | 20 |
| semantic | dev | door/color/white | 0.821 | [0.677, 0.930] | +1.32 | [+0.60, +2.04] | 20 |
| semantic | dev | fence/material/wood | 0.905 | [0.810, 0.980] | +2.08 | [+1.41, +2.75] | 20 |
| semantic | dev | jacket/color/black | 0.769 | [0.605, 0.907] | +1.18 | [+0.47, +1.94] | 18 |
| semantic | dev | table/color/brown | 0.826 | [0.667, 0.943] | +1.00 | [+0.42, +1.53] | 20 |
| semantic | test | building/color/white | 0.748 | [0.603, 0.873] | +1.17 | [+0.39, +2.05] | 20 |
| semantic | test | door/color/white | 0.801 | [0.635, 0.928] | +1.30 | [+0.66, +1.98] | 20 |
| semantic | test | fence/material/wood | 0.889 | [0.750, 0.980] | +2.39 | [+1.53, +3.23] | 20 |
| semantic | test | jacket/color/black | 0.678 | [0.512, 0.833] | +0.80 | [+0.17, +1.50] | 20 |
| semantic | test | table/color/brown | 0.814 | [0.677, 0.940] | +1.12 | [+0.54, +1.71] | 20 |

## E1 resolution ablation (spatial test)

| group | AUC no-split | AUC splitting | delta no-split | delta splitting |
|---|---|---|---|---|
| chair/to the left of/table | 0.725 | 0.787 | +0.47 | +1.06 |
| man/to the left of/person | 0.405 | 0.477 | -0.21 | -0.15 |
| man/to the right of/woman | 0.618 | 0.723 | +0.54 | +1.73 |
| person/to the right of/man | 0.400 | 0.505 | -0.27 | -0.00 |
| tree/to the left of/building | 0.455 | 0.765 | +0.01 | +1.48 |
| woman/to the left of/man | 0.605 | 0.762 | +0.58 | +2.02 |

## E2 target region minus matched controls

| task | split | group | stage | delta m | 95% CI | n |
|---|---|---|---|---|---|---|
| spatial | test | chair/to the left of/table | S0 | -0.45 | [-0.63, -0.26] | 20 |
| spatial | test | chair/to the left of/table | S1 | -0.42 | [-0.60, -0.25] | 20 |
| spatial | test | chair/to the left of/table | S6 | -0.39 | [-0.56, -0.23] | 20 |
| spatial | test | chair/to the left of/table | S12 | -0.38 | [-0.57, -0.19] | 20 |
| spatial | test | chair/to the left of/table | S18 | -0.05 | [-0.09, -0.01] | 20 |
| spatial | test | man/to the left of/person | S0 | -0.02 | [-0.19, +0.15] | 20 |
| spatial | test | man/to the left of/person | S1 | +0.00 | [-0.16, +0.16] | 20 |
| spatial | test | man/to the left of/person | S6 | -0.08 | [-0.23, +0.05] | 20 |
| spatial | test | man/to the left of/person | S12 | -0.14 | [-0.28, -0.01] | 20 |
| spatial | test | man/to the left of/person | S18 | -0.02 | [-0.05, +0.02] | 20 |
| spatial | test | man/to the right of/woman | S0 | -0.62 | [-0.98, -0.29] | 20 |
| spatial | test | man/to the right of/woman | S1 | -0.67 | [-1.03, -0.35] | 20 |
| spatial | test | man/to the right of/woman | S6 | -0.74 | [-1.14, -0.39] | 20 |
| spatial | test | man/to the right of/woman | S12 | -0.45 | [-0.72, -0.21] | 20 |
| spatial | test | man/to the right of/woman | S18 | -0.03 | [-0.06, +0.01] | 20 |
| spatial | test | person/to the right of/man | S0 | -0.15 | [-0.27, -0.04] | 20 |
| spatial | test | person/to the right of/man | S1 | -0.10 | [-0.21, +0.02] | 20 |
| spatial | test | person/to the right of/man | S6 | -0.11 | [-0.22, -0.03] | 20 |
| spatial | test | person/to the right of/man | S12 | -0.12 | [-0.24, -0.02] | 20 |
| spatial | test | person/to the right of/man | S18 | -0.02 | [-0.06, +0.02] | 20 |
| spatial | test | tree/to the left of/building | S0 | -0.64 | [-1.09, -0.30] | 20 |
| spatial | test | tree/to the left of/building | S1 | -0.62 | [-1.04, -0.29] | 20 |
| spatial | test | tree/to the left of/building | S6 | -0.43 | [-0.73, -0.16] | 20 |
| spatial | test | tree/to the left of/building | S12 | -0.56 | [-1.03, -0.19] | 20 |
| spatial | test | tree/to the left of/building | S18 | +0.01 | [-0.02, +0.04] | 20 |
| spatial | test | woman/to the left of/man | S0 | -0.58 | [-1.10, -0.15] | 20 |
| spatial | test | woman/to the left of/man | S1 | -0.59 | [-1.10, -0.15] | 20 |
| spatial | test | woman/to the left of/man | S6 | -0.63 | [-1.12, -0.22] | 20 |
| spatial | test | woman/to the left of/man | S12 | -0.37 | [-0.62, -0.15] | 20 |
| spatial | test | woman/to the left of/man | S18 | -0.01 | [-0.04, +0.03] | 20 |
| semantic | test | building/color/white | S0 | -0.56 | [-1.14, +0.02] | 20 |
| semantic | test | building/color/white | S1 | -0.58 | [-1.16, -0.01] | 20 |
| semantic | test | building/color/white | S6 | -0.55 | [-1.10, -0.00] | 20 |
| semantic | test | building/color/white | S12 | -0.55 | [-1.08, -0.08] | 20 |
| semantic | test | building/color/white | S18 | -0.03 | [-0.08, +0.02] | 20 |
| semantic | test | door/color/white | S0 | -0.45 | [-0.82, -0.14] | 20 |
| semantic | test | door/color/white | S1 | -0.42 | [-0.76, -0.12] | 20 |
| semantic | test | door/color/white | S6 | -0.35 | [-0.63, -0.11] | 20 |
| semantic | test | door/color/white | S12 | -0.37 | [-0.56, -0.19] | 20 |
| semantic | test | door/color/white | S18 | +0.02 | [-0.01, +0.05] | 20 |
| semantic | test | fence/material/wood | S0 | -1.21 | [-1.58, -0.84] | 20 |
| semantic | test | fence/material/wood | S1 | -1.22 | [-1.60, -0.83] | 20 |
| semantic | test | fence/material/wood | S6 | -1.27 | [-1.65, -0.87] | 20 |
| semantic | test | fence/material/wood | S12 | -0.90 | [-1.22, -0.57] | 20 |
| semantic | test | fence/material/wood | S18 | -0.03 | [-0.07, +0.00] | 20 |
| semantic | test | jacket/color/black | S0 | -0.07 | [-0.23, +0.07] | 20 |
| semantic | test | jacket/color/black | S1 | -0.10 | [-0.25, +0.03] | 20 |
| semantic | test | jacket/color/black | S6 | -0.08 | [-0.21, +0.06] | 20 |
| semantic | test | jacket/color/black | S12 | -0.10 | [-0.21, -0.01] | 20 |
| semantic | test | jacket/color/black | S18 | +0.02 | [-0.01, +0.06] | 20 |
| semantic | test | table/color/brown | S0 | -0.50 | [-0.89, -0.11] | 20 |
| semantic | test | table/color/brown | S1 | -0.42 | [-0.80, -0.04] | 20 |
| semantic | test | table/color/brown | S6 | -0.42 | [-0.79, -0.03] | 20 |
| semantic | test | table/color/brown | S12 | -0.27 | [-0.60, +0.02] | 20 |
| semantic | test | table/color/brown | S18 | +0.01 | [-0.03, +0.05] | 20 |

## E2 splitting-resolution global-block ablation (S1, mean delta m)

| group | n | blocks | global target | global reference | global background | full block |
|---|---|---|---|---|---|---|
| woman/to the left of/man | 5 | 13.0 | -0.00 | +0.07 | -0.06 | -2.67 |
| man/to the right of/woman | 5 | 13.0 | -0.07 | -0.07 | -0.06 | -3.02 |
| tree/to the left of/building | 5 | 13.0 | +0.02 | -0.10 | -0.07 | -3.41 |
| chair/to the left of/table | 5 | 13.8 | +0.07 | -0.03 | -0.05 | -0.95 |

## E3 full visual block recovery

| task | split | group | stage | median R | 95% CI | frac R>0.5 | n |
|---|---|---|---|---|---|---|---|
| spatial | dev | chair/to the left of/table | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 7 |
| spatial | dev | chair/to the left of/table | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 7 |
| spatial | dev | chair/to the left of/table | S6 | +1.23 | [+0.85, +1.50] | 1.00 | 7 |
| spatial | dev | chair/to the left of/table | S12 | +1.01 | [+0.40, +1.51] | 0.71 | 7 |
| spatial | dev | chair/to the left of/table | S18 | +0.10 | [-0.11, +0.33] | 0.14 | 7 |
| spatial | dev | man/to the left of/person | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | man/to the left of/person | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | man/to the left of/person | S6 | +0.94 | [+0.66, +1.38] | 0.90 | 10 |
| spatial | dev | man/to the left of/person | S12 | +0.49 | [+0.16, +0.79] | 0.50 | 10 |
| spatial | dev | man/to the left of/person | S18 | +0.06 | [-0.02, +0.20] | 0.00 | 10 |
| spatial | dev | man/to the right of/woman | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | man/to the right of/woman | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | man/to the right of/woman | S6 | +0.98 | [+0.89, +1.12] | 1.00 | 10 |
| spatial | dev | man/to the right of/woman | S12 | +0.73 | [+0.33, +1.17] | 0.60 | 10 |
| spatial | dev | man/to the right of/woman | S18 | +0.00 | [-0.00, +0.04] | 0.00 | 10 |
| spatial | dev | person/to the right of/man | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | person/to the right of/man | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | person/to the right of/man | S6 | +0.94 | [+0.80, +1.25] | 1.00 | 10 |
| spatial | dev | person/to the right of/man | S12 | +0.85 | [+0.38, +1.23] | 0.80 | 10 |
| spatial | dev | person/to the right of/man | S18 | -0.01 | [-0.10, +0.13] | 0.10 | 10 |
| spatial | dev | tree/to the left of/building | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | tree/to the left of/building | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | tree/to the left of/building | S6 | +1.08 | [+0.80, +1.99] | 0.90 | 10 |
| spatial | dev | tree/to the left of/building | S12 | +0.22 | [-0.12, +1.10] | 0.50 | 10 |
| spatial | dev | tree/to the left of/building | S18 | -0.02 | [-0.20, +0.02] | 0.10 | 10 |
| spatial | dev | woman/to the left of/man | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | woman/to the left of/man | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| spatial | dev | woman/to the left of/man | S6 | +0.98 | [+0.88, +1.05] | 1.00 | 10 |
| spatial | dev | woman/to the left of/man | S12 | +0.55 | [+0.26, +0.84] | 0.50 | 10 |
| spatial | dev | woman/to the left of/man | S18 | -0.03 | [-0.17, +0.01] | 0.00 | 10 |
| spatial | test | chair/to the left of/table | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | chair/to the left of/table | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | chair/to the left of/table | S6 | +0.92 | [+0.75, +1.00] | 0.90 | 20 |
| spatial | test | chair/to the left of/table | S12 | +0.55 | [+0.25, +0.98] | 0.65 | 20 |
| spatial | test | chair/to the left of/table | S18 | +0.00 | [-0.02, +0.07] | 0.10 | 20 |
| spatial | test | man/to the left of/person | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | man/to the left of/person | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | man/to the left of/person | S6 | +0.91 | [+0.66, +1.00] | 0.75 | 20 |
| spatial | test | man/to the left of/person | S12 | +0.40 | [-0.84, +0.81] | 0.55 | 20 |
| spatial | test | man/to the left of/person | S18 | -0.00 | [-0.00, +0.01] | 0.05 | 20 |
| spatial | test | man/to the right of/woman | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | man/to the right of/woman | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | man/to the right of/woman | S6 | +1.02 | [+1.00, +1.09] | 0.90 | 20 |
| spatial | test | man/to the right of/woman | S12 | +0.54 | [+0.35, +0.70] | 0.55 | 20 |
| spatial | test | man/to the right of/woman | S18 | +0.00 | [-0.02, +0.05] | 0.00 | 20 |
| spatial | test | person/to the right of/man | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | person/to the right of/man | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | person/to the right of/man | S6 | +1.02 | [+0.91, +1.26] | 1.00 | 20 |
| spatial | test | person/to the right of/man | S12 | +0.72 | [+0.51, +0.97] | 0.80 | 20 |
| spatial | test | person/to the right of/man | S18 | +0.03 | [-0.03, +0.18] | 0.05 | 20 |
| spatial | test | tree/to the left of/building | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | tree/to the left of/building | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | tree/to the left of/building | S6 | +0.94 | [+0.87, +1.00] | 1.00 | 20 |
| spatial | test | tree/to the left of/building | S12 | +0.72 | [+0.43, +1.00] | 0.65 | 20 |
| spatial | test | tree/to the left of/building | S18 | +0.01 | [-0.00, +0.06] | 0.05 | 20 |
| spatial | test | woman/to the left of/man | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | woman/to the left of/man | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| spatial | test | woman/to the left of/man | S6 | +0.94 | [+0.88, +1.00] | 0.85 | 20 |
| spatial | test | woman/to the left of/man | S12 | +0.54 | [+0.45, +0.62] | 0.60 | 20 |
| spatial | test | woman/to the left of/man | S18 | +0.00 | [-0.00, +0.02] | 0.00 | 20 |
| semantic | dev | building/color/white | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | building/color/white | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | building/color/white | S6 | +0.86 | [+0.70, +1.53] | 0.90 | 10 |
| semantic | dev | building/color/white | S12 | -0.38 | [+0.15, +0.61] | 0.40 | 10 |
| semantic | dev | building/color/white | S18 | -0.06 | [-0.49, +0.04] | 0.00 | 10 |
| semantic | dev | door/color/white | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | door/color/white | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | door/color/white | S6 | +1.17 | [+1.04, +1.42] | 1.00 | 10 |
| semantic | dev | door/color/white | S12 | +0.56 | [+0.05, +0.85] | 0.60 | 10 |
| semantic | dev | door/color/white | S18 | +0.04 | [-0.01, +0.11] | 0.00 | 10 |
| semantic | dev | fence/material/wood | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | fence/material/wood | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | fence/material/wood | S6 | +0.77 | [-0.44, +1.04] | 0.80 | 10 |
| semantic | dev | fence/material/wood | S12 | +0.05 | [-0.04, +0.35] | 0.20 | 10 |
| semantic | dev | fence/material/wood | S18 | -0.01 | [-0.20, +0.00] | 0.10 | 10 |
| semantic | dev | jacket/color/black | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | jacket/color/black | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | jacket/color/black | S6 | +0.93 | [+0.78, +1.11] | 1.00 | 10 |
| semantic | dev | jacket/color/black | S12 | +0.35 | [+0.22, +0.53] | 0.30 | 10 |
| semantic | dev | jacket/color/black | S18 | -0.01 | [-0.17, +0.05] | 0.00 | 10 |
| semantic | dev | table/color/brown | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | table/color/brown | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 10 |
| semantic | dev | table/color/brown | S6 | +0.76 | [+0.25, +0.92] | 0.70 | 10 |
| semantic | dev | table/color/brown | S12 | +0.29 | [-0.20, +1.29] | 0.40 | 10 |
| semantic | dev | table/color/brown | S18 | -0.02 | [-0.14, +0.08] | 0.00 | 10 |
| semantic | test | building/color/white | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | building/color/white | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | building/color/white | S6 | +0.99 | [+0.80, +1.13] | 0.80 | 20 |
| semantic | test | building/color/white | S12 | +0.43 | [+0.12, +0.55] | 0.35 | 20 |
| semantic | test | building/color/white | S18 | +0.01 | [-0.00, +0.10] | 0.10 | 20 |
| semantic | test | door/color/white | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | door/color/white | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | door/color/white | S6 | +0.95 | [+0.86, +1.02] | 1.00 | 20 |
| semantic | test | door/color/white | S12 | +0.27 | [+0.11, +0.57] | 0.30 | 20 |
| semantic | test | door/color/white | S18 | +0.00 | [-0.03, +0.04] | 0.05 | 20 |
| semantic | test | fence/material/wood | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | fence/material/wood | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | fence/material/wood | S6 | +0.98 | [+0.89, +1.14] | 0.95 | 20 |
| semantic | test | fence/material/wood | S12 | +0.20 | [+0.08, +0.30] | 0.15 | 20 |
| semantic | test | fence/material/wood | S18 | -0.00 | [-0.00, +0.00] | 0.05 | 20 |
| semantic | test | jacket/color/black | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | jacket/color/black | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | jacket/color/black | S6 | +0.96 | [+0.85, +1.00] | 0.95 | 20 |
| semantic | test | jacket/color/black | S12 | +0.37 | [+0.09, +0.69] | 0.45 | 20 |
| semantic | test | jacket/color/black | S18 | -0.00 | [-0.05, +0.00] | 0.05 | 20 |
| semantic | test | table/color/brown | S0 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | table/color/brown | S1 | +1.00 | [+1.00, +1.00] | 1.00 | 20 |
| semantic | test | table/color/brown | S6 | +0.97 | [+0.83, +1.14] | 0.90 | 20 |
| semantic | test | table/color/brown | S12 | +0.34 | [+0.25, +0.61] | 0.35 | 20 |
| semantic | test | table/color/brown | S18 | -0.01 | [-0.06, +0.00] | 0.00 | 20 |

## E3 exploratory regional patching (median R)

| group | n | S1 target | S1 ref | S1 irr | S1 random | S6 target | S6 ref | S6 irr | S6 random |
|---|---|---|---|---|---|---|---|---|---|
| woman/to the left of/man | 20 | +0.22 | +0.17 | +0.13 | +0.00 | +0.24 | +0.05 | +0.07 | +0.03 |
| man/to the right of/woman | 20 | +0.12 | +0.17 | +0.01 | +0.00 | +0.11 | +0.24 | +0.05 | +0.04 |
| man/to the left of/person | 20 | -0.00 | +0.12 | +0.00 | +0.15 | -0.11 | +0.13 | +0.00 | +0.01 |
| person/to the right of/man | 20 | +0.17 | +0.20 | +0.17 | -0.00 | +0.17 | +0.13 | +0.12 | +0.00 |
| tree/to the left of/building | 20 | +0.29 | +0.21 | +0.00 | +0.09 | +0.32 | +0.23 | +0.00 | +0.04 |
| chair/to the left of/table | 20 | +0.38 | +0.17 | +0.00 | +0.01 | +0.34 | +0.20 | +0.00 | +0.00 |

## E2 coverage ablation (mean delta m)

| case | region | S1 | S6 |
|---|---|---|---|
| semantic_fence_material_wood | single | -0.18 | -0.12 |
| semantic_fence_material_wood | small | -0.36 | -0.31 |
| semantic_fence_material_wood | object | -1.24 | -1.19 |
| semantic_fence_material_wood | full | -0.80 | -0.86 |
| spatial_chair_to-the-left-of_table | single | -0.04 | -0.02 |
| spatial_chair_to-the-left-of_table | small | -0.15 | -0.09 |
| spatial_chair_to-the-left-of_table | object | -0.30 | -0.20 |
| spatial_chair_to-the-left-of_table | full | +0.13 | +0.44 |

## E3 splitting full-block recovery (median R)

| group | n | blocks | S1 | S6 |
|---|---|---|---|---|
| woman/to the left of/man | 5 | 13.0 | +1.00 | +0.86 |
| man/to the right of/woman | 5 | 13.0 | +1.00 | +0.73 |
| tree/to the left of/building | 4 | 13.0 | +1.00 | +0.83 |
| chair/to the left of/table | 3 | 13.0 | +1.00 | +0.56 |

## E4 linear probe (source vs target)

| group | stage | AUC | accuracy | majority | n train | n test |
|---|---|---|---|---|---|---|
| woman/to the left of/man | S0 | 0.543 | 0.537 | 0.500 | 52 | 80 |
| woman/to the left of/man | S1 | 0.539 | 0.550 | 0.500 | 52 | 80 |
| man/to the right of/woman | S0 | 0.543 | 0.537 | 0.500 | 52 | 80 |
| man/to the right of/woman | S1 | 0.539 | 0.550 | 0.500 | 52 | 80 |
| man/to the left of/person | S0 | 0.528 | 0.475 | 0.500 | 40 | 80 |
| man/to the left of/person | S1 | 0.534 | 0.525 | 0.500 | 40 | 80 |
| person/to the right of/man | S0 | 0.528 | 0.475 | 0.500 | 40 | 80 |
| person/to the right of/man | S1 | 0.534 | 0.525 | 0.500 | 40 | 80 |
| tree/to the left of/building | S0 | 0.444 | 0.425 | 0.500 | 44 | 80 |
| tree/to the left of/building | S1 | 0.452 | 0.475 | 0.500 | 44 | 80 |
| chair/to the left of/table | S0 | 0.490 | 0.463 | 0.500 | 14 | 80 |
| chair/to the left of/table | S1 | 0.472 | 0.463 | 0.500 | 14 | 80 |

## E4 stage-wise region probe with position baseline

| group | stage | AUC | 95% CI | random-label AUC | n train | n test |
|---|---|---|---|---|---|---|
| woman/to the left of/man | S0 | 0.79 | [0.69,0.89] | 0.50 | 52 | 80 |
| woman/to the left of/man | S1 | 0.87 | [0.79,0.94] | 0.47 | 52 | 80 |
| woman/to the left of/man | S6 | 0.89 | [0.81,0.95] | 0.48 | 52 | 80 |
| woman/to the left of/man | S12 | 0.89 | [0.81,0.95] | 0.48 | 52 | 80 |
| woman/to the left of/man | position | 0.99 | [0.98,1.00] | 0.38 | 52 | 80 |
| man/to the right of/woman | S0 | 0.79 | [0.69,0.89] | 0.50 | 52 | 80 |
| man/to the right of/woman | S1 | 0.87 | [0.79,0.94] | 0.47 | 52 | 80 |
| man/to the right of/woman | S6 | 0.89 | [0.81,0.95] | 0.48 | 52 | 80 |
| man/to the right of/woman | S12 | 0.89 | [0.81,0.95] | 0.48 | 52 | 80 |
| man/to the right of/woman | position | 0.99 | [0.98,1.00] | 0.38 | 52 | 80 |
| man/to the left of/person | S0 | 0.66 | [0.53,0.78] | 0.48 | 40 | 80 |
| man/to the left of/person | S1 | 0.74 | [0.62,0.85] | 0.46 | 40 | 80 |
| man/to the left of/person | S6 | 0.74 | [0.62,0.86] | 0.47 | 40 | 80 |
| man/to the left of/person | S12 | 0.75 | [0.64,0.86] | 0.44 | 40 | 80 |
| man/to the left of/person | position | 0.98 | [0.95,1.00] | 0.39 | 40 | 80 |
| person/to the right of/man | S0 | 0.66 | [0.53,0.78] | 0.48 | 40 | 80 |
| person/to the right of/man | S1 | 0.74 | [0.62,0.85] | 0.46 | 40 | 80 |
| person/to the right of/man | S6 | 0.74 | [0.62,0.86] | 0.47 | 40 | 80 |
| person/to the right of/man | S12 | 0.75 | [0.64,0.86] | 0.44 | 40 | 80 |
| person/to the right of/man | position | 0.98 | [0.95,1.00] | 0.39 | 40 | 80 |
| tree/to the left of/building | S0 | 0.71 | [0.59,0.82] | 0.56 | 44 | 80 |
| tree/to the left of/building | S1 | 0.83 | [0.74,0.91] | 0.57 | 44 | 80 |
| tree/to the left of/building | S6 | 0.86 | [0.78,0.93] | 0.57 | 44 | 80 |
| tree/to the left of/building | S12 | 0.87 | [0.80,0.94] | 0.57 | 44 | 80 |
| tree/to the left of/building | position | 0.99 | [0.98,1.00] | 0.62 | 44 | 80 |
| chair/to the left of/table | S0 | 0.66 | [0.54,0.77] | 0.48 | 14 | 80 |
| chair/to the left of/table | S1 | 0.72 | [0.61,0.83] | 0.49 | 14 | 80 |
| chair/to the left of/table | S6 | 0.73 | [0.61,0.83] | 0.50 | 14 | 80 |
| chair/to the left of/table | S12 | 0.76 | [0.66,0.86] | 0.51 | 14 | 80 |
| chair/to the left of/table | position | 0.86 | [0.77,0.92] | 0.51 | 14 | 80 |

## E4 question-context target specificity

| group | stage | own question | control question |
|---|---|---|---|
| woman/to the left of/man | S0 | -1.16 | +0.16 |
| woman/to the left of/man | S1 | -1.10 | +0.24 |
| woman/to the left of/man | S6 | -1.02 | +0.16 |
| man/to the right of/woman | S0 | -0.29 | -0.08 |
| man/to the right of/woman | S1 | -0.28 | -0.16 |
| man/to the right of/woman | S6 | -0.33 | -0.20 |
| man/to the left of/person | S0 | -0.05 | +0.29 |
| man/to the left of/person | S1 | +0.01 | +0.24 |
| man/to the left of/person | S6 | -0.12 | +0.15 |
| person/to the right of/man | S0 | -0.25 | -0.36 |
| person/to the right of/man | S1 | -0.21 | -0.29 |
| person/to the right of/man | S6 | -0.25 | -0.21 |

## Minimal same-image flip counterfactual

| group | n | clean AUC | delta | AUC after target abl S1 | S6 | target patch R S1 | S6 | random patch R S1 | S6 |
|---|---|---|---|---|---|---|---|---|---|
| chair/to the left of/table | 10 | 0.60 | +0.15 | 0.50 | 0.60 | +0.01 | +0.23 | -0.00 | -0.00 |
| woman/to the left of/man | 10 | 0.70 | +0.83 | 0.80 | 0.70 | +0.46 | +0.47 | -0.00 | +0.00 |
| man/to the left of/person | 10 | 0.50 | +0.05 | 0.70 | 0.60 | +0.38 | +0.26 | -0.36 | +0.00 |

## E4 joint target+reference ablation (mean delta m)

| group | stage | target | reference | joint | full |
|---|---|---|---|---|---|
| woman/to the left of/man | S1 | -0.79 | -0.47 | -0.62 | -1.16 |
| woman/to the left of/man | S6 | -0.77 | -0.54 | -0.79 | -1.01 |
| woman/to the left of/man | S12 | -0.27 | -0.26 | -0.31 | -0.14 |
| man/to the right of/woman | S1 | -0.54 | -0.10 | -0.37 | -0.12 |
| man/to the right of/woman | S6 | -0.66 | -0.16 | -0.65 | -0.25 |
| man/to the right of/woman | S12 | -0.31 | -0.07 | -0.26 | +0.34 |
| chair/to the left of/table | S1 | -0.30 | +0.03 | -0.08 | +0.13 |
| chair/to the left of/table | S6 | -0.20 | +0.26 | +0.23 | +0.44 |
| chair/to the left of/table | S12 | -0.24 | +0.14 | +0.00 | +0.20 |

## E3 semantic exploratory regional patching (median R)

| group | n | S1 target | S1 irr | S1 random | S6 target | S6 irr | S6 random |
|---|---|---|---|---|---|---|---|
| fence/material/wood | 20 | +0.66 | +0.11 | +0.06 | +0.43 | +0.10 | +0.05 |
| table/color/brown | 20 | +0.48 | +0.31 | +0.07 | +0.63 | +0.36 | +0.04 |
| jacket/color/black | 20 | +0.11 | +0.00 | +0.05 | +0.11 | +0.09 | +0.02 |
| building/color/white | 20 | +0.51 | +0.00 | +0.02 | +0.53 | +0.00 | -0.00 |
| door/color/white | 20 | +0.16 | +0.12 | +0.07 | +0.16 | +0.17 | +0.11 |
