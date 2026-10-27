# Breast Cancer Wisconsin (Original) Dataset

The dataset is stored in [`breast-cancer-wisconsin-original.csv`](breast-cancer-wisconsin-original.csv).

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)
- **Rows:** 699
- **Features:** 9, plus sample ID and target class
- **Target:** `class` — `2` for benign, `4` for malignant
- **Missing values:** `?` in `bare_nuclei`

## Columns

| Column | Description |
| --- | --- |
| `sample_code_number` | Sample identifier |
| `clump_thickness` | Clump thickness (1–10) |
| `uniformity_of_cell_size` | Uniformity of cell size (1–10) |
| `uniformity_of_cell_shape` | Uniformity of cell shape (1–10) |
| `marginal_adhesion` | Marginal adhesion (1–10) |
| `single_epithelial_cell_size` | Single epithelial cell size (1–10) |
| `bare_nuclei` | Bare nuclei (1–10; `?` when missing) |
| `bland_chromatin` | Bland chromatin (1–10) |
| `normal_nucleoli` | Normal nucleoli (1–10) |
| `mitoses` | Mitoses (1–10) |
| `class` | Diagnosis class: `2` benign or `4` malignant |
