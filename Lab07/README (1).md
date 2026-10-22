# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

## Objective

Modify the traditional Histogram of Oriented Gradients (HOG) feature-extraction pipeline by adding a Genetic Algorithm (GA) for feature selection, and evaluate whether the modified approach improves classification performance on a medical image dataset.

**Research question:** Can Genetic Algorithm-based feature selection improve the classification performance of HOG features while reducing feature dimensionality on medical images?

## Folder Structure

```
Lab07/
└── GA-HOG-Feature-Selection/
    ├── assets/
    ├── GA_HOG_Feature_Selection.ipynb
    └── README.md
```

## Dataset

- **Dataset:** BloodMNIST (MedMNIST+, 128 x 128 version), 8 blood cell classes: basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil, platelet.
- **Download link:** https://zenodo.org/records/10519652/files/bloodmnist_128.npz
- **References:** Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification", Scientific Data, 2023. Original blood cell images from Acevedo et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems", Data in Brief, 2020.
- **Split:** the fixed MedMNIST split, used identically for every method: 11,959 train / 1,712 validation / 3,421 test.

## Methodology

1. **Preprocessing:** images converted to grayscale, split used as provided.
2. **HOG extraction:** 9 orientations, 8 x 8 pixels per cell, 2 x 2 cells per block, L2-Hys normalization. This gives 15 x 15 blocks x 36 values = **8100 features** per image. Features are standardized (scaler fitted on the training set only).
3. **Baseline:** HOG + SVM (RBF kernel) using all 8100 features. C in {1, 10, 100} and gamma multiplier in {0.5, 1, 2} (relative to the "scale" gamma) were tuned on the validation set.
4. **Genetic Algorithm:** each individual is a binary mask over the 8100 features.
   - Population initialization: random masks with per-individual density between 10% and 60%
   - Selection: tournament selection (size 3)
   - Crossover: uniform crossover, probability 0.8
   - Mutation: per-bit flip
   - Elitism: best 2 individuals kept each generation
   - **Fitness** = validation macro-F1 - 0.1 x (selected features / total features)
   - Fitness is evaluated with an SVM trained on a stratified 3000-image subset of the training set and scored on the full validation set, to keep the search affordable.
5. **GA settings compared:** GA-1 (population 20, mutation 0.005) and GA-2 (population 30, mutation 0.02), 20 generations each. The best setting was chosen by validation fitness only.
6. **GA-HOG + SVM:** SVM trained on the full training set using only the selected features, with the same tuning grid as the baseline.
7. **Final evaluation:** the test set was used only once, after model selection was complete.
8. **Implementation note:** HOG is computed with OpenCV. The RBF kernel matrix is computed on the GPU and passed to scikit-learn's `SVC(kernel="precomputed")`, which is mathematically the same as a normal RBF SVM with the same gamma.

---

## Results

### Table 1. Experimental configuration and HOG feature-extraction parameters

| Parameter                   | Baseline: HOG + SVM                                      | Proposed: GA-HOG + SVM                                   |
|:----------------------------|:---------------------------------------------------------|:---------------------------------------------------------|
| Medical dataset             | BloodMNIST (MedMNIST+, 128x128)                          | BloodMNIST (MedMNIST+, 128x128)                          |
| Image resolution            | 128 x 128                                                | 128 x 128                                                |
| HOG orientations            | 9                                                        | 9                                                        |
| Pixels per cell             | 8 x 8                                                    | 8 x 8                                                    |
| Cells per block             | 2 x 2                                                    | 2 x 2                                                    |
| Block normalization         | L2-Hys                                                   | L2-Hys                                                   |
| HOG feature dimension       | 8100                                                     | 8100                                                     |
| Feature selection           | None                                                     | Genetic Algorithm                                        |
| Features used by classifier | 8100                                                     | 2525                                                     |
| HOG implementation          | OpenCV HOGDescriptor                                     | OpenCV HOGDescriptor                                     |
| Classifier                  | SVM                                                      | SVM                                                      |
| SVM kernel                  | RBF (precomputed)                                        | RBF (precomputed)                                        |
| Hyperparameter tuning       | Validation set (C [1, 10, 100], gamma x [0.5, 1.0, 2.0]) | Validation set (C [1, 10, 100], gamma x [0.5, 1.0, 2.0]) |
| Selected C                  | 10.0                                                     | 10.0                                                     |
| Selected gamma multiplier   | 1.0                                                      | 1.0                                                      |
| Data split                  | Fixed (11959 / 1712 / 3421)                              | Fixed (11959 / 1712 / 3421)                              |
| Random seed                 | 42                                                       | 42                                                       |

### Dataset overview (Tasks 1 and 2)

![Class distribution](assets/class_distribution.png)

![Dataset samples](assets/dataset_samples.png)

![Preprocessing example](assets/preprocessing_example.png)

### HOG feature extraction (Task 4)

![HOG visualization](assets/hog_visualization.png)

### Table 2. Main performance comparison on the test set: without GA versus with GA

| Metric                           | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Improvement |
|:---------------------------------|---------------------:|------------------------:|------------:|
| Accuracy (%)                     |              81.1166 |                 79.3335 |     -1.7831 |
| Precision (Macro, %)             |              81.7464 |                 80.6543 |     -1.0921 |
| Recall (Macro, %)                |              79.1591 |                 77.6554 |     -1.5038 |
| Macro-F1 (%)                     |              80.2297 |                 78.8666 |     -1.3631 |
| Weighted-F1 (%)                  |              80.9905 |                 79.2244 |     -1.7661 |
| Balanced accuracy (%)            |              79.1591 |                 77.6554 |     -1.5038 |
| Specificity (Macro, %)           |              97.2092 |                 96.9345 |     -0.2747 |
| Cohen's kappa                    |               0.7785 |                  0.7575 |      -0.021 |
| Matthews correlation coefficient |               0.7789 |                   0.758 |     -0.0209 |

Improvement = GA-HOG minus baseline. A negative value means the GA-HOG model scored lower. Specificity is the macro average of one-vs-rest specificity over the 8 classes.

![Metric comparison](assets/metric_comparison.png)

### Table 3. Feature count and computational cost

| Item                        | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) |
|:----------------------------|---------------------:|------------------------:|
| Feature dimension           |                 8100 |                    2525 |
| Feature reduction (%)       |                    0 |                  68.827 |
| HOG extraction time (s)     |                7.365 |                   7.365 |
| GA selection time (s)       |                    0 |                 114.984 |
| SVM tuning time (s)         |                31.16 |                    30.2 |
| Final SVM training time (s) |                4.123 |                   3.665 |
| Test inference time (s)     |                0.978 |                   0.741 |
| Total pipeline time (s)     |               42.648 |                 156.213 |

![Feature count comparison](assets/feature_count_comparison.png)

### Table 4. GA settings comparison

| Setting | Population | Generations | Mutation prob. | Crossover prob. | Best fitness | Val Macro-F1 (GA subset) | Selected features | Reduction (%) | GA time (s) |
|:--------|-----------:|------------:|---------------:|----------------:|-------------:|-------------------------:|------------------:|--------------:|------------:|
| GA-1    |         20 |          20 |          0.005 |             0.8 |       0.7249 |                    0.756 |              2525 |       68.8272 |     114.984 |
| GA-2    |         30 |          20 |           0.02 |             0.8 |       0.7171 |                    0.755 |              3071 |       62.0864 |     190.372 |

GA-1 had the higher validation fitness, so it was selected as the final GA setting.

### GA fitness across generations (Task 9)

![GA fitness curves](assets/ga_fitness_curves.png)

The best fitness increased over the generations for both settings (GA-1 ended at 0.7249, GA-2 at 0.7171). The validation macro-F1 of the best individual also rose, and the number of selected features of the best individual stayed in the range of roughly 2100 to 3200.

### Confusion matrices and per-class results

![Confusion matrices](assets/confusion_matrices.png)

| Class                | Baseline F1 | GA-HOG F1 | Support | Difference |
|:---------------------|------------:|----------:|--------:|-----------:|
| basophil             |      0.7366 |    0.7483 |     244 |     0.0117 |
| eosinophil           |      0.7951 |    0.7559 |     624 |    -0.0392 |
| erythroblast         |      0.8624 |    0.8548 |     311 |    -0.0076 |
| immature granulocyte |      0.6922 |    0.6783 |     579 |    -0.0139 |
| lymphocyte           |      0.8487 |    0.8468 |     243 |    -0.0019 |
| monocyte             |       0.642 |     0.616 |     284 |     -0.026 |
| neutrophil           |      0.8767 |    0.8488 |     666 |    -0.0279 |
| platelet             |      0.9646 |    0.9604 |     470 |    -0.0042 |

![Per-class F1](assets/per_class_f1.png)

GA-HOG was slightly better only for basophil. It was lower or equal for the other seven classes, with the largest drops on eosinophil (-0.039), neutrophil (-0.028) and monocyte (-0.026).

---

## Bonus Tasks

### Bonus 1. GA vs random selection vs ANOVA F-test selection (same number of features)

| Metric                           | All HOG (8100) | Random (2525) | ANOVA F-test (2525) | GA (2525) |
|:---------------------------------|---------------:|--------------:|--------------------:|----------:|
| Accuracy (%)                     |        81.1166 |       79.1581 |             80.3566 |   79.3335 |
| Precision (Macro, %)             |        81.7464 |       80.1642 |             80.5025 |   80.6543 |
| Recall (Macro, %)                |        79.1591 |       77.3106 |             78.1846 |   77.6554 |
| Macro-F1 (%)                     |        80.2297 |       78.4382 |             79.0676 |   78.8666 |
| Weighted-F1 (%)                  |        80.9905 |       79.0292 |             80.1124 |   79.2244 |
| Balanced accuracy (%)            |        79.1591 |       77.3106 |             78.1846 |   77.6554 |
| Specificity (Macro, %)           |        97.2092 |        96.913 |             97.1017 |   96.9345 |
| Cohen's kappa                    |         0.7785 |        0.7554 |              0.7696 |    0.7575 |
| Matthews correlation coefficient |         0.7789 |         0.756 |              0.7701 |     0.758 |

![Selection methods comparison](assets/bonus_selection_methods.png)

With the same 2525 features, the GA (78.87% macro-F1) is slightly better than random selection (78.44%) but slightly worse than the simple ANOVA F-test filter (79.07%). All three reduced sets stay below the full HOG feature set (80.23%).

### Bonus 2. Spatial map of selected HOG features

![Spatial heatmap](assets/bonus_spatial_heatmap.png)

The ANOVA F-test keeps features mainly from the central blocks and drops most of the border blocks. The GA selection is spread almost evenly over the whole image and looks close to the random selection, so within 20 generations the GA did not discover a clear spatial pattern.

### Bonus 3. Bootstrap confidence interval of the macro-F1 difference

![Bootstrap CI](assets/bonus_bootstrap_ci.png)

The bootstrap distribution (1000 resamples of the test set) of the macro-F1 difference (GA-HOG minus baseline) is centred at about -1.4 percentage points. The 95% interval is roughly from -2.3 to -0.4 and stays below zero, so the small drop of GA-HOG is unlikely to be only test-set noise.

### Bonus 4. Selected feature indices

The indices of the 2525 features chosen by the best GA individual are saved in `assets/ga_selected_feature_indices.npy`.

---

## Discussion (Task 10)

### Did GA-based feature selection improve classification performance?

No. The GA reduced the HOG dimension from 8100 to 2525 features (68.8% fewer), but test performance dropped slightly: macro-F1 went from 80.23% to 78.87% (-1.36 points), accuracy from 81.12% to 79.33% and balanced accuracy from 79.16% to 77.66%. The bootstrap interval of the difference is below zero, so the drop is small but consistent. The answer to the research question is therefore: **the GA clearly reduced dimensionality, but it did not improve performance on this dataset and setting.** It kept most of the accuracy (about 98% of the baseline macro-F1) with less than a third of the features.

### Strengths

- Large reduction in dimensionality (68.8%) with only a small loss in accuracy.
- Slightly faster final SVM training (4.12 s to 3.67 s) and inference (0.98 s to 0.74 s), and a smaller model.
- The fitness function directly balances accuracy and feature count, and is easy to change (for example the penalty weight).
- Selected indices can be reused and analysed, and the GA beat random selection of the same size.

### Limitations

- No accuracy gain: the baseline with all features and tuned C and gamma was better on every overall metric.
- The search space is huge (2^8100) and only 20 generations with small populations were run, so the GA is far from converged. The spatial map of GA selection is nearly uniform, which suggests it has not found a strong structure yet.
- A simple ANOVA F-test filter, which is almost free to compute, gave a slightly better result than the GA with the same number of features.
- The fitness was computed on a 3000-image subset of the training set to keep the cost low, so the selected mask may be tuned to that subset and the validation set.
- Results come from a single random seed and one dataset split.

### Computational cost

- Total pipeline time went from 42.6 s for the baseline to 156.2 s for GA-HOG (about 3.7 times more), almost all of it from the GA search (115 s for GA-1, 190 s for GA-2).
- This cost is paid once, and the savings in SVM training and inference time are small here because the kernel is computed on the GPU.
- The GA needs hundreds of SVM fits, so its cost grows quickly with population size and generations. A larger population (GA-2, 30 individuals) cost about 65% more time and did not give a better fitness than GA-1.
- Without GPU acceleration, the GA search would take much longer.

### GA settings comparison

GA-1 (population 20, mutation 0.005) reached a higher fitness (0.7249) and a stronger reduction (68.8%) than GA-2 (population 30, mutation 0.02, fitness 0.7171, 62.1% reduction), in less time. The lower mutation rate possibly suits this large binary search space better, since a high mutation rate keeps flipping bits of good masks, but this was not tested separately.

---

## Conclusion

The GA-based feature selection reduced the HOG features by 68.8% and kept performance close to the full-feature baseline (macro-F1 78.87% vs 80.23%), but it did not improve it. On this dataset, a tuned HOG + SVM using all features remains the most accurate, while GA-HOG is useful only when a smaller feature set is a priority. Longer GA runs, a larger fitness subset or a seeded initial population (for example from the ANOVA ranking) are possible ways to improve it further.

## How to Run

1. Open `GA_HOG_Feature_Selection.ipynb` in Google Colab with a GPU runtime (T4).
2. Run all cells from top to bottom. The dataset is downloaded automatically from the Zenodo link above.
3. Figures and tables are saved next to the notebook.
