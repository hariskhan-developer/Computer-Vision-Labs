# Model Evaluation & Performance Benchmark Results

This repository contains benchmarking results and performance evaluations for various machine learning and deep learning classification models. Below are the summary tables generated from the evaluation pipeline [cite: 1].

## Table 1: Classification Models Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.872879 | 0.779234 | 0.637286 | 0.699770 | 0.888218 | 1.454194 | 0.034384 |
| **Decision Tree** | 0.856865 | 0.700512 | 0.623251 | 0.678684 | 0.707956 | 2.751669 | 0.028889 |
| **Random Forest** | 0.825558 | 0.763523 | 0.707469 | 0.748432 | 0.705204 | 2.200465 | 0.087474 |
| **K-Nearest Neighbors** | 0.755053 | 0.670995 | 0.712718 | 0.795040 | 0.713440 | 2.896109 | 0.097367 |
| **Linear SVM** | 0.827238 | 0.786824 | 0.684664 | 0.764316 | 0.796974 | 3.369455 | 0.014425 |
| **XGBoost** | 0.718587 | 0.605646 | 0.672959 | 0.698771 | 0.853946 | 2.831834 | 0.046473 |

---

## Table 2: Additional System & Complexity Metrics

| Category | Metric Value A | Metric Value B | Metric Value C |
| :--- | :---: | :---: | :---: |
| **Model Size (MB)** | 95.842696 | 4.378597 | 0.146289 |
| **FLOPs (Giga)** | 95.697309 | 6.929667 | 0.413974 |
| **Memory (MB)** | 51.700712 | 9.715791 | 0.424574 |
| **Latency (ms)** | 106.871141 | 6.653294 | 0.187037 |

---

## Table 3: Final Comparison & Evaluation Metrics

| Comparison Category | Performance Score A | Performance Score B | Error Rate C |
| :--- | :---: | :---: | :---: |
| **Overall Performance** | 38.130679 | 2.995200 | 0.043096 |
| **Robustness** | 85.050273 | 3.132155 | 0.025715 |
| **Scalability** | 75.016033 | 9.339624 | 0.075196 |
| **Energy Efficiency** | 24.809783 | 5.406934 | 0.069178 |

---

## References
* Source Notebook and Data Generation [cite: 1]
