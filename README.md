# Machine Learning 📊

A collection of machine learning experiments, notebooks, and practice projects.

## 📁 Contents

| File | Description |
|------|-------------|
| `LinearRegression/LinearRegression.ipynb` | Linear regression — implementation, training, and evaluation. |
| `LogisticRegression/LogisticRegression.ipynb` | Logistic regression for binary classification. |
| `TextClassification/TextClassfication.ipynb` | **Yelp review sentiment classification** — TF-IDF + Logistic Regression baseline vs. fine-tuned DistilBERT, with EDA, confusion matrices, and an inference demo. |
| `ReinforcementLearning/ReinforcementLearning.ipynb` | **Q-learning on FrozenLake** — tabular RL from scratch: ε-greedy exploration, Bellman updates, training curves, policy visualization, stochastic vs. deterministic comparison. |
| `ReinforcementLearning/DQN_CartPole.ipynb` | **Deep Q-Network on CartPole** — scaling RL to continuous states: Q-network, experience replay, target network, training curves, step-by-step episode replay. |
| `BreastCancerDetection/BreastCancerDetection.ipynb` | **Breast cancer detection in histopathology images** — PyTorch CNN for benign vs. malignant (IDC) patch classification: augmentation, training curves, ROC/AUC, confusion matrix, inference demo, Grad-CAM interpretability. Runs on real IDC/BreaKHis data or a built-in synthetic fallback dataset. |

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook / JupyterLab (or VS Code with the Jupyter extension)
- Common ML libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch torchvision transformers accelerate gymnasium jupyter
```

### Running the Notebooks

```bash
jupyter notebook
```

Then open any `.ipynb` file from the browser interface and run the cells (Shift + Enter).

## 📚 Topics Covered

- **Supervised Learning**
  - Linear Regression
  - Logistic Regression
- **NLP**
  - Text Classification (TF-IDF baseline + DistilBERT fine-tuning)
- **Computer Vision / Deep Learning**
  - Breast Cancer Detection (CNN on histopathology images, Grad-CAM)
- **Reinforcement Learning**
  - Q-Learning (tabular, model-free)
  - Deep Q-Network (function approximation, experience replay, target network)

## 📝 Notes

- Each notebook is self-contained — datasets are either generated inline or loaded from standard sources.
- Plots and evaluation metrics are included inline for quick interpretation.

---

*Last updated: October 2026*
