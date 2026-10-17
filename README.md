# Machine Learning 📊

A collection of machine learning experiments, notebooks, and practice projects.

## 📁 Contents

| File | Description |
|------|-------------|
| `LinearRegression/LinearRegression.ipynb` | Linear regression — implementation, training, and evaluation. |
| `LogisticRegression/LogisticRegression.ipynb` | Logistic regression for binary classification. |
| `TextClassification/TextClassfication.ipynb` | **Yelp review sentiment classification** — TF-IDF + Logistic Regression baseline vs. fine-tuned DistilBERT, with EDA, confusion matrices, and an inference demo. |
| `ReinforcementLearning/ReinforcementLearning.ipynb` | **Q-learning on FrozenLake** — tabular RL from scratch: ε-greedy exploration, Bellman updates, training curves, policy visualization, stochastic vs. deterministic comparison. |

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook / JupyterLab (or VS Code with the Jupyter extension)
- Common ML libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch transformers accelerate gymnasium jupyter
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
- **Reinforcement Learning**
  - Q-Learning (tabular, model-free)

## 📝 Notes

- Each notebook is self-contained — datasets are either generated inline or loaded from standard sources.
- Plots and evaluation metrics are included inline for quick interpretation.

---

*Last updated: October 2026*
