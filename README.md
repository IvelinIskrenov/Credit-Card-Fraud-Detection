Isolation Forest 
This file implements the unsupervised anomaly detection algorithm:
Anomaly Detection: Builds an ensemble of randomized decision trees (IsolationTree). Anomalies are isolated closer to the root of the tree and thus have shorter path lengths.
Hyperparameters: Configurable constants:
 - TREE_COUNT: The number of trees in the forest.
 - MAX_DEPTH: The maximum depth a tree can grow to.

Feature Selection: Includes a featureSelection function using a variance threshold to filter out low-variance features before training, which can be useful for reducing noise and training time.

Cross-Validation: Implements Stratified k-Fold Cross-Validation (specifically 10-fold in the main loop) to provide a robust estimate of model performance across the dataset. The user is prompted to run this test at runtime.


Logistic Regression 
This implementation is a classic logistic regression model:
Core Logic: Uses Gradient Descent to minimize the cost (Negative Log-Likelihood) and update weights and bias.
Data Preprocessing: Implements Standardization (Z-score normalization) to normalize feature attributes before training.
Evaluation: Focuses on imbalance metrics: Recall, Precision, and F1-Score.
Feature Sparsity (Potential): The code contains commented-out logic for LASSO (L1) Regularization, which can be uncommented and configured to induce sparsity in the weights, potentially helping with feature selection
