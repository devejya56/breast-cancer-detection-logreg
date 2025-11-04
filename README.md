# Breast Cancer Detection using Logistic Regression

## Introduction

This project implements a machine learning solution for breast cancer detection using logistic regression. The model classifies tumors as malignant or benign based on various features extracted from cell nuclei present in breast mass images. This implementation demonstrates the effectiveness of logistic regression in medical diagnosis applications.

## Dataset

This project uses the **Wisconsin Breast Cancer Dataset**, a widely recognized dataset in machine learning and medical diagnosis research. The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses, describing characteristics of cell nuclei present in the images.

**Dataset Characteristics:**
- 569 instances
- 30 numeric features
- Binary classification (Malignant/Benign)
- No missing values
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension

## Motivation

Breast cancer is one of the most common cancers affecting women worldwide. Early detection significantly improves treatment outcomes and survival rates. This project aims to:

- Demonstrate how machine learning can assist in medical diagnosis
- Provide an accessible implementation of logistic regression for classification
- Achieve high accuracy in distinguishing between malignant and benign tumors
- Serve as an educational resource for students and practitioners learning machine learning

## Project Workflow

1. **Data Loading**: Import the Wisconsin Breast Cancer Dataset
2. **Exploratory Data Analysis**: Understand data distribution and feature relationships
3. **Data Preprocessing**: Handle missing values, normalize/standardize features
4. **Train-Test Split**: Divide data into training and testing sets
5. **Model Training**: Train logistic regression classifier
6. **Model Evaluation**: Assess performance using accuracy, precision, recall, F1-score, and confusion matrix
7. **Prediction**: Make predictions on new data
8. **Visualization**: Plot results and performance metrics

## Repository Structure

```
breast-cancer-detection-logreg/
│
├── README.md                          # Project documentation
├── breast_cancer_detection.ipynb      # Main Jupyter notebook
├── breast_cancer_detection.py         # Python script version
├── data/                              # Dataset directory
│   └── wisconsin_breast_cancer.csv
├── models/                            # Saved models
│   └── logistic_regression_model.pkl
├── results/                           # Output visualizations and reports
│   ├── confusion_matrix.png
│   └── performance_metrics.txt
└── requirements.txt                   # Project dependencies
```

## Quickstart

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/devejya56/breast-cancer-detection-logreg.git
cd breast-cancer-detection-logreg
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Running the Project

**Option 1: Jupyter Notebook**
```bash
jupyter notebook breast_cancer_detection.ipynb
```

**Option 2: Python Script**
```bash
python breast_cancer_detection.py
```

### Expected Output

- Model accuracy and performance metrics
- Confusion matrix visualization
- Classification report with precision, recall, and F1-scores
- Saved model file for future predictions

## Model Details

### Algorithm: Logistic Regression

Logistic regression is a statistical method for binary classification that models the probability of an instance belonging to a particular class.

**Key Parameters:**
- Solver: liblinear or lbfgs
- Regularization: L1 or L2 (to prevent overfitting)
- Max iterations: 1000 (default)

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visualization of true positives, true negatives, false positives, and false negatives

### Expected Results

Typical performance on the Wisconsin dataset:
- Accuracy: ~95-97%
- Precision: ~94-96%
- Recall: ~93-96%
- F1-Score: ~94-96%

## Notes and Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed using `pip install -r requirements.txt`
2. **Data Not Found**: Verify the dataset file path in the script matches your directory structure
3. **Low Accuracy**: Check data preprocessing steps, feature scaling, and model parameters
4. **Convergence Warnings**: Increase max_iter parameter or try different solvers

### Tips

- **Feature Scaling**: Standardization significantly improves logistic regression performance
- **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
- **Feature Selection**: Experiment with different feature combinations to optimize performance
- **Threshold Tuning**: Adjust classification threshold based on whether you want to minimize false negatives or false positives

### Dependencies

Key libraries used:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- jupyter (optional, for notebook)

## Extending the Project

### Potential Enhancements

1. **Compare Multiple Algorithms**: Implement and compare with SVM, Random Forest, Neural Networks
2. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV for optimal parameters
3. **Feature Engineering**: Create new features from existing ones
4. **Model Interpretation**: Add SHAP or LIME for explainability
5. **Web Application**: Deploy as a Flask or Streamlit app for interactive predictions
6. **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
7. **ROC Curve Analysis**: Add ROC-AUC analysis for threshold selection
8. **Ensemble Methods**: Combine multiple models for improved performance

### Research Directions

- Investigate deep learning approaches (CNNs, LSTMs)
- Explore transfer learning with pre-trained models
- Implement ensemble methods for improved accuracy
- Add real-time prediction capabilities

## Author & Credits

**Author**: devejya56

**Dataset Source**: 
- UCI Machine Learning Repository - Wisconsin Breast Cancer Dataset
- Original creators: Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian

**Acknowledgments**:
- Wisconsin Breast Cancer Dataset contributors
- Scikit-learn library developers
- Open-source machine learning community

**License**: MIT License (or specify your chosen license)

**Contact**: For questions, issues, or contributions, please open an issue on GitHub.

---

*This project is for educational purposes and should not be used as a substitute for professional medical diagnosis.*
