# 🌧️ Rainfall Prediction Classifier

Welcome to the **Rainfall Prediction Classifier** project! 🌦️ This project is part of the **IBM AI Engineering Professional Certificate** on Coursera, specifically the final project for the first course. It focuses on predicting whether it will rain today in the Melbourne area using historical weather data and machine learning classification techniques. The dataset is sourced from the **Australian Government's Bureau of Meteorology**, covering weather observations from 2008 to 2017.

## 🎯 Objectives

- 📊 **Explore and preprocess the rainfall dataset**: Perform feature engineering, handle missing values, and create new features to prepare high-quality input for modeling.
- 🤖 **Build and optimize a classifier pipeline**: Implement machine learning models using scikit-learn pipelines and optimize them with grid search cross-validation.
- 📈 **Evaluate model performance**: Assess models using various metrics and visualizations, such as accuracy, F1-score, and confusion matrices, to understand their effectiveness.
- 🔄 **Compare multiple classifiers**: Implement and compare different classification algorithms to identify the best-performing model for rainfall prediction.

## 🛠️ Algorithms Used

This project implements and compares two classification algorithms:

1. **Random Forest Classifier** 🌳
   - A robust ensemble method that combines multiple decision trees to predict whether it will rain today.
   - Optimized using grid search with parameters: `n_estimators` ([50, 100]), `max_depth` ([None, 10, 20]), and `min_samples_split` ([2, 5]).
2. **Logistic Regression** 🧑‍💼
   - A linear model used for binary classification, suitable for handling imbalanced datasets.
   - Optimized with parameters: `solver` (['liblinear']), `penalty` (['l1', 'l2']), and `class_weight` ([None, 'balanced']).

Additional algorithms mentioned in the original README (Linear Regression, K-Nearest Neighbors, Support Vector Machines) were not implemented in the provided project document but could be explored in future iterations.

## 📊 Evaluation Metrics

The models are evaluated using the following metrics to assess their performance in predicting rainfall:

- ✅ **Accuracy Score**: Measures the proportion of correct predictions (Random Forest: 0.85, Logistic Regression: 0.83).
- ⚖️ **Precision, Recall, and F1-Score**: Evaluates model performance for each class (No/Yes for rainfall).
  - Random Forest: Precision (No: 0.86, Yes: 0.76), Recall (No: 0.95, Yes: 0.52), F1-Score (No: 0.90, Yes: 0.62).
  - Logistic Regression: Precision (No: 0.86, Yes: 0.68), Recall (No: 0.93, Yes: 0.51), F1-Score (No: 0.89, Yes: 0.58).
- 📊 **Confusion Matrix**: Visualizes true positives, true negatives, false positives, and false negatives to assess prediction errors.
- 🌟 **Feature Importance**: Identifies key features influencing predictions (e.g., `Humidity3pm` was the most important for Random Forest).

## 📝 Dataset Description

The dataset, sourced from the [Australian Government's Bureau of Meteorology](http://www.bom.gov.au/climate/dwo/) via [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/), contains daily weather observations from 2008 to 2017 for multiple Australian locations. This project focuses on a subset of data from Melbourne, Melbourne Airport, and Watsonia (7557 records after preprocessing).

### Key Features
- **Numerical Features**: MinTemp, MaxTemp, Rainfall, Evaporation, Sunshine, WindGustSpeed, WindSpeed9am, WindSpeed3pm, Humidity9am, Humidity3pm, Pressure9am, Pressure3pm, Cloud9am, Cloud3pm, Temp9am, Temp3pm (all float64).
- **Categorical Features**: Location, WindGustDir, WindDir9am, WindDir3pm, RainYesterday, Season (all object).
- **Target Variable**: `RainToday` (Yes/No, indicating if rainfall ≥ 1mm occurred today).
- **Engineered Feature**: `Season` (Winter, Spring, Summer, Autumn), derived from the `Date` column.

### Preprocessing Steps
1. **Data Cleaning**: Dropped rows with missing values, reducing the dataset from 145,460 to 56,420 records, and further filtered to 7,557 records for Melbourne-area locations.
2. **Feature Engineering**: Created a `Season` feature from the `Date` column and dropped `Date` to focus on seasonal patterns.
3. **Renaming Columns**: Renamed `RainToday` to `RainYesterday` and `RainTomorrow` to `RainToday` to align with the goal of predicting today’s rainfall using historical data.
4. **Encoding and Scaling**:
   - Numerical features scaled using `StandardScaler`.
   - Categorical features one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`.
5. **Data Splitting**: Split into training (80%) and test (20%) sets with stratification to maintain class balance (No: 76%, Yes: 24%).

## 📂 Repository Structure

The project files are organized as follows:

```
Course1_Rainfall_Prediction/
├── rainfall_prediction.ipynb  # Jupyter Notebook with the complete project code
├── weatherAUS.csv            # Dataset (not included in repo due to size; download from Kaggle)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

## 🛠️ Requirements

To run this project, install the following Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Detailed versions (as per the project document):
- `pandas==2.3.0`
- `numpy==2.3.1`
- `scikit-learn==1.7.0`
- `matplotlib==3.10.3`
- `seaborn==0.13.2`

Additional dependencies:
- `python-dateutil>=2.9.0.post0`
- `scipy>=1.8.0`
- `joblib>=1.2.0`

The project was developed in a Jupyter Notebook environment, but the code can be adapted for other Python environments.

## 🚀 Running the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/IBM-AI-Engineering-Professional-Certificate.git
   cd IBM-AI-Engineering-Professional-Certificate/Course1_Rainfall_Prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**:
   - Obtain the `weatherAUS.csv` file from [Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/).
   - Place it in the `Course1_Rainfall_Prediction` folder.

4. **Run the Notebook**:
   - Open `rainfall_prediction.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells to preprocess the data, train the models, and evaluate results.

## 📈 Results

- **Random Forest Classifier**:
  - Best Parameters: `max_depth=20`, `min_samples_split=5`, `n_estimators=100`.
  - Cross-Validation Accuracy: 0.85.
  - Test Set Accuracy: 0.85.
  - Classification Report: High precision and recall for "No" (dry days), but lower recall for "Yes" (rainy days) due to class imbalance (76% No, 24% Yes).
  - Most Important Feature: `Humidity3pm`.

- **Logistic Regression**:
  - Best Parameters: Not explicitly shown in the document, but optimized with `solver='liblinear'`, `penalty=['l1', 'l2']`, and `class_weight=[None, 'balanced']`.
  - Test Set Accuracy: 0.83.
  - Classification Report: Slightly lower performance than Random Forest, particularly in recall for "Yes" (0.51 vs. 0.52).
  - Confusion Matrix: Improved handling of imbalanced classes when `class_weight='balanced'`.

- **Key Insights**:
  - The dataset is imbalanced (76% "No" vs. 24% "Yes"), affecting recall for rainy days.
  - Random Forest outperforms Logistic Regression slightly, likely due to its ability to capture non-linear relationships.
  - `Humidity3pm` is the most critical feature, indicating its strong influence on rainfall prediction.

## 🙏 Acknowledgments

- **Australian Government's Bureau of Meteorology** 🌏 for providing the weather dataset.
- **IBM Corporation** for designing the AI Engineering Professional Certificate course.
- **Jeff Grossman** and **Abhishek Gagneja** for creating the project structure and guidelines.

## 🌟 Future Improvements

- **Address Class Imbalance**: Apply techniques like SMOTE or oversampling to improve recall for rainy days.
- **Incorporate More Data**: Include additional locations or extend the time frame to increase dataset size.
- **Feature Engineering**: Explore clustering-based features or impute missing values to retain more data.
- **Try Additional Models**: Implement K-Nearest Neighbors or Support Vector Machines, as mentioned in the original README, to compare performance.

Thank you for exploring the Rainfall Prediction Classifier project! 🌟 Feel free to contribute, open issues, or suggest improvements via GitHub! 🚀
