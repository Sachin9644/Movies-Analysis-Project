# Movie Revenue Prediction

A machine learning project that predicts movie revenue using various features like popularity, vote average, vote count, and runtime. This project compares the performance of four different regression models to determine which best estimates movie revenue.

## 📋 Project Overview

This project analyzes a dataset of 10,000 movies and builds predictive models to estimate movie revenue. The analysis includes exploratory data visualization and comparison of multiple machine learning algorithms.

## 🎯 Objectives

- Predict movie revenue based on movie characteristics
- Compare performance of different regression models
- Visualize relationships between movie features and revenue
- Provide insights into factors that influence movie success

## 📊 Dataset

The project uses `Top_10000_Movies.csv` containing information about 10,000+ movies with the following key features:
- **popularity**: Movie popularity score
- **vote_average**: Average user rating
- **vote_count**: Number of votes received
- **runtime**: Movie duration in minutes
- **revenue**: Box office revenue (target variable)

## 🛠️ Models Used

The project implements and compares four regression models:

1. **Linear Regression** - Simple linear relationship modeling
2. **Decision Tree Regressor** - Non-linear pattern recognition
3. **Random Forest Regressor** - Ensemble method for improved accuracy
4. **K-Nearest Neighbors** - Instance-based learning approach

## 📈 Visualizations

The project generates 8 different visualizations:
- Revenue distribution histogram
- Revenue boxplot
- Popularity vs Revenue scatter plot
- Vote Count vs Revenue scatter plot
- Vote Average vs Revenue bar plot
- Runtime distribution histogram
- Interactive Runtime vs Revenue plot (Plotly)
- Interactive Vote Average distribution (Plotly)

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd movie-revenue-prediction
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset**
   - Place `Top_10000_Movies.csv` in the project root directory
   - The CSV file should contain the required columns: popularity, vote_average, vote_count, runtime, revenue

### Running the Project

1. **Execute the main script**
   ```bash
   python revenue_prediction.py
   ```

2. **Expected Output**
   - Dataset information and statistics
   - 8 different data visualization plots
   - Model training progress for 4 algorithms
   - Performance metrics (MSE and R² score) for each model
   - Actual vs Predicted comparison plots
   - Linear regression coefficients

## 📁 Project Structure

```
movie-revenue-prediction/
│
├── revenue_prediction.py          # Main Python script
├── Movie_Revenue_Prediction_Report.docx  # Detailed project report
├── Top_10000_Movies.csv           # Dataset (required)
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation


## 📊 Results

The project evaluates each model using:
- **Mean Squared Error (MSE)**: Lower values indicate better performance
- **R-squared (R²) Score**: Higher values (closer to 1) indicate better fit

Based on the analysis, **Random Forest Regressor** typically shows the best performance among the four models tested.

## 🔍 Key Findings

- Movie popularity and vote count show strong correlation with revenue
- Runtime has a moderate impact on revenue prediction
- Ensemble methods (Random Forest) outperform simple linear models
- Data quality significantly affects model performance

## 🚀 Future Improvements

- **Feature Engineering**: Add genres, budget, release month/year
- **Data Transformation**: Apply log transformation to revenue for better distribution
- **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters
- **Advanced Models**: Implement XGBoost, Support Vector Regression
- **Cross-Validation**: Add k-fold cross-validation for robust evaluation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

 

 
