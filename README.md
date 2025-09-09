🎬 Movie Success Predictor (MSP)
📖 Overview

MSP (Movie Success Predictor) is a machine learning–powered web application built with Streamlit.
It allows users to:

Upload their own movie dataset (CSV format)

Perform exploratory data analysis (EDA) with clean visualizations

Train models to predict if a movie will be successful or not

View the Top 10 successful movies from the dataset with posters

This project is inspired by IMDb-style analysis and prediction.

⚡ Features

Upload custom datasets or use the default dataset

Automatic column mapping & preprocessing

Choose ML model:

Logistic Regression

Random Forest Classifier

Decision Tree Classifier

EDA visualizations (success distribution, feature plots, heatmaps)

Prediction results with accuracy, precision, recall, and confusion matrix

Top 10 movies display with poster matching

🛠️ Tech Stack

Python 3.9+

Streamlit → Web framework for interactive dashboards

Pandas → Data cleaning & manipulation

NumPy → Numerical operations

Matplotlib & Seaborn → Data visualization

Scikit-learn → Machine Learning models & evaluation

🚀 Getting Started
1. Clone the repository
git clone https://github.com/your-username/movie-success-predictor.git
cd movie-success-predictor

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run project.py

📂 Dataset

By default, the app uses movie_success_rate.csv.

You can upload your own dataset (CSV).

The app auto-detects important columns like title, rating, votes, revenue, metascore, and success.

📊 Example Outputs

Movie success predictions

EDA plots

Confusion matrix

Feature importance analysis

Top 10 successful movies with posters
