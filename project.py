import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import numpy as np
import requests
import urllib.parse
import os
import base64
import difflib

# Reset functionality
if "trigger_reset" not in st.session_state:
    st.session_state["trigger_reset"] = False

if st.session_state["trigger_reset"]:
    st.session_state.clear()
    st.session_state["trigger_reset"] = False
    st.rerun()




# Initialize default dataset tracker on first load
if "eda_last_used" not in st.session_state:
    st.session_state["eda_last_used"] = "default"

# 🔄 Improved Column Mapping Function (Fixed)
def auto_map_columns(df):
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Define your standard columns and their potential variations
    standard_columns = {
        "title": ["title", "movie_title", "name", "movie", "film"],
        "rating": ["rating", "user_rating", "score", "ratings", "imdb_rating"],
        "votes": ["votes", "vote_count", "num_votes", "vote"],
        "revenue_(millions)": ["revenue", "box_office", "collection", "revenue_m", "earnings", "revenue_(millions)"],
        "metascore": ["metascore", "critic_score", "meta_score", "meta"],
        "runtime_(minutes)": ["runtime", "duration", "length", "runtime_min", "runtime_(minutes)"],
        "year": ["year", "release_year", "release"],
        "success": ["success", "label", "target", "outcome", "is_successful"]
    }

    new_cols = {}
    
    # Normalize column names for comparison
    df_cols_normalized = {col: col.lower().strip().replace("-", "_").replace(" ", "_") for col in df_copy.columns}
    
    for std_col, aliases in standard_columns.items():
        for alias in aliases:
            for original_col, normalized_col in df_cols_normalized.items():
                if alias.lower() == normalized_col:
                    new_cols[original_col] = std_col
                    break
            if std_col in new_cols.values():  # If we found a match, break out of alias loop
                break
    
    # Apply the mapping
    df_copy = df_copy.rename(columns=new_cols)
    return df_copy

# 🧹 Enhanced Dataset Preprocessing Function (Fixed)
def preprocess_dataset(df_raw):
    if df_raw is None or not isinstance(df_raw, pd.DataFrame):
        return None, None, None

    # Apply column mapping FIRST
    df = auto_map_columns(df_raw)
    
    # Store original columns before normalization
    original_columns = df.columns.tolist()
    
    # Standard clean-up: lowercase, trim, underscore
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")

    # Find success column with priority order
    success_col = None
    success_candidates = ["success", "target", "label", "outcome", "is_successful"]
    
    for candidate in success_candidates:
        if candidate in df.columns:
            success_col = candidate
            break
    
    # If no success column found, check original raw data
    if not success_col:
        # Check original raw dataframe columns
        for col in df_raw.columns:
            col_lower = col.lower().strip().replace(" ", "_").replace("-", "_")
            if any(candidate in col_lower for candidate in success_candidates):
                # Find the corresponding column in processed df
                processed_col = col.lower().strip().replace(" ", "_").replace("-", "_")
                if processed_col in df.columns:
                    success_col = processed_col
                    break
    
    if success_col and success_col in df.columns:
        # Clean the success column
        df[success_col] = pd.to_numeric(df[success_col], errors="coerce")
        df = df.dropna(subset=[success_col])
        df[success_col] = df[success_col].round().astype(int)
        # st.success(f"✅ Found success column: '{success_col}'")->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        # Create fallback success column but warn user
        df["success"] = np.random.randint(0, 2, size=len(df))
        success_col = "success"
        st.warning("⚠️ No success column found. Created random success values for demo purposes.")

    # Find title column with priority order
    title_candidates = ["title", "movie_title", "name", "movie", "film"]
    title_col = None
    
    for candidate in title_candidates:
        if candidate in df.columns:
            title_col = candidate
            break
    
    if not title_col:
        # Fallback: pick first string-based column with many unique values
        object_cols = df.select_dtypes(include="object").columns
        if len(object_cols) > 0:
            # Find the one with most unique values (likely to be titles)
            unique_counts = {col: df[col].nunique() for col in object_cols}
            title_col = max(unique_counts, key=unique_counts.get)
        else:
            title_col = df.columns[0]  # Last resort fallback

    return df, success_col, title_col

# 🔍 Function to find best matching poster
def find_best_poster(title, poster_folder="posters"):
    if not os.path.exists(poster_folder):
        return None
    
    poster_files = os.listdir(poster_folder)
    poster_files = [f for f in poster_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0].lower() for f in poster_files]
    match = difflib.get_close_matches(title.lower(), base_names, n=1, cutoff=0.5)
    if match:
        matched_file = match[0]
        ext = os.path.splitext(poster_files[base_names.index(matched_file)])[1]
        return os.path.join(poster_folder, matched_file + ext)
    return None

# 🚀 Page Setup
st.set_page_config(page_title="MSP", layout="wide")
st.markdown("<h1 style='display: inline-block;'>🎬 MSP</h1>", unsafe_allow_html=True)

# ✅ Initialize dataset source as 'default'
if "dataset_source" not in st.session_state:
    st.session_state["dataset_source"] = "default"

# 🔹 Sidebar Model Selection
with st.sidebar:
    st.title("🧠 Choose ML Model")
    model_choice = st.selectbox(
        "Select a model for prediction:",
        ("Logistic Regression", "Random Forest", "Decision Tree")
    )
    for _ in range(6):  # Increase/decrease to adjust space
        st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown("###")
    st.markdown("---")
    if st.button("🔁 Reset App"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["uploader_key"] = str(np.random.randint(100000))
        st.session_state["dataset_source"] = "default"
        st.rerun()    
        # Regenerate uploader key to force reset file_uploader
        st.session_state["uploader_key"] = str(np.random.randint(100000))
        st.session_state["dataset_source"] = "default"
        st.rerun()




# 🔄 Centralized Data Loader (Improved)
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("movie_success_rate.csv")
    except:
        return None

    # Apply preprocessing
    df, success_col, title_col = preprocess_dataset(df)
    return df

# Tabs - Reordered as requested
tabs = st.tabs(["📚 Summary", "➕ Add Yours", "📊 EDA", "🎯 Prediction", "🏆 Top 10"])

# --- Summary Tab ---
with tabs[0]:
    st.title("📚 MSP")
    st.markdown("""
    **MSP (Movie Success Predictor)** is a platform that uses real-world movie data to:

    - Analyze trends that make movies successful 📈  
    - Predict movie success with Machine Learning 🎯  
    - Upload your own data and get insights 📂  
    - Inspired by IMDb, with visualizations, prediction, and clean UI  
    """)

# --- Add Yours Tab ---
with tabs[1]:
    st.title("📂 Add Your Dataset")
    
    uploader_key = st.session_state.get("uploader_key", "default")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key=uploader_key)


    if uploaded_file:
        try:
            # Read the uploaded file
            user_df = pd.read_csv(uploaded_file)
            
            # Store in session state
            st.session_state["user_dataset"] = user_df
            st.session_state["dataset_source"] = "user"
            
            # Clear any previous processed state to force reprocessing
            for key in ["processed_df", "success_col", "title_col", "last_dataset_source"]:
                st.session_state.pop(key, None)

            st.success("✅ Dataset uploaded successfully! Preview:")
            st.dataframe(user_df.head())
            
            # Show column mapping preview
            mapped_df = auto_map_columns(user_df)
            if not mapped_df.columns.equals(user_df.columns):
                st.info("🔄 Column mapping applied:")
                mapping_info = []
                for orig, mapped in zip(user_df.columns, mapped_df.columns):
                    if orig != mapped:
                        mapping_info.append(f"'{orig}' → '{mapped}'")
                if mapping_info:
                    st.write(" | ".join(mapping_info))
            
            # Validate dataset
            st.subheader("📊 Dataset Validation")
            validation_results = []
            
            # Check for success column
            success_found = False
            for col in user_df.columns:
                if any(keyword in col.lower() for keyword in ['success', 'target', 'label', 'outcome']):
                    validation_results.append(f"✅ Success column found: '{col}'")
                    success_found = True
                    break
            
            if not success_found:
                validation_results.append("⚠️ No success column found - predictions will use random values")
            
            # Check for title column
            title_found = False
            for col in user_df.columns:
                if any(keyword in col.lower() for keyword in ['title', 'movie', 'name', 'film']):
                    validation_results.append(f"✅ Title column found: '{col}'")
                    title_found = True
                    break
            
            if not title_found:
                validation_results.append("⚠️ No clear title column found - will use first string column")
            
            # Check numeric features
            numeric_cols = user_df.select_dtypes(include=[np.number]).columns
            validation_results.append(f"📊 Found {len(numeric_cols)} numeric features for prediction")
            
            # Compact badge-style summary
            valid_parts = []
            missing_parts = []

# Check for success column
            success_found = any(any(keyword in col.lower() for keyword in ['success', 'target', 'label', 'outcome'])
                    for col in user_df.columns)
            if success_found:
                valid_parts.append("success")
            else:
                missing_parts.append("success")

# Check for title column
            title_found = any(any(keyword in col.lower() for keyword in ['title', 'movie', 'name', 'film'])
                  for col in user_df.columns)
            if title_found:
                valid_parts.append("title") 
            else:
                missing_parts.append("title")

# Check numeric features
            numeric_cols = user_df.select_dtypes(include=[np.number]).columns
            numeric_count = len(numeric_cols)

# 🔽 Display all results in a compact format
            summary = f"🟢 Valid: {', '.join(valid_parts)}\n"
            summary += f"🔢 Features: {numeric_count} numeric\n"
            if missing_parts:
                summary += f"⚠️ Missing: {', '.join(missing_parts)}"

            st.markdown(f"```markdown\n{summary}\n```")


        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")

    else:
        # Show currently active dataset preview
        source = st.session_state.get("dataset_source", "default")
        if source == "user" and "user_dataset" in st.session_state:
            st.write("📄 Current User Dataset:")
            st.dataframe(st.session_state["user_dataset"].head())
        else:
            st.info("📁 Upload a CSV file to get started, or use the default dataset in other tabs.")

# --- EDA Tab ---
with tabs[2]:
    st.title("📊 Exploratory Data Analysis")

    # Clear any cached processed data when switching datasets
    if "last_dataset_source" not in st.session_state:
        st.session_state["last_dataset_source"] = "default"
    
    current_source = st.session_state.get("dataset_source", "default")
    
    # If dataset source changed, clear cached processing
    if st.session_state["last_dataset_source"] != current_source:
        for key in ["processed_df", "success_col", "title_col"]:
            st.session_state.pop(key, None)
        st.session_state["last_dataset_source"] = current_source

    st.markdown(f"**📂 EDA Dataset Source:** `{current_source}`")

    # Load dataset based on source
    if current_source == "user" and "user_dataset" in st.session_state:
        df_raw = st.session_state["user_dataset"]
    else:
        try:
            df_raw = pd.read_csv("movie_success_rate.csv")
        except:
            st.warning("Default dataset not found.")
            st.stop()

    # Process dataset
    df, success_col, title_col = preprocess_dataset(df_raw)
    
    if df is None or df.empty:
        st.warning("Dataset is empty or invalid.")
        st.stop()

    # Store processed data in session state
    st.session_state["processed_df"] = df
    st.session_state["success_col"] = success_col
    st.session_state["title_col"] = title_col

    # --- EDA Plots ---
    figsize = (5, 3.5)
    col1, col2, col3 = st.columns([1, 2, 1])

    # Success Count Plot
    if success_col and success_col in df.columns and df[success_col].nunique() <= 10:
        fig1, ax1 = plt.subplots(figsize=figsize)
        sns.countplot(x=success_col, data=df, palette=["#FF6961", "#77DD77"], ax=ax1)
        ax1.set_title("Movie Success Distribution")
        with col2:
            st.subheader("🎯 Success Count")
            st.pyplot(fig1)

    # Feature plots
    eda_candidates = df.select_dtypes(include=[np.number]).drop(columns=[success_col], errors="ignore")
    eda_features = eda_candidates.loc[:, eda_candidates.nunique() > 5].columns[:4]
    
    for feature in eda_features:
        if feature in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            if success_col and success_col in df.columns:
                if feature == "rating":
                    sns.histplot(data=df, x=feature, hue=success_col, bins=30, kde=True,
                                 palette=["#FF6961", "#77DD77"], ax=ax)
                else:
                    sns.boxplot(x=success_col, y=feature, data=df,
                                palette=["#FF6961", "#77DD77"], ax=ax, dodge=False)
            else:
                sns.histplot(data=df, x=feature, bins=30, kde=True, ax=ax)

            ax.set_title(f"{feature.replace('_', ' ').title()} vs Success" if success_col in df.columns else f"{feature.replace('_', ' ').title()} Distribution")
            with col2:
                st.subheader(f"📊 {feature.replace('_', ' ').title()}")
                st.pyplot(fig)

    # Correlation heatmap
    default_corr_cols = ['rating', 'votes', 'revenue_(millions)', 'metascore']
    compact_cols = [col for col in default_corr_cols if col in df.columns]
    if success_col and success_col in df.columns:
        compact_cols.append(success_col)

    if len(compact_cols) >= 2:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(df[compact_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.4,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
        ax.set_title("Feature Correlation Matrix", fontsize=10)
        with col2:
            st.subheader("📊 Correlation Heatmap")
            st.pyplot(fig)

# --- Prediction Tab ---
with tabs[3]:
    st.title("🎯 Predict Movie Success")

    # Determine dataset source and load data
    # current_source = st.session_state.get("dataset_source", "default")
    
    # if current_source == "user" and "user_dataset" in st.session_state:
    #     df_raw = st.session_state["user_dataset"]
    #     st.info("📊 Using uploaded dataset for predictions")
    # else:
    #     try:
    #         df_raw = pd.read_csv("movie_success_rate.csv")
    #         st.info("📊 Using default dataset for predictions")
    
        # except:
        #     st.error("❌ Could not load dataset.")
        #     st.stop()

# Determine dataset source and load data
    current_source = st.session_state.get("dataset_source", "default")

    if current_source == "user" and "user_dataset" in st.session_state:
        df_raw = st.session_state["user_dataset"]
        prediction_info = ["📊 Using uploaded dataset"]
    else:
        try:
            df_raw = pd.read_csv("movie_success_rate.csv")
            prediction_info = ["📊 Using default dataset"]
        except:
            st.error("❌ Could not load dataset.")
            st.stop()

# Process the dataset
    df, success_col, title_col = preprocess_dataset(df_raw)

    if df is None or df.empty:
        st.warning("Dataset is empty or invalid.")
        st.stop()

# Prepare features
    all_cols = df.columns.tolist()
    exclude_cols = ["rank", success_col, title_col]
    numeric_cols = [col for col in all_cols if df[col].dtype in [np.float64, np.int64] and col not in exclude_cols]
    feature_cols = numeric_cols

# Clean numeric features
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())

#  Prepare target variable
#     if success_col in df.columns:
#         df[success_col] = pd.to_numeric(df[success_col], errors="coerce")
#         df = df.dropna(subset=[success_col])
#         df[success_col] = (df[success_col] >= 0.5).astype(int)
#         y = df[success_col]
#         prediction_info.append(f"🟢 Target column: {success_col}")
#     else:
#         st.error("❌ No valid success column found")
#         st.stop()

#     if numeric_cols:
#         prediction_info.append(f"🔢 Features used: {len(numeric_cols)}")

# # ✅ Show badge-style info summary
#     st.markdown(f"```markdown\n" + "\n".join(prediction_info) + "\n```")


#     # Prepare features and split data
#     X = df[feature_cols]
    
#     if len(X) == 0:
#         st.error("❌ No numeric features available for prediction")
#         st.stop()
        
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     st.subheader("🧠 Model Training")

#     # Train model based on selection
#     if model_choice == "Logistic Regression":
#         model = LogisticRegression(max_iter=2000, class_weight="balanced")
#     elif model_choice == "Random Forest":
#         model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
#     else:
#         model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

#     model.fit(X_train, y_train)
#     st.success(f"✅ {model_choice} training complete!")


# === Combined badge-style summary and training ===

    prediction_info = []

    if current_source == "user":
        prediction_info.append("📊 Using uploaded dataset")
    else:
        prediction_info.append("📊 Using default dataset")

# Prepare target variable
    if success_col in df.columns:
        df[success_col] = pd.to_numeric(df[success_col], errors="coerce")
        df = df.dropna(subset=[success_col])
        df[success_col] = (df[success_col] >= 0.5).astype(int)
        y = df[success_col]
        prediction_info.append(f"🟢 Target column: {success_col}")
        prediction_info.append(f"✅ Found success column: '{success_col}'")

        
    else:
        st.error("❌ No valid success column found")
        st.stop()

    if numeric_cols:
        prediction_info.append(f"🔢 Features used: {len(numeric_cols)}")

    # Prepare features and split data
    X = df[feature_cols]

    if len(X) == 0:
        st.error("❌ No numeric features available for prediction")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model based on selection
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    else:
        model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    model.fit(X_train, y_train)
    prediction_info.append(f"✅ {model_choice} training complete")

# ✅ Show everything in one clean summary block
    st.markdown(f"```markdown\n" + "\n".join(prediction_info) + "\n```")

    # Movie selection for prediction
    movie_titles = df[title_col].dropna().astype(str).unique()
    selected_movie = st.selectbox("🎬 Choose a Movie to Predict", sorted(movie_titles))

    if selected_movie:
        row = df[df[title_col].astype(str) == selected_movie].iloc[0]
        input_data = {col: row[col] if col in row else 0 for col in feature_cols}
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        st.markdown("### 🎬 Movie Details")
        for col in row.index:
            value = row[col]
            if pd.notnull(value) and str(value).strip() != "" and value != 0:
                st.markdown(f"**{col.replace('_', ' ').title()}**: {value}")

        st.markdown("### 🎯 Prediction Result")
        if prediction == 1:
            st.success("✅ This movie is **likely to be successful**.")
        else:
            st.error("❌ This movie is **less likely to succeed**.")

        # Model evaluation
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.markdown(f"**✅ Model Accuracy:** {acc:.2%}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        st.subheader("📊 Precision, Recall, F1-Score")
        st.dataframe(pd.DataFrame(report).transpose().round(2))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Not Successful", "Successful"]
        fig, ax = plt.subplots(figsize=(2.5, 2.2), dpi=120)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                    xticklabels=labels, yticklabels=labels,
                    annot_kws={"size": 10, "weight": "bold"},
                    linewidths=0.4, linecolor="white", square=True, ax=ax)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title("Confusion Matrix", fontsize=9)
        ax.tick_params(labelsize=8)

        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(fig)

        # Feature Importance (moved after confusion matrix)
        st.subheader("📊 Feature Importance Analysis")
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=X_train.columns)
            importances = importances.sort_values(ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(6,3))
            importances.head(10).plot(kind='barh', ax=ax_imp, color='skyblue')
            ax_imp.set_title("Top 10 Most Important Features")
            ax_imp.set_xlabel("Importance Score")
            plt.tight_layout()
            st.pyplot(fig_imp)
            
        elif model_choice == "Logistic Regression":
            if hasattr(model, "coef_"):
                coefs = pd.Series(abs(model.coef_[0]), index=X_train.columns)
                coefs = coefs.sort_values(ascending=False)
                
                fig_coef, ax_coef = plt.subplots(figsize=(6,3))
                coefs.head(10).plot(kind='barh', ax=ax_coef, color='lightcoral')
                ax_coef.set_title("Top 10 Most Important Features (Absolute Coefficients)")
                ax_coef.set_xlabel("Coefficient Magnitude")
                plt.tight_layout()
                st.pyplot(fig_coef)

def get_relevant_fields(row, title_col, max_fields=7):
    relevant_keywords = [
        "rating", "score", "metascore", "votes", "revenue", "runtime",
        "year", "genre", "language", "country", "budget", "profit",
        "actor", "cast", "director", "writer", "producer", "plot", "awards"
    ]

    selected = []
    for field in row.index:
        val = row[field]
        if field == title_col or pd.isnull(val) or val == 0:
            continue
        if isinstance(val, str) and (val.strip() == "" or val.startswith("http")):
            continue
        if not any(keyword in field.lower() for keyword in relevant_keywords):
            continue
        selected.append(field)
        if len(selected) >= max_fields:
            break
    return selected

# --- Top 10 Tab ---
with tabs[4]:
    st.title("🏆 Top 10 Successful Movies (Based on Dataset)")

    # Load and preprocess
    current_source = st.session_state.get("dataset_source", "default")
    
    if current_source == "user" and "user_dataset" in st.session_state:
        df_raw = st.session_state["user_dataset"]
    else:
        try:
            df_raw = pd.read_csv("movie_success_rate.csv")
        except:
            st.error("❌ Could not load dataset.")
            st.stop()

    df, success_col, title_col = preprocess_dataset(df_raw)

    if df is None or df.empty:
        st.warning("Dataset not available or invalid.")
        st.stop()

    # Filter only successful movies
    if success_col not in df.columns:
        st.warning(f"'{success_col}' column missing.")
        st.stop()

    df = df[df[success_col] == 1]

    if len(df) == 0:
        st.warning("No successful movies found in the dataset.")
        st.stop()

    # Drop rows missing any scoring-relevant data
    score_cols = ["rating", "votes", "revenue_(millions)", "metascore"]
    score_cols = [col for col in score_cols if col in df.columns]
    df = df.dropna(subset=score_cols, how='all')  # Changed to 'all' to be less restrictive

    # Normalize scores
    scaler = MinMaxScaler()
    if "votes" in df.columns:
        df["votes_scaled"] = scaler.fit_transform(df[["votes"]])
    else:
        df["votes_scaled"] = 0

    if "revenue_(millions)" in df.columns:
        df["revenue_scaled"] = scaler.fit_transform(df[["revenue_(millions)"]])
    else:
        df["revenue_scaled"] = 0

    # Weighted scoring logic
    df["score"] = (
        (df.get("rating", 0) * 3) +
        (df.get("votes_scaled", 0) * 2) +
        (df.get("revenue_scaled", 0) * 2) +
        (df.get("metascore", 0) * 0.5)
    )

    top10 = df.sort_values(by="score", ascending=False).drop_duplicates(title_col).head(10)

    # Display results
    for _, row in top10.iterrows():
        title = row.get(title_col, "Untitled")
        poster_path = find_best_poster(title)

        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                if poster_path:
                    st.image(poster_path, width=240)
                else:
                    st.info("📷 Poster not available")
                    
            with col2:
                st.markdown(f"### 🎬 {title}")
                if "rating" in row and pd.notnull(row["rating"]): 
                    st.markdown(f"**⭐ Rating:** {row['rating']}")
                if "genre" in row and pd.notnull(row["genre"]): 
                    st.markdown(f"**🎭 Genre:** {row['genre']}")
                if "metascore" in row and pd.notnull(row["metascore"]): 
                    st.markdown(f"**🧠 Metascore:** {row['metascore']}")
                if "revenue_(millions)" in row and pd.notnull(row["revenue_(millions)"]): 
                    st.markdown(f"**💰 Revenue:** ${row['revenue_(millions)']}M")
                if "votes" in row and pd.notnull(row["votes"]): 
                    st.markdown(f"**📊 Votes:** {int(row['votes'])}")
                st.markdown("---")