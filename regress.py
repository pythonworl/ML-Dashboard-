import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define file paths for saving/loading
MODELS_DIR = "models_regression"
RESULTS_DIR = "results_regression"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 ML Model Performance Dashboard")
st.write("This dashboard evaluates different ML models for regression, classification, and time series prediction tasks.")

# Create tabs for different model types
regression_tab, classification_tab, rnn_tab, logistic_tab, lstm_tab = st.tabs([
    "Regression (Uber Fares)", 
    "Classification (Drug Prediction)", 
    "Time Series (Stock Prediction)",
    "Logistic Regression (Marketing)",
    "LSTM Text Generation"
])

# ================== REGRESSION TAB ==================
with regression_tab:
    # Load and sample dataset (use a smaller subset for faster development)
    file_path = r"C:\Users\fnu.sawera\Downloads\archive (12)\uber.csv"  # Update this path to your dataset location
    data = pd.read_csv(file_path)

    # Sample 10% of the data for faster development (remove or adjust for full dataset)
    data = data.sample(frac=0.1, random_state=42)  # Use 10% of data; adjust frac or remove for full dataset

    # Preprocess the dataset
    # Convert pickup_datetime to datetime and extract useful features
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day_of_week'] = data['pickup_datetime'].dt.dayofweek
    data['month'] = data['pickup_datetime'].dt.month

    # Drop unnecessary columns and handle missing values
    data = data.drop(columns=['key', 'pickup_datetime'])  # Drop non-predictive columns
    data = data.dropna()  # Drop rows with missing values

    # Define features and target (fare_amount is the target for regression)
    X = data.drop(columns=['fare_amount'])
    y = data['fare_amount']

    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define regressors with optimized parameters
    regressors = {
        "Simple Linear Regression": LinearRegression(),
        "Multiple Linear Regression": LinearRegression(),  # Same as Simple Linear for this case, but can handle multiple features
        "Decision Tree": DecisionTreeRegressor(max_depth=3),  # Reduced max_depth for speed
        "k-NN": KNeighborsRegressor(n_neighbors=3, n_jobs=-1),  # Reduced neighbors, parallel processing
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=3, n_jobs=-1),  # Reduced estimators and depth, parallel processing
        "AdaBoost": AdaBoostRegressor(n_estimators=50, learning_rate=0.1),  # Reduced estimators
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=-1)  # Reduced estimators and depth, parallel processing
    }

    # Note about Naïve Bayes
    st.write("**Note**: Naïve Bayes is available in the Classification tab as it is primarily a classification algorithm.")

    # Function to evaluate models (regression metrics)
    @st.cache_resource  # Cache the model evaluation for speed
    def evaluate_model(_model, save_path=None):
        if save_path and os.path.exists(save_path):
            # Load pre-trained model if it exists
            model = joblib.load(save_path)
        else:
            # Train and save the model if it doesn't exist
            _model.fit(X_train, y_train)
            if save_path:
                joblib.dump(_model, save_path)
            model = _model
        
        y_pred = model.predict(X_test)
        return {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R²": r2_score(y_test, y_pred)
        }

    # Function for cross-validation (regression metrics) with fewer folds
    @st.cache_resource  # Cache the cross-validation for speed
    def cross_validate_model(_model, save_path=None):
        if save_path and os.path.exists(save_path):
            # Load pre-computed cross-validation results if they exist
            return joblib.load(save_path)
        else:
            # Compute cross-validation with 3 folds instead of 5 for speed
            cv_scores = {
                "CV RMSE": np.mean(np.sqrt(-cross_val_score(_model, X, y, cv=3, scoring='neg_mean_squared_error'))),
                "CV MAE": np.mean(-cross_val_score(_model, X, y, cv=3, scoring='neg_mean_absolute_error')),
                "CV R²": np.mean(cross_val_score(_model, X, y, cv=3, scoring='r2'))
            }
            if save_path:
                joblib.dump(cv_scores, save_path)
            return cv_scores

    # Select model
    selected_model_name = st.selectbox("Choose a model:", list(regressors.keys()))
    selected_model = regressors[selected_model_name]

    # Define save/load paths for the selected model
    model_save_path = os.path.join(MODELS_DIR, f"{selected_model_name}_model.joblib")
    cv_save_path = os.path.join(RESULTS_DIR, f"{selected_model_name}_cv_results.joblib")

    # Get results (load or compute)
    results_before_cv = evaluate_model(selected_model, model_save_path)
    results_after_cv = cross_validate_model(selected_model, cv_save_path)

    # Convert results to DataFrames
    df_before_cv = pd.DataFrame([results_before_cv], index=[selected_model_name])
    df_after_cv = pd.DataFrame([results_after_cv], index=[selected_model_name])

    # Display results
    st.subheader("📌 Performance Before Cross-Validation")
    st.dataframe(df_before_cv.style.format("{:.4f}"))

    st.subheader("📌 Performance After Cross-Validation")
    st.dataframe(df_after_cv.style.format("{:.4f}"))

# ================== CLASSIFICATION TAB (NAIVE BAYES) ==================
with classification_tab:
    st.header("Drug Prediction using Naive Bayes")
    
    # Load the dataset
    drug_data = pd.read_csv(r"C:\Users\fnu.sawera\OneDrive - University of Central Asia\Desktop\Machine learning\drug200.csv")

    # Encode categorical variables
    label_encoders = {}
    for column in ['Sex', 'BP', 'Cholesterol']:
        le = LabelEncoder()
        drug_data[column] = le.fit_transform(drug_data[column])
        label_encoders[column] = le

    # Features and target
    X_drug = drug_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
    y_drug = drug_data['Drug']

    # Split the data
    X_train_drug, X_test_drug, y_train_drug, y_test_drug = train_test_split(
        X_drug, y_drug, test_size=0.2, random_state=42)

    # Naive Bayes model
    nb_model = GaussianNB()
    nb_model.fit(X_train_drug, y_train_drug)

    # Make predictions on test set
    y_pred_drug = nb_model.predict(X_test_drug)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test_drug, y_pred_drug)
    precision = precision_score(y_test_drug, y_pred_drug, average='weighted')
    recall = recall_score(y_test_drug, y_pred_drug, average='weighted')
    f1 = f1_score(y_test_drug, y_pred_drug, average='weighted')
    conf_matrix = confusion_matrix(y_test_drug, y_pred_drug)

    # Calculate specificity (requires calculating for each class and then averaging)
    tn_fp_fn_tp = confusion_matrix(y_test_drug, y_pred_drug)
    specificity = []
    for i in range(len(tn_fp_fn_tp)):
        tn = sum(sum(tn_fp_fn_tp)) - sum(tn_fp_fn_tp[i]) - sum(tn_fp_fn_tp[:, i]) + tn_fp_fn_tp[i][i]
        fp = sum(tn_fp_fn_tp[:, i]) - tn_fp_fn_tp[i][i]
        specificity.append(tn / (tn + fp))
    specificity = np.mean(specificity)  # Convert to single float value

    # Prediction Section
    st.subheader("Make a Prediction")

    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=30, key='drug_age')
        sex = st.selectbox('Sex', label_encoders['Sex'].classes_, key='drug_sex')
        bp = st.selectbox('BP', label_encoders['BP'].classes_, key='drug_bp')
    with col2:
        cholesterol = st.selectbox('Cholesterol', label_encoders['Cholesterol'].classes_, key='drug_chol')
        na_to_k = st.number_input('Na to K Ratio', min_value=0.0, max_value=50.0, value=15.0, key='drug_na_k')

    # Encode user inputs
    sex_encoded = label_encoders['Sex'].transform([sex])[0]
    bp_encoded = label_encoders['BP'].transform([bp])[0]
    cholesterol_encoded = label_encoders['Cholesterol'].transform([cholesterol])[0]

    # Prediction button
    if st.button('Predict Drug', key='drug_predict'):
        user_data = [[age, sex_encoded, bp_encoded, cholesterol_encoded, na_to_k]]
        prediction = nb_model.predict(user_data)[0]
        st.success(f'The predicted drug is: {prediction}')

    # Performance Metrics Section
    with st.expander("View Model Performance Metrics", expanded=False):
        st.subheader("Model Performance Metrics")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
            st.metric("Precision", f"{precision:.2f}")
        with col2:
            st.metric("Recall", f"{recall:.2f}")
            st.metric("F1 Score", f"{f1:.2f}")
        with col3:
            st.metric("Specificity", f"{float(specificity):.2f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=nb_model.classes_, yticklabels=nb_model.classes_, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

# ================== RNN TAB (STOCK PREDICTION) ==================
with rnn_tab:
    st.header("Google Stock Price Prediction with Simple RNN")
    st.write("This implements a simple RNN from scratch to predict stock prices")

    # Load the data
    @st.cache_data
    def load_data():
        data = pd.read_csv(r"C:\Users\fnu.sawera\OneDrive - University of Central Asia\Desktop\Machine learning\GOOG.csv", parse_dates=['date'])
        data = data.sort_values('date')
        return data

    data = load_data()

    # Show raw data if checkbox is checked
    if st.checkbox('Show raw data', key='rnn_raw_data'):
        st.subheader('Raw data')
        st.write(data)

    # Sidebar controls
    st.sidebar.header('RNN Parameters')
    seq_length = st.sidebar.slider('Sequence length (days)', 5, 30, 10, key='rnn_seq_len')
    hidden_size = st.sidebar.slider('Hidden layer size', 16, 128, 32, key='rnn_hidden_size')
    epochs = st.sidebar.slider('Training epochs', 10, 500, 50, key='rnn_epochs')
    learning_rate = st.sidebar.slider('Learning rate', 0.0001, 0.01, 0.001, key='rnn_lr')

    # Model file path
    MODEL_FILE = 'rnn_model.pkl'
    SCALER_FILE = 'scaler.pkl'

    # Use closing prices and normalize them
    prices = data['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences for RNN
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data)-seq_length-1):
            X.append(data[i:(i+seq_length), 0])
            y.append(data[i+seq_length, 0])
        return np.array(X), np.array(y)

    X_rnn, y_rnn = create_sequences(prices_scaled, seq_length)

    # Split into train and test
    train_size = int(0.8 * len(X_rnn))
    X_train_rnn, X_test_rnn = X_rnn[:train_size], X_rnn[train_size:]
    y_train_rnn, y_test_rnn = y_rnn[:train_size], y_rnn[train_size:]

    # Reshape for RNN (samples, time steps, features)
    X_train_rnn = np.reshape(X_train_rnn, (X_train_rnn.shape[0], X_train_rnn.shape[1], 1))
    X_test_rnn = np.reshape(X_test_rnn, (X_test_rnn.shape[0], X_test_rnn.shape[1], 1))

    # RNN implementation
    class SimpleRNN:
        def __init__(self, input_size, hidden_size, output_size):
            # Initialize weights
            self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
            self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
            self.Why = np.random.randn(output_size, hidden_size) * 0.01
            self.bh = np.zeros((hidden_size, 1))
            self.by = np.zeros((output_size, 1))
            self.hidden_states = []
            
        def forward(self, inputs):
            h_prev = np.zeros((self.Whh.shape[0], 1))
            self.hidden_states = [h_prev]
            
            for x in inputs:
                x = x.reshape(-1, 1)
                h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev) + self.bh)
                self.hidden_states.append(h)
                h_prev = h

            y_hat = np.dot(self.Why, h) + self.by
            return y_hat, h

        def backward(self, inputs, y_true, y_pred, learning_rate):
            dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
            dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
            dh_next = np.zeros_like(self.hidden_states[0])

            dy = y_pred - y_true.reshape(-1, 1)

            for t in reversed(range(len(inputs))):
                dWhy += np.dot(dy, self.hidden_states[t+1].T)
                dby += dy

                dh = np.dot(self.Why.T, dy) + dh_next
                dh_raw = (1 - self.hidden_states[t+1] ** 2) * dh

                dbh += dh_raw
                x = inputs[t].reshape(-1, 1)
                dWxh += np.dot(dh_raw, x.T)
                dWhh += np.dot(dh_raw, self.hidden_states[t].T)

                dh_next = np.dot(self.Whh.T, dh_raw)

            for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(dparam, -5, 5, out=dparam)

            self.Wxh -= learning_rate * dWxh
            self.Whh -= learning_rate * dWhh
            self.Why -= learning_rate * dWhy
            self.bh -= learning_rate * dbh
            self.by -= learning_rate * dby

        def train(self, X, y, epochs=100, learning_rate=0.001):
            losses = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                total_loss = 0
                for i in range(len(X)):
                    y_pred, _ = self.forward(X[i])
                    loss = np.mean((y_pred - y[i]) ** 2)
                    total_loss += loss
                    self.backward(X[i], y[i], y_pred, learning_rate)

                avg_loss = total_loss / len(X)
                losses.append(avg_loss)

                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')

            return losses

        def predict(self, X):
            predictions = []
            for i in range(len(X)):
                y_pred, _ = self.forward(X[i])
                predictions.append(y_pred[0, 0])
            return np.array(predictions)

        def save(self, filename):
            """Save model parameters to a file"""
            with open(filename, 'wb') as f:
                pickle.dump({
                    'Wxh': self.Wxh,
                    'Whh': self.Whh,
                    'Why': self.Why,
                    'bh': self.bh,
                    'by': self.by,
                    'input_size': self.Wxh.shape[1],
                    'hidden_size': self.Wxh.shape[0],
                    'output_size': self.Why.shape[0]
                }, f)

        @staticmethod
        def load(filename):
            """Load model parameters from a file"""
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                model = SimpleRNN(data['input_size'], data['hidden_size'], data['output_size'])
                model.Wxh = data['Wxh']
                model.Whh = data['Whh']
                model.Why = data['Why']
                model.bh = data['bh']
                model.by = data['by']
                return model

    # Check if saved model exists
    model_exists = os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)

    # Train or load model
    if st.sidebar.button('Train New Model', key='rnn_train'):
        st.subheader("Training the RNN")
        with st.spinner('Initializing model...'):
            rnn = SimpleRNN(input_size=1, hidden_size=hidden_size, output_size=1)

        with st.spinner('Training in progress...'):
            losses = rnn.train(X_train_rnn, y_train_rnn, epochs=epochs, learning_rate=learning_rate)
        
        # Save the trained model and scaler
        rnn.save(MODEL_FILE)
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        st.success("Model trained and saved successfully!")
    elif model_exists:
        st.subheader("Loading Pre-trained Model")
        with st.spinner('Loading model...'):
            rnn = SimpleRNN.load(MODEL_FILE)
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
        st.success("Pre-trained model loaded successfully!")
    else:
        st.warning("No pre-trained model found. Please train a new model.")

    # Make predictions if model is available
    if 'rnn' in locals():
        st.subheader("Making Predictions")
        with st.spinner('Generating predictions...'):
            train_predict = rnn.predict(X_train_rnn)
            test_predict = rnn.predict(X_test_rnn)

            # Inverse transform to original scale
            train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
            test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
            y_train_orig = scaler.inverse_transform(y_train_rnn.reshape(-1, 1))
            y_test_orig = scaler.inverse_transform(y_test_rnn.reshape(-1, 1))

        # Plot the results
        st.subheader("Prediction Results")
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(data['date'][seq_length:train_size+seq_length], y_train_orig, label='Actual Train')
        plt.plot(data['date'][seq_length:train_size+seq_length], train_predict, label='Predicted Train')
        plt.plot(data['date'][train_size+seq_length:train_size+seq_length+len(y_test_rnn)], y_test_orig, label='Actual Test')
        plt.plot(data['date'][train_size+seq_length:train_size+seq_length+len(y_test_rnn)], test_predict, label='Predicted Test')
        plt.title('Google Stock Price Prediction with Simple RNN')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig1)

        # Plot training loss if we just trained
        if 'losses' in locals():
            st.subheader("Training Loss")
            fig2 = plt.figure(figsize=(12, 6))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            st.pyplot(fig2)

# ================== LOGISTIC REGRESSION TAB ==================
with logistic_tab:
    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv(r"C:\Users\fnu.sawera\Downloads\archive (17)\Marketingcampaigns.csv")

    data = load_data()
    st.title("📊 Logistic Regression Dashboard – Marketing Campaign")

    # Show raw data
    if st.checkbox("Show Raw Data", key='logistic_raw_data'):
        st.write(data)

    # Preprocessing
    st.subheader("Data Preprocessing")

    # Drop missing
    data = data.dropna()

    # Drop ID and target from features
    X = data.drop(columns=['Customer id', 'Purchased'])
    y = data['Purchased']

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save model
    model_filename = "logistic_model.pkl"
    joblib.dump(model, model_filename)

    # Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # -----------------------------------
    # 🧠 User Input Prediction Section
    # -----------------------------------
    st.subheader("🔮 Predict Purchase Based on Input")

    # Get unique locations for input dropdown
    locations = data['Location'].unique().tolist()

    # Create input form
    with st.form("user_input_form"):
        age = st.number_input("Age", min_value=10, max_value=100, value=30, key='logistic_age')
        gender = st.selectbox("Gender", ["Male", "Female"], key='logistic_gender')
        location = st.selectbox("Location", locations, key='logistic_location')
        email_opened = st.selectbox("Email Opened", [0, 1], key='logistic_email_opened')
        email_clicked = st.selectbox("Email Clicked", [0, 1], key='logistic_email_clicked')
        page_visits = st.slider("Product Page Visits", 0, 10, 2, key='logistic_page_visits')
        discount = st.selectbox("Discount Offered", [0, 1], key='logistic_discount')
        submit = st.form_submit_button("Predict")

    if submit:
        # Map gender to binary
        gender_val = 1 if gender == "Female" else 0

        # Create a single row dataframe
        input_dict = {
            'Age': age,
            'Gender': gender_val,
            'Email Opened': email_opened,
            'Email Clicked': email_clicked,
            'Product page visit': page_visits,
            'Discount offered': discount,
        }

        # Add one-hot encoding for location
        for loc in [col for col in X.columns if "Location_" in col]:
            input_dict[loc] = 1 if f"Location_{location}" == loc else 0

        # Make sure all features match training data
        input_df = pd.DataFrame([input_dict], columns=X.columns)

        # Load saved model and predict
        loaded_model = joblib.load(model_filename)
        prediction = loaded_model.predict(input_df)[0]

        # Output
        st.success("🎯 Prediction: **Purchase**" if prediction == 1 else "❌ Prediction: **No Purchase**")

    # Download model
    with open(model_filename, "rb") as file:
        st.download_button("📥 Download Trained Logistic Model", file, file_name=model_filename)

# ================== LSTM TEXT GENERATION TAB ==================
with lstm_tab:
    # Function to clean the text (remove Project Gutenberg headers/footers)
    def clean_text(text):
        # Find the start of the actual content (after Project Gutenberg header)
        start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            start_idx = text.find('\n', start_idx) + 1  # Skip to the line after the marker
            text = text[start_idx:end_idx].strip()
        return text

    # Function to preprocess the text and create sequences
    def preprocess_text(text, max_length=100, step=3):
        chars = sorted(list(set(text)))
        char_to_idx = {char: idx for idx, char in enumerate(chars)}
        idx_to_char = {idx: char for idx, char in enumerate(chars)}
        sequences = []
        next_chars = []
        for i in range(0, len(text) - max_length, step):
            sequences.append(text[i:i + max_length])
            next_chars.append(text[i + max_length])
        X = np.zeros((len(sequences), max_length, len(chars)), dtype=np.bool_)
        y = np.zeros((len(sequences), len(chars)), dtype=np.bool_)
        for i, seq in enumerate(sequences):
            for t, char in enumerate(seq):
                X[i, t, char_to_idx[char]] = 1
            y[i, char_to_idx[next_chars[i]]] = 1
        return X, y, chars, char_to_idx, idx_to_char

    # Function to build and train the LSTM model
    def train_model(X, y, chars, max_length, epochs=50):
        model = Sequential([
            LSTM(256, input_shape=(max_length, len(chars)), return_sequences=True),
            Dropout(0.2),  # Add dropout to prevent overfitting
            LSTM(256),
            Dropout(0.2),
            Dense(len(chars), activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        # Add early stopping and model checkpointing
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        checkpoint = ModelCheckpoint('lstm_model.h5', monitor='loss', save_best_only=True, verbose=1)
        model.fit(X, y, batch_size=128, epochs=epochs, callbacks=[early_stopping, checkpoint], verbose=1)
        return model

    # Function to generate text with top-k sampling
    def generate_text(model, seed_text, num_chars, char_to_idx, idx_to_char, max_length, chars, temperature=0.7, top_k=5):
        generated_text = seed_text
        for _ in range(num_chars):
            sequence = np.zeros((1, max_length, len(chars)))
            for t, char in enumerate(seed_text[-max_length:]):
                if char in char_to_idx:
                    sequence[0, t, char_to_idx[char]] = 1
            preds = model.predict(sequence, verbose=0)[0]
            # Top-k sampling: select top k probabilities
            top_indices = np.argsort(preds)[-top_k:]
            top_probs = preds[top_indices]
            top_probs = top_probs / np.sum(top_probs)  # Normalize
            # Apply temperature
            top_probs = np.log(top_probs + 1e-10) / temperature
            exp_probs = np.exp(top_probs)
            top_probs = exp_probs / np.sum(exp_probs)
            next_idx = np.random.choice(top_indices, p=top_probs)
            next_char = idx_to_char[next_idx]
            generated_text += next_char
            seed_text = generated_text[-max_length:]
        return generated_text

    # Main function for LSTM tab
    dataset_path = '1661-0.txt'
    model_path = 'lstm_model.h5'
    mappings_path = 'char_mappings.pkl'
    max_length = 100
    step = 3
    epochs = 50

    # Check if model and mappings exist
    model_exists = os.path.exists(model_path) and os.path.exists(mappings_path)

    # Load or train the model
    if not model_exists:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            text = file.read()
        # Clean the text
        text = clean_text(text)
        X, y, chars, char_to_idx, idx_to_char = preprocess_text(text, max_length, step)
        st.write("Training the LSTM model...")
        model = train_model(X, y, chars, max_length, epochs)
        with open(mappings_path, 'wb') as f:
            pickle.dump({'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}, f)
        st.write(f"Model saved as '{model_path}' and mappings saved as '{mappings_path}'.")
    else:
        model = tf.keras.models.load_model(model_path)
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        char_to_idx = mappings['char_to_idx']
        idx_to_char = mappings['idx_to_char']
        chars = list(char_to_idx.keys())
        st.write("Loaded existing model and mappings.")

    # Streamlit app
    st.title("Sherlock Holmes Text Generator")
    st.write("Generate text in the style of 'The Adventures of Sherlock Holmes' using an LSTM model.")
    seed_text = st.text_input("Enter seed text (at least a few characters):", "To Sherlock Holmes", key='lstm_seed')
    num_chars = st.slider("Number of characters to generate:", 50, 500, 200, key='lstm_num_chars')
    temperature = st.slider("Temperature (controls randomness, lower = less random):", 0.1, 1.0, 0.7, key='lstm_temp')
    top_k = st.slider("Top-k sampling (higher = more diverse):", 1, 10, 5, key='lstm_topk')
    if st.button("Generate Text", key='lstm_generate'):
        if len(seed_text) < 10:
            st.error("Please enter a seed text with at least 10 characters.")
        else:
            with st.spinner("Generating text..."):
                generated_text = generate_text(model, seed_text, num_chars, char_to_idx, idx_to_char, max_length, chars, temperature, top_k)
            st.subheader("Generated Text:")
            st.write(generated_text)
    st.markdown("""
    ### Instructions
    - **Seed Text**: Enter a starting text (at least 10 characters).
    - **Number of Characters**: Choose how many characters to generate (50–500).
    - **Temperature**: Lower values (e.g., 0.5) make text more predictable; higher values (e.g., 1.0) increase randomness.
    - **Top-k**: Higher values allow more character choices, increasing diversity but possibly randomness.
    - The model was trained on 'The Adventures of Sherlock Holmes' by Arthur Conan Doyle.
    """)