import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import psutil
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class IdleDetectionModel:
    """Lightweight ML model for detecting idle applications."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'cpu_usage_percent', 'memory_usage_mb', 'mouse_clicks_per_min',
            'keyboard_strokes_per_min', 'network_activity_kb', 
            'window_focus_time_sec', 'time_since_last_interaction_min',
            'power_consumption_watts'
        ]

    def load_data(self, data_path):
        """Load training data from CSV file."""
        try:
            df = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_data(self, df):
        """Preprocess the data for training."""
        # Select features and target
        X = df[self.feature_columns]
        y = df['status']

        # Handle missing values
        X = X.fillna(X.mean())

        return X, y

    def train_model(self, data_path, model_type='random_forest'):
        """Train the idle detection model."""
        print("Loading and preprocessing data...")
        df = self.load_data(data_path)
        if df is None:
            return False

        X, y = self.preprocess_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Choose lightweight model for Green AI
        if model_type == 'random_forest':
            # Lightweight Random Forest with limited trees
            self.model = RandomForestClassifier(
                n_estimators=50,  # Small number for efficiency
                max_depth=10,     # Limit depth
                random_state=42,
                n_jobs=1          # Single thread for lower power
            )
        else:
            # Even more lightweight - Logistic Regression
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='lbfgs'    # Efficient solver
            )

        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        return True

    def save_model(self, model_dir):
        """Save the trained model and scaler."""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, 'idle_detection_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    def load_model(self, model_dir):
        """Load trained model and scaler."""
        model_path = os.path.join(model_dir, 'idle_detection_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Model and scaler loaded successfully")
            return True
        else:
            print("Model files not found")
            return False

    def predict_idle_status(self, features):
        """Predict if an app is idle based on features."""
        if self.model is None:
            print("Model not loaded!")
            return None

        # Ensure features are in correct format
        if isinstance(features, dict):
            feature_array = np.array([[features[col] for col in self.feature_columns]])
        else:
            feature_array = np.array(features).reshape(1, -1)

        # Scale features
        scaled_features = self.scaler.transform(feature_array)

        # Make prediction
        prediction = self.model.predict(scaled_features)[0]
        probability = self.model.predict_proba(scaled_features)[0]

        return {
            'prediction': prediction,
            'idle_probability': probability[0] if prediction == 'Idle' else probability[1],
            'active_probability': probability[1] if prediction == 'Idle' else probability[0]
        }

class SystemMonitor:
    """Monitor system resources and app usage for idle detection."""

    def __init__(self):
        self.last_mouse_pos = None
        self.last_mouse_time = time.time()
        self.mouse_click_count = 0
        self.keyboard_stroke_count = 0

    def get_system_stats(self):
        """Get current system statistics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.used / (1024 * 1024)  # MB

        # Network activity (simplified)
        network = psutil.net_io_counters()
        network_kb = (network.bytes_sent + network.bytes_recv) / 1024

        return {
            'cpu_usage_percent': cpu_percent,
            'memory_usage_mb': memory_usage,
            'network_activity_kb': network_kb
        }

    def simulate_app_features(self, app_name="Unknown"):
        """Simulate app features for demonstration (in real app, this would collect real data)."""
        # Get actual system stats
        sys_stats = self.get_system_stats()

        # Simulate other features (in real implementation, these would be tracked)
        features = {
            'cpu_usage_percent': max(0, sys_stats['cpu_usage_percent'] + np.random.normal(0, 5)),
            'memory_usage_mb': max(10, np.random.normal(150, 50)),
            'mouse_clicks_per_min': max(0, np.random.poisson(3)),
            'keyboard_strokes_per_min': max(0, np.random.poisson(10)),
            'network_activity_kb': max(0, np.random.normal(20, 15)),
            'window_focus_time_sec': max(0, np.random.normal(60, 30)),
            'time_since_last_interaction_min': max(0, np.random.exponential(5)),
            'power_consumption_watts': max(5, np.random.normal(40, 15))
        }

        return features

class PowerSavingsCalculator:
    """Calculate potential power savings from closing idle apps."""

    @staticmethod
    def estimate_power_saved(idle_apps):
        """Estimate power savings from closing idle apps."""
        base_power_per_app = 15  # Watts per idle app
        total_apps = len(idle_apps)
        estimated_savings = total_apps * base_power_per_app

        # Calculate cost savings (assuming $0.12 per kWh)
        cost_per_kwh = 0.12
        daily_savings_kwh = (estimated_savings * 24) / 1000
        daily_cost_savings = daily_savings_kwh * cost_per_kwh
        monthly_cost_savings = daily_cost_savings * 30

        return {
            'power_saved_watts': estimated_savings,
            'daily_savings_kwh': daily_savings_kwh,
            'daily_cost_savings': daily_cost_savings,
            'monthly_cost_savings': monthly_cost_savings
        }

    @staticmethod
    def get_energy_tips():
        """Provide energy-saving tips."""
        tips = [
            "Close unused browser tabs to save CPU and memory",
            "Use sleep mode when away for more than 15 minutes",
            "Reduce screen brightness to save battery power",
            "Enable power saving mode on laptops",
            "Close background apps that aren't needed",
            "Use SSD instead of HDD for better efficiency",
            "Keep your system updated for optimal performance",
            "Use dark themes to reduce screen power consumption"
        ]
        return tips

if __name__ == "__main__":
    # Example usage
    print("Smart Power Reminder - ML Backend")
    print("=" * 40)

    # Initialize model
    model = IdleDetectionModel()
    
        # Train model
    # Use absolute path relative to project root for robustness
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "app_usage_data.csv")
    model_dir = os.path.join(project_root, "models")
    if os.path.exists(data_path):
        success = model.train_model(data_path, model_type='random_forest')
        if success:
            model.save_model(model_dir)
    else:
        print(f"Data file not found: {data_path}")
