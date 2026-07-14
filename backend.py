import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import folium
import joblib
import os

class WomenSafetyModel:
    def __init__(self):
        self.model = None
        self.le_region = None
        self.le_weather = None
        self.region_features = None
        self.df = None
        self.accuracy = None
        
    def generate_sample_data(self, n_samples=2400):
        """Generate synthetic dataset for demonstration"""
        np.random.seed(42)
        
        regions = ['Downtown', 'Northside', 'Southside', 'Eastside', 'Westside', 
                   'Suburb A', 'Suburb B', 'City Center', 'Old Town', 'Industrial Area']
        crime_types = ['Theft', 'Assault', 'Harassment', 'Robbery', 'Vandalism', 
                      'Stalking', 'Fraud', 'Burglary']
        weather_types = ['Clear', 'Rain', 'Fog', 'Snow']
        
        # Region coordinates (lat, lon)
        region_coords = {
            'Downtown': (40.7128, -74.0060),
            'Northside': (40.7282, -73.9942),
            'Southside': (40.6989, -74.0132),
            'Eastside': (40.7185, -73.9910),
            'Westside': (40.7053, -74.0168),
            'Suburb A': (40.6850, -74.0450),
            'Suburb B': (40.7500, -73.9800),
            'City Center': (40.7200, -74.0000),
            'Old Town': (40.7300, -73.9950),
            'Industrial Area': (40.6900, -74.0300)
        }
        
        data = []
        for _ in range(n_samples):
            region = np.random.choice(regions)
            lat, lon = region_coords[region]
            
            # Time-based severity (higher at night)
            hour = np.random.randint(0, 24)
            is_night = 1 if hour >= 18 or hour <= 6 else 0
            is_weekend = np.random.choice([0, 1])
            
            # Weather impact
            weather = np.random.choice(weather_types)
            weather_severity = {'Clear': 1, 'Rain': 1.5, 'Fog': 1.3, 'Snow': 1.2}
            
            # Crime type and severity
            crime_type = np.random.choice(crime_types)
            base_severity = {
                'Theft': 3, 'Assault': 7, 'Harassment': 5, 'Robbery': 8,
                'Vandalism': 2, 'Stalking': 6, 'Fraud': 4, 'Burglary': 6
            }
            
            severity = base_severity[crime_type] * weather_severity[weather]
            if is_night:
                severity *= 1.5
            if is_weekend:
                severity *= 1.2
                
            severity = min(10, severity)
            
            # Safety score (inverse of severity)
            safety_score = max(0, 100 - (severity * 10))
            
            # Add some randomness
            safety_score += np.random.normal(0, 5)
            safety_score = np.clip(safety_score, 0, 100)
            
            # Determine if safe (safety_score > 60)
            is_safe = 1 if safety_score > 60 else 0
            
            data.append({
                'region': region,
                'latitude': lat,
                'longitude': lon,
                'crime_type': crime_type,
                'severity': severity,
                'hour': hour,
                'is_night': is_night,
                'is_weekend': is_weekend,
                'weather': weather,
                'safety_score': safety_score,
                'is_safe': is_safe
            })
        
        return pd.DataFrame(data)
    
    def load_or_generate_data(self, filepath='crime_dataset.csv'):
        """Load existing data or generate new one"""
        if os.path.exists(filepath):
            self.df = pd.read_csv(filepath)
        else:
            self.df = self.generate_sample_data()
            self.df.to_csv(filepath, index=False)
        return self.df
    
    def train_model(self):
        """Train the Random Forest model"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_or_generate_data first.")
        
        # Feature engineering
        features = pd.DataFrame()
        features['severity'] = self.df['severity']
        features['hour'] = self.df['hour']
        features['is_night'] = self.df['is_night']
        features['is_weekend'] = self.df['is_weekend']
        
        # Encode categorical variables
        self.le_weather = LabelEncoder()
        features['weather'] = self.le_weather.fit_transform(self.df['weather'])
        
        # Region encoding
        self.le_region = LabelEncoder()
        region_encoded = self.le_region.fit_transform(self.df['region'])
        features['region'] = region_encoded
        
        # Aggregated features per region
        self.region_features = self.df.groupby('region').agg({
            'severity': 'mean',
            'is_night': 'mean',
            'safety_score': 'mean'
        }).reset_index()
        
        target = self.df['is_safe']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.classification_report = classification_report(y_test, y_pred, output_dict=True)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': self.accuracy,
            'classification_report': self.classification_report,
            'confusion_matrix': self.confusion_matrix,
            'feature_importance': self.model.feature_importances_
        }
    
    def predict_safety(self, region, hour, weather, is_weekend):
        """Predict safety for given parameters"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Prepare input
        input_data = pd.DataFrame({
            'severity': [5],  # default medium severity
            'hour': [hour],
            'is_night': [1 if hour >= 18 or hour <= 6 else 0],
            'is_weekend': [is_weekend],
            'weather': [self.le_weather.transform([weather])[0]],
            'region': [self.le_region.transform([region])[0]]
        })
        
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0]
        
        return {
            'is_safe': bool(prediction),
            'confidence': float(probability[1] if prediction == 1 else probability[0]),
            'probability_safe': float(probability[1]),
            'probability_unsafe': float(probability[0])
        }
    
    def get_region_safety_stats(self):
        """Get safety statistics by region"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        region_stats = self.df.groupby('region').agg({
            'safety_score': 'mean',
            'is_safe': 'mean',
            'latitude': 'mean',
            'longitude': 'mean',
            'severity': 'mean'
        }).reset_index()
        
        region_stats['safe_percentage'] = region_stats['is_safe'] * 100
        region_stats['safety_level'] = region_stats['safe_percentage'].apply(
            lambda x: 'Safe' if x > 60 else ('Moderate' if x > 40 else 'Unsafe')
        )
        
        return region_stats
    
    def get_hourly_stats(self):
        """Get hourly crime statistics"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        hourly = self.df.groupby('hour').agg({
            'severity': 'mean',
            'is_safe': 'mean',
            'safety_score': 'mean'
        }).reset_index()
        
        return hourly
    
    def get_crime_stats(self):
        """Get crime type statistics"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        crime_stats = self.df.groupby('crime_type').agg({
            'severity': 'mean',
            'is_safe': 'mean'
        }).reset_index()
        
        return crime_stats
    
    def get_weather_impact(self):
        """Get weather impact on safety"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        weather_impact = self.df.groupby('weather').agg({
            'safety_score': 'mean',
            'severity': 'mean'
        }).reset_index()
        
        return weather_impact
    
    def create_safety_map(self):
        """Create an interactive Folium map"""
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        region_stats = self.get_region_safety_stats()
        
        # Create map centered on mean coordinates
        center_lat = region_stats['latitude'].mean()
        center_lon = region_stats['longitude'].mean()
        
        safety_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add markers for each region
        for _, row in region_stats.iterrows():
            safety_percent = row['safe_percentage']
            
            if safety_percent > 60:
                color = 'green'
                icon = 'check-circle'
                popup_color = 'green'
            elif safety_percent > 40:
                color = 'orange'
                icon = 'exclamation-triangle'
                popup_color = 'orange'
            else:
                color = 'red'
                icon = 'times-circle'
                popup_color = 'red'
            
            popup_text = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4 style="color: {popup_color};">{row['region']}</h4>
                <b>Safety Score:</b> {row['safety_score']:.1f}/100<br>
                <b>Safe Zone Probability:</b> {safety_percent:.1f}%<br>
                <b>Safety Level:</b> {row['safety_level']}<br>
                <b>Avg Severity:</b> {row['severity']:.1f}/10
            </div>
            """
            
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                tooltip=f"{row['region']} - {row['safety_level']}"
            ).add_to(safety_map)
        
        return safety_map
    
    def save_model(self, filepath='safety_model.pkl'):
        """Save the trained model and encoders"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        model_data = {
            'model': self.model,
            'le_region': self.le_region,
            'le_weather': self.le_weather,
            'region_features': self.region_features,
            'accuracy': self.accuracy
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath='safety_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.le_region = model_data['le_region']
            self.le_weather = model_data['le_weather']
            self.region_features = model_data['region_features']
            self.accuracy = model_data['accuracy']
            return True
        return False

# Singleton instance
safety_model = WomenSafetyModel()