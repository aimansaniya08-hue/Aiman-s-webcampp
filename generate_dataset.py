import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Define regions in a city
regions = ['Downtown', 'Northside', 'Southside', 'Eastside', 'Westside', 
           'University Area', 'Industrial Zone', 'Shopping District', 
           'Residential A', 'Residential B', 'Park Area', 'Nightlife District']

# Coordinates for each region (approx)
coordinates = {
    'Downtown': (40.7128, -74.0060),
    'Northside': (40.7282, -74.0776),
    'Southside': (40.6782, -74.0418),
    'Eastside': (40.7130, -73.9870),
    'Westside': (40.7350, -74.0090),
    'University Area': (40.7580, -73.9850),
    'Industrial Zone': (40.6900, -74.0850),
    'Shopping District': (40.7480, -73.9860),
    'Residential A': (40.7200, -73.9950),
    'Residential B': (40.7050, -74.0150),
    'Park Area': (40.7820, -73.9650),
    'Nightlife District': (40.7600, -74.0000)
}

# Crime types relevant to women safety
crime_types = {
    'Harassment': 3,
    'Assault': 5,
    'Stalking': 4,
    'Eve Teasing': 2,
    'Robbery': 3,
    'Sexual Assault': 8,
    'Verbal Abuse': 2,
    'Indecent Exposure': 4
}

# Generate data
data = []
for region in regions:
    # Base safety score (higher = safer)
    base_safety = {
        'Downtown': 65, 'Northside': 45, 'Southside': 35, 'Eastside': 55,
        'Westside': 40, 'University Area': 70, 'Industrial Zone': 25,
        'Shopping District': 75, 'Residential A': 80, 'Residential B': 60,
        'Park Area': 50, 'Nightlife District': 30
    }[region]
    
    for _ in range(200):  # 200 crimes per region
        crime = np.random.choice(list(crime_types.keys()))
        severity = crime_types[crime]
        
        # Time of day (0-23)
        p = np.array([0.02]*8 + [0.08]*8 + [0.03]*8)
        p = p / p.sum()
        hour = np.random.choice(range(24), p=p)
        #hour = np.random.choice(range(24), p=[0.02]*8 + [0.08]*8 + [0.03]*8)  # More crimes at night
        
        # Is night? (6 PM - 6 AM)
        is_night = 1 if hour >= 18 or hour < 6 else 0
        
        # Weather conditions
        weather = np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], p=[0.6, 0.2, 0.15, 0.05])
        
        # Weekend or weekday
        is_weekend = 1 if np.random.random() > 0.7 else 0
        
        # Add noise to safety score
        actual_safety = base_safety + np.random.normal(0, 10)
        actual_safety = max(0, min(100, actual_safety))
        
        # Determine if area is safe (safety score > 50)
        is_safe = 1 if actual_safety > 50 else 0
        
        data.append({
            'region': region,
            'latitude': coordinates[region][0] + np.random.normal(0, 0.01),
            'longitude': coordinates[region][1] + np.random.normal(0, 0.01),
            'crime_type': crime,
            'severity': severity,
            'hour': hour,
            'is_night': is_night,
            'weather': weather,
            'is_weekend': is_weekend,
            'safety_score': actual_safety,
            'is_safe': is_safe
        })

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv('crime_dataset.csv', index=False)
print(f"✅ Dataset created with {len(df)} records")
print(f"\n📊 Dataset Info:")
print(f"Total crimes: {len(df)}")
print(f"Regions: {df['region'].nunique()}")
print(f"Safe areas: {df['is_safe'].sum()} crimes in safe zones")
print(f"Unsafe areas: {(~df['is_safe'].astype(bool)).sum()} crimes in unsafe zones")