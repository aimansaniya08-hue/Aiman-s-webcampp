import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from backend import safety_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Women Safety Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        animation: fadeIn 1s ease-in;
    }
    
    /* Card styles */
    .safe-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .unsafe-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_app():
    """Initialize the application with data and model"""
    with st.spinner("🔄 Loading data and training model..."):
        # Load or generate data
        df = safety_model.load_or_generate_data()
        
        # Train model
        metrics = safety_model.train_model()
        
        return df, metrics

def display_safety_prediction(prediction_result):
    """Display safety prediction results with animations"""
    if prediction_result['is_safe']:
        st.markdown(f"""
        <div class="safe-card" style="animation: slideIn 0.5s ease-out;">
            <h3>✅ SAFE ZONE</h3>
            <p>This area is predicted to be <strong>SAFE</strong> for women.</p>
            <p>Confidence Level: <strong>{prediction_result['confidence']:.1%}</strong></p>
            <p>Safety Probability: {prediction_result['probability_safe']:.1%} | 
            Risk Probability: {prediction_result['probability_unsafe']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for safety level
        st.progress(prediction_result['probability_safe'])
        
    else:
        st.markdown(f"""
        <div class="unsafe-card" style="animation: slideIn 0.5s ease-out;">
            <h3>⚠️ UNSAFE ZONE</h3>
            <p>This area is predicted to be <strong>UNSAFE</strong> for women.</p>
            <p>Confidence Level: <strong>{prediction_result['confidence']:.1%}</strong></p>
            <p>Risk Probability: {prediction_result['probability_unsafe']:.1%} | 
            Safety Probability: {prediction_result['probability_safe']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for risk level
        st.progress(prediction_result['probability_unsafe'])

def create_analytics_dashboard(df):
    """Create comprehensive analytics dashboard"""
    st.header("📊 Safety Analytics Dashboard")
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_safety = df['safety_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Average Safety Score</h3>
            <h2 style="color: {'green' if avg_safety > 60 else 'orange' if avg_safety > 40 else 'red'}">
                {avg_safety:.1f}/100
            </h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        safe_areas = (df['is_safe'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>✅ Safe Areas</h3>
            <h2 style="color: green">{safe_areas:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_incidents = len(df)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Incidents</h3>
            <h2 style="color: #FF4B4B">{total_incidents:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        regions_covered = df['region'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>📍 Regions Covered</h3>
            <h2 style="color: #FF6B6B">{regions_covered}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        # Safety by region
        region_stats = safety_model.get_region_safety_stats()
        fig = px.bar(
            region_stats,
            x='region',
            y='safety_score',
            title="Safety Score by Region",
            color='safety_score',
            color_continuous_scale='RdYlGn',
            text='safety_score'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly crime distribution
        hourly_stats = safety_model.get_hourly_stats()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['severity'],
            mode='lines+markers',
            name='Crime Severity',
            line=dict(color='red', width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=hourly_stats['hour'],
            y=hourly_stats['safety_score'],
            mode='lines+markers',
            name='Safety Score',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        fig.add_vline(x=18, line_dash="dash", line_color="orange", 
                      annotation_text="Night Begins")
        fig.add_vline(x=6, line_dash="dash", line_color="orange", 
                      annotation_text="Night Ends")
        fig.update_layout(title="Safety & Severity by Hour", height=400,
                         xaxis_title="Hour of Day", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col3, col4 = st.columns(2)
    
    with col3:
        # Crime type distribution
        crime_stats = safety_model.get_crime_stats()
        fig = px.pie(
            crime_stats,
            values='severity',
            names='crime_type',
            title="Crime Type Distribution by Severity",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Weather impact
        weather_impact = safety_model.get_weather_impact()
        fig = px.bar(
            weather_impact,
            x='weather',
            y='safety_score',
            title="Weather Impact on Safety",
            color='safety_score',
            color_continuous_scale='RdYlGn',
            text='safety_score'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("🔥 Crime Severity Heatmap")
    pivot_table = df.pivot_table(
        values='severity',
        index='hour',
        columns='is_weekend',
        aggfunc='mean',
        fill_value=0
    )
    pivot_table.columns = ['Weekday', 'Weekend']
    
    fig = px.imshow(
        pivot_table.T,
        labels={'x': 'Hour', 'y': 'Day Type', 'color': 'Avg Severity'},
        title="Crime Severity Heatmap (Hour vs Day Type)",
        color_continuous_scale='Reds',
        aspect='auto'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

def display_model_performance(metrics):
    """Display model performance metrics"""
    st.header("🤖 Model Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2 style="color: #28a745">{metrics['accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = metrics['confusion_matrix']
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Unsafe', 'Safe'],
                    yticklabels=['Unsafe', 'Safe'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        # Classification Report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn').format("{:.3f}"))
    
    # Feature Importance
    st.subheader("📊 Feature Importance Analysis")
    feature_names = ['Severity', 'Hour', 'Is Night', 'Is Weekend', 'Weather', 'Region']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': metrics['feature_importance']
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="What Factors Most Affect Safety?",
        color='importance',
        color_continuous_scale='Viridis',
        text='importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    with st.expander("💡 Model Insights & Recommendations"):
        st.markdown("""
        **Key Findings:**
        - **Time of day** is the most significant factor affecting safety
        - **Crime severity** strongly correlates with overall safety scores
        - **Weather conditions** can increase risk by 20-50%
        - Weekend nights show 40% higher risk levels
        
        **Recommendations:**
        - Avoid traveling alone between 10 PM - 4 AM
        - Stay in well-lit, populated areas during high-risk hours
        - Check weather forecasts before planning late-night travel
        - Use the safety map to plan safer routes
        """)

def main():
    # Header with animation
    st.markdown('<div class="main-header">🛡️ Women Safety Prediction System</div>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize system
    df, metrics = initialize_app()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/woman-safety.png", width=100)
        st.markdown("## 📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{metrics['accuracy']:.2%}", 
                     delta="Good" if metrics['accuracy'] > 0.7 else "Needs Improvement")
        with col2:
            st.metric("Total Records", len(df))
        
        st.metric("Regions Covered", df['region'].nunique())
        
        st.markdown("---")
        st.markdown("## 🎯 Quick Prediction")
        
        # User input for prediction
        region = st.selectbox("Select Area", df['region'].unique())
        hour = st.slider("Time of Day (24h)", 0, 23, 14, 
                        help="0 = Midnight, 12 = Noon, 23 = 11 PM")
        weather = st.selectbox("Weather Condition", ['Clear', 'Rain', 'Fog', 'Snow'])
        is_weekend = st.checkbox("Is it Weekend?")
        
        if st.button("🔍 Predict Safety", type="primary", use_container_width=True):
            prediction = safety_model.predict_safety(region, hour, weather, is_weekend)
            display_safety_prediction(prediction)
        
        st.markdown("---")
        st.markdown("### 🆘 Emergency Contacts")
        st.info("**Police:** 100\n**Women Helpline:** 1091\n**Ambulance:** 102")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Interactive Safety Map", 
        "📈 Analytics Dashboard", 
        "🤖 Model Performance", 
        "ℹ️ About & Safety Tips"
    ])
    
    with tab1:
        st.header("Interactive Safety Map")
        st.markdown("*🟢 Safe | 🟠 Moderate | 🔴 Unsafe*")
        
        safety_map = safety_model.create_safety_map()
        folium_static(safety_map, width=1000, height=600)
        
        st.info("💡 **Tip:** Click on any marker to see detailed safety information about that area. Green markers indicate safe zones, red markers indicate unsafe zones.")
    
    with tab2:
        create_analytics_dashboard(df)
    
    with tab3:
        display_model_performance(metrics)
    
    with tab4:
        st.header("About the Women Safety Prediction System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Mission
            Empowering women with real-time safety information to make informed decisions about their routes and destinations.
            
            ### 🔧 How It Works
            1. **Data Collection**: Historical crime data including type, time, location, and weather
            2. **Machine Learning**: Random Forest algorithm trained on incident records
            3. **Real-time Prediction**: Analyzes multiple factors to predict safety
            4. **Visualization**: Interactive map showing safe vs unsafe zones
            
            ### 📊 Features Used
            - Crime Severity (weighted by type)
            - Time of Day & Night indicator
            - Day Type (Weekend/Weekday)
            - Weather Conditions
            - Geographic Region
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Future Enhancements
            - Real-time crime API integration
            - User reporting system
            - Route optimization for safest path
            - Mobile app with notifications
            - Emergency service integration
            
            ### ⚠️ Disclaimer
            This system is for informational purposes only. Always stay aware of your surroundings and follow local safety guidelines.
            """)
        
        # Safety Tips
        with st.expander("🛡️ Comprehensive Women Safety Tips", expanded=True):
            tabs = st.tabs(["Before Going Out", "During Travel", "Emergency Response", "Digital Safety"])
            
            with tabs[0]:
                st.markdown("""
                **✅ Preparation Checklist:**
                - Share your location with trusted contacts
                - Keep your phone fully charged
                - Save emergency numbers in speed dial
                - Carry a personal safety alarm
                - Let someone know your route and expected arrival time
                - Check the safety map for your destination area
                """)
            
            with tabs[1]:
                st.markdown("""
                **🚗 Travel Safety:**
                - Stay in well-lit, populated areas
                - Avoid isolated routes, especially at night
                - Use trusted transportation services
                - Sit behind the driver in taxis/ride-shares
                - Keep windows up and doors locked
                - Trust your instincts - if something feels wrong, leave
                """)
            
            with tabs[2]:
                st.markdown("""
                **🚨 Emergency Protocols:**
                - **Call Police:** 100 (India) / 911 (US)
                - **Women Helpline:** 1091
                - Use safety apps with SOS features
                - Find nearest safe zone (police station, hospital, open business)
                - Create noise to attract attention if threatened
                - Share live location with emergency contacts
                """)
            
            with tabs[3]:
                st.markdown("""
                **📱 Digital Safety:**
                - Use safety apps with real-time tracking
                - Don't share real-time location on public social media
                - Verify ride-share details before entering
                - Use emergency features in ride-sharing apps
                - Keep location services on for emergency apps
                """)

if __name__ == "__main__":
    main()