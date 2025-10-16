import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime, timedelta
import joblib
import pickle
from pathlib import Path
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="SmartPredict AI | Predictive Maintenance Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glass morphism and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: white;
        color: black;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 5px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.20);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.2rem;
        margin: 0.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .alert-card {
        background: rgba(255, 87, 87, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 87, 87, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(255, 87, 87, 0.37);
    }
    
    .success-card {
        background: rgba(76, 175, 80, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(76, 175, 80, 0.37);
    }
    
    .confidence-card {
        background: rgba(106, 90, 205, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(106, 90, 205, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(106, 90, 205, 0.37);
    }
    
    .pause-card {
        background: rgba(255, 193, 7, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 193, 7, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(255, 193, 7, 0.37);
    }
    
    .scenario-active-card {
        background: rgba(33, 150, 243, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(33, 150, 243, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.37);
    }
    
    .maintenance-card {
        background: rgba(156, 39, 176, 0.2);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(156, 39, 176, 0.3);
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(156, 39, 176, 0.37);
    }
    
    .stButton button {
        background: white;
        color: black;
        border: #F06A2C 1pt solid;
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stop-button button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%) !important;
    }
    
    .pause-button button {
        background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%) !important;
        color: black !important;
    }
    
    h1, h2, h3 {
        color: #151413;
        text-align: center;
        font-weight: 600;
    }

    h1 {
        margin-bottom: 1rem;
    }
    
    .css-1d391kg {
        background-color: transparent;
    }
    
    .sensor-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4FC3F7;
        text-align: center;
    }
    
    .sensor-label {
        font-size: 1rem;
        color: rgba(94, 94, 94, 0.8);
        text-align: center;
    }
    
    .last-update {
        font-size: 0.8rem;
        color: rgba(94, 94, 94, 0.6);
        text-align: center;
        margin-top: 0.5rem;
    }
    
    .explanation-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4FC3F7;
    }
    
    .critical-item {
        border-left: 4px solid #F44336;
        background: rgba(244, 67, 54, 0.1);
    }
    
    .warning-item {
        border-left: 4px solid #FF9800;
        background: rgba(255, 152, 0, 0.1);
    }
    
    .feature-importance-bar {
        background: linear-gradient(90deg, #4FC3F7, #2196F3);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
        transition: all 0.3s ease;
    }
    
    .feature-importance-bar:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.4);
    }
    
    /* Hide the streamlit refresh spinner */
    .stStatusWidget {
        display: none;
    }
    
    .scenario-comparison {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .scenario-improvement {
        border-left: 4px solid #4CAF50;
        background: rgba(76, 175, 80, 0.1);
    }
    
    .scenario-worsening {
        border-left: 4px solid #F44336;
        background: rgba(244, 67, 54, 0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-running {
        background-color: #4CAF50;
        animation: pulse 2s infinite;
    }
    
    .status-paused {
        background-color: #FFC107;
    }
    
    .status-stopped {
        background-color: #F44336;
    }
    
    .status-scenario {
        background-color: #2196F3;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .simulation-stopped-container {
        text-align: center;
        padding: 3rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        margin: 1rem 0;
    }
    
    .maintenance-schedule {
        background: rgba(156, 39, 176, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #9C27B0;
    }
    
    .action-item {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-left: 4px solid #FF9800;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Robust model & scaler loader
# -----------------------
@st.cache_resource
def load_model_and_scaler():
    msgs = []
    model = None
    scaler = None

    # Try model files
    model_candidates = ["predictive_model.keras", "predictive_model.h5", "model.keras", "model.h5"]
    model_path = None
    for c in model_candidates:
        if Path(c).exists():
            model_path = c
            break

    if model_path is None:
        msgs.append(("info", "No model found. Using rule-based predictions."))
    else:
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            msgs.append(("success", f"Loaded model (compile=False): {model_path}"))
        except Exception as e:
            msgs.append(("warn", "Normal load failed"))
            

    # Try scaler
    scaler_path = Path("scaler.pkl")
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            msgs.append(("success", "Loaded scaler with joblib (scaler.pkl)"))
        except Exception as e1:
            try:
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                msgs.append(("success", "Loaded scaler with pickle (scaler.pkl)"))
            except Exception as e2:
                msgs.append(("error", f"Failed loading scaler: joblib error={e1}, pickle error={e2}"))
                scaler = None
    else:
        msgs.append(("info", "scaler.pkl not found. Using raw data."))

    return model, scaler, msgs

# -----------------------
# Feature settings
# -----------------------
REQUIRED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]", 
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

OPTIONAL_FEATURES = [
    "product_ID", "Type_H", "Type_L", "Type_M", "Power", "Temp_Ratio"
]

# -----------------------
# REAL-WORLD PREDICTION LOGIC WITH TIME-BASED UPDATES
# -----------------------

def calculate_rolling_failure_probability(history_data, window_size=10):
    """Calculate failure probability based on recent trend, not just current state"""
    if len(history_data) < window_size:
        return rule_based_prediction(history_data[-1]) if history_data else 0.05
    
    # Get recent data points
    recent_data = history_data[-window_size:]
    
    # Calculate probabilities for each point
    probabilities = []
    for data_point in recent_data:
        prob = rule_based_prediction(data_point)
        probabilities.append(prob)
    
    # Use weighted average (recent points matter more)
    weights = np.linspace(0.5, 1.5, len(probabilities))
    weighted_avg = np.average(probabilities, weights=weights)
    
    return weighted_avg

def should_trigger_maintenance(current_prob, historical_probs, threshold=0.3):
    """Only trigger maintenance if probability is consistently high"""
    if len(historical_probs) < 5:
        return current_prob > threshold
    
    # Check if current probability is significantly above recent average
    recent_avg = np.mean(historical_probs[-5:])
    return current_prob > max(threshold, recent_avg * 1.2)

def predict_failure_prob_from_sample(row: dict, model, scaler, history_data=None) -> tuple:
    """Predict failure probability and return explanation"""
    try:
        # Use rule-based prediction (more reliable for demo)
        if history_data and len(history_data) >= 5:
            # Use rolling average for more stable predictions
            prob = calculate_rolling_failure_probability(history_data, window_size=5)
        else:
            prob = rule_based_prediction(row)
        
        # Generate explanations and maintenance actions
        explanations, maintenance_actions = generate_explanations_and_actions(row, prob)
        return prob, explanations, maintenance_actions
        
    except Exception as e:
        # Fallback to basic prediction
        prob = 0.05
        explanations, maintenance_actions = generate_explanations_and_actions(row, prob)
        return prob, explanations, maintenance_actions

def rule_based_prediction(row: dict) -> float:
    """Rule-based failure prediction using domain knowledge"""
    failure_prob = 0.05  # Base failure probability
    
    # Temperature-based rules - MORE SENSITIVE THRESHOLDS
    air_temp = row.get("Air temperature [K]", 300)
    if air_temp > 315:  # Lowered threshold
        failure_prob += 0.3
    elif air_temp > 305:
        failure_prob += 0.15
    elif air_temp > 300:
        failure_prob += 0.05
        
    process_temp = row.get("Process temperature [K]", 305)
    if process_temp > 320:  # Lowered threshold
        failure_prob += 0.25
    elif process_temp > 310:
        failure_prob += 0.12
    elif process_temp > 305:
        failure_prob += 0.05
        
    # Speed and torque rules - MORE SENSITIVE THRESHOLDS
    rpm = row.get("Rotational speed [rpm]", 1500)
    if rpm > 2500:  # Lowered threshold
        failure_prob += 0.25
    elif rpm > 2200:
        failure_prob += 0.12
    elif rpm > 2000:
        failure_prob += 0.05
        
    torque = row.get("Torque [Nm]", 40)
    if torque > 70:  # Lowered threshold
        failure_prob += 0.20
    elif torque > 55:
        failure_prob += 0.10
    elif torque > 45:
        failure_prob += 0.05
        
    # Tool wear rules - MORE SENSITIVE THRESHOLDS
    tool_wear = row.get("Tool wear [min]", 0)
    if tool_wear > 180:  # Lowered threshold
        failure_prob += 0.25
    elif tool_wear > 120:
        failure_prob += 0.15
    elif tool_wear > 80:
        failure_prob += 0.05
        
    # Power calculation
    power = (rpm * torque) / 9.5488
    if power > 12000:  # Lowered threshold
        failure_prob += 0.15
    elif power > 8000:
        failure_prob += 0.08
    elif power > 5000:
        failure_prob += 0.03
    
    # Ensure probability is reasonable
    failure_prob = max(0.01, min(0.95, failure_prob))
    
    return failure_prob

def generate_explanations_and_actions(row: dict, probability: float) -> tuple:
    """Generate explanations and maintenance actions"""
    explanations = []
    maintenance_actions = []
    
    # Temperature analysis - ADJUSTED THRESHOLDS
    air_temp = row.get("Air temperature [K]", 300)
    if air_temp > 315:
        explanations.append(("Air Temperature", f"Critical air temperature ({air_temp:.1f}K)", "critical"))
        maintenance_actions.append("Immediate cooling system inspection required")
    elif air_temp > 305:
        explanations.append(("Air Temperature", f"High air temperature ({air_temp:.1f}K)", "warning"))
        maintenance_actions.append("Check cooling system and ventilation")
    
    process_temp = row.get("Process temperature [K]", 305)
    if process_temp > 320:
        explanations.append(("Process Temperature", f"Critical process temperature ({process_temp:.1f}K)", "critical"))
        maintenance_actions.append("Emergency process temperature reduction needed")
    elif process_temp > 310:
        explanations.append(("Process Temperature", f"High process temperature ({process_temp:.1f}K)", "warning"))
        maintenance_actions.append("Monitor process temperature closely")
    
    # Mechanical stress analysis - ADJUSTED THRESHOLDS
    rpm = row.get("Rotational speed [rpm]", 1500)
    if rpm > 2500:
        explanations.append(("Rotational Speed", f"Excessive rotational speed ({rpm:.0f}RPM)", "critical"))
        maintenance_actions.append("Reduce rotational speed immediately")
    elif rpm > 2200:
        explanations.append(("Rotational Speed", f"High rotational speed ({rpm:.0f}RPM)", "warning"))
        maintenance_actions.append("Consider reducing operational speed")
    
    torque = row.get("Torque [Nm]", 40)
    if torque > 70:
        explanations.append(("Torque", f"Critical torque level ({torque:.1f}Nm)", "critical"))
        maintenance_actions.append("Immediate torque reduction required")
    elif torque > 55:
        explanations.append(("Torque", f"High torque level ({torque:.1f}Nm)", "warning"))
        maintenance_actions.append("Monitor torque levels")
    
    # Tool wear analysis - ADJUSTED THRESHOLDS
    tool_wear = row.get("Tool wear [min]", 0)
    if tool_wear > 180:
        explanations.append(("Tool Wear", f"Critical tool wear ({tool_wear:.0f}min)", "critical"))
        maintenance_actions.append("Replace tool immediately")
    elif tool_wear > 120:
        explanations.append(("Tool Wear", f"Advanced tool wear ({tool_wear:.0f}min)", "warning"))
        maintenance_actions.append("Schedule tool replacement")
    
    # Power analysis - ADJUSTED THRESHOLDS
    power = (rpm * torque) / 9.5488
    if power > 12000:
        explanations.append(("Power Output", f"Critical power output ({power:.0f}W)", "critical"))
        maintenance_actions.append("Reduce operational load immediately")
    elif power > 8000:
        explanations.append(("Power Output", f"High power output ({power:.0f}W)", "warning"))
        maintenance_actions.append("Monitor power consumption")
    
    # Sort by severity
    explanations.sort(key=lambda x: 0 if x[2] == "critical" else 1)
    
    return explanations, maintenance_actions

def schedule_maintenance(machine_id, failure_prob, maintenance_actions, severity):
    """Schedule maintenance based on failure probability and severity"""
    maintenance_schedule = {
        'machine_id': machine_id,
        'scheduled_time': datetime.now(),
        'failure_probability': failure_prob,
        'severity': severity,
        'actions': maintenance_actions,
        'status': 'Scheduled'
    }
    
    # Determine maintenance urgency
    if failure_prob > 0.7:  # 70%
        maintenance_schedule['priority'] = 'EMERGENCY'
        maintenance_schedule['due_date'] = datetime.now() + timedelta(hours=2)
    elif failure_prob > 0.5:  # 50%
        maintenance_schedule['priority'] = 'HIGH'
        maintenance_schedule['due_date'] = datetime.now() + timedelta(hours=6)
    elif failure_prob > 0.3:  # 30%
        maintenance_schedule['priority'] = 'MEDIUM'
        maintenance_schedule['due_date'] = datetime.now() + timedelta(days=1)
    else:
        maintenance_schedule['priority'] = 'LOW'
        maintenance_schedule['due_date'] = datetime.now() + timedelta(days=3)
    
    return maintenance_schedule

# -----------------------
# DATA LOADING AND PROCESSING
# -----------------------
@st.cache_data
def load_sample_datasets():
    """Load sample datasets for demonstration"""
    try:
        # Create realistic sample data with more failure scenarios
        np.random.seed(42)
        sample_data = []
        
        for i in range(400):
            # Generate machine IDs
            machine_id = f"Machine_{np.random.randint(1, 6):02d}"
            
            # Generate realistic sensor data with more variation
            if np.random.random() < 0.3:  # 30% chance of problematic data
                # Generate data that will trigger high failure probability
                base_data = {
                    "machine_id": machine_id,
                    "Air temperature [K]": float(np.random.uniform(315, 340)),  # High temp
                    "Process temperature [K]": float(np.random.uniform(320, 340)),  # High temp
                    "Rotational speed [rpm]": float(np.random.uniform(2500, 3000)),  # High RPM
                    "Torque [Nm]": float(np.random.uniform(60, 90)),  # High torque
                    "Tool wear [min]": float(np.random.uniform(180, 300)),  # High wear
                    "product_ID": np.random.randint(0, 3),
                    "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 1440))
                }
            else:
                # Generate normal data
                base_data = {
                    "machine_id": machine_id,
                    "Air temperature [K]": float(np.random.uniform(295, 310)),
                    "Process temperature [K]": float(np.random.uniform(305, 315)),
                    "Rotational speed [rpm]": float(np.random.uniform(1200, 2200)),
                    "Torque [Nm]": float(np.random.uniform(30, 50)),
                    "Tool wear [min]": float(np.random.uniform(0, 120)),
                    "product_ID": np.random.randint(0, 3),
                    "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 1440))
                }
            
            sample_data.append(base_data)
        
        df = pd.DataFrame(sample_data)
        
        # Add engineered features
        df["Power"] = (df["Rotational speed [rpm]"] * df["Torque [Nm]"]) / 9.5488
        df["Temp_Ratio"] = df["Process temperature [K]"] / df["Air temperature [K]"]
        
        # Add one-hot encoded product types
        df["Type_H"] = (df["product_ID"] == 0).astype(int)
        df["Type_L"] = (df["product_ID"] == 1).astype(int)
        df["Type_M"] = (df["product_ID"] == 2).astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Error creating sample data: {e}")
        return pd.DataFrame()

def load_external_data(file):
    """Load data from uploaded file"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Validate required columns
        missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.info(f"Required columns: {REQUIRED_FEATURES}")
            return None
        
        # Add machine_id if not present
        if 'machine_id' not in df.columns:
            df['machine_id'] = 'Machine_01'
            st.info("Added 'machine_id' column with default value 'Machine_01'")
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()
            st.info("Added 'timestamp' column with current time")
        
        # Add missing optional features with default values
        for feature in OPTIONAL_FEATURES:
            if feature not in df.columns:
                if feature == 'product_ID':
                    df[feature] = 0
                elif feature in ['Type_H', 'Type_L', 'Type_M']:
                    df[feature] = 0
                elif feature == 'Power':
                    df[feature] = (df['Rotational speed [rpm]'] * df['Torque [Nm]']) / 9.5488
                elif feature == 'Temp_Ratio':
                    df[feature] = df['Process temperature [K]'] / df['Air temperature [K]']
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# -----------------------
# DASHBOARD CLASS WITH SMOOTH UPDATES
# -----------------------
class PredictiveMaintenanceDashboard:
    def __init__(self):
        self.model, self.scaler, self.load_msgs = load_model_and_scaler()
        self.model_loaded = self.model is not None
        
        # Initialize session state
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'simulation_started': False,
            'simulation_paused': False,
            'current_machine_index': 0,
            'dataset_loaded': False,
            'dataset': None,
            'maintenance_schedule': [],
            'last_update': None,
            'last_probability_update': None,
            'data_points': 0,
            'current_data': None,
            'failure_prob': 0.0,
            'failure_explanations': [],
            'maintenance_actions': [],
            'history_data': [],
            'alert_history': [],
            'selected_machine': 'Machine_01',
            'probability_history': []  # Track probability over time
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def load_data_section(self):
        """Data loading section in sidebar"""
        st.sidebar.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.sidebar.header("Data Source")
        
        data_source = st.sidebar.radio(
            "Select data source:",
            ["Sample Dataset", "Upload File"],
            key="data_source"
        )
        
        if data_source == "Sample Dataset":
            if st.sidebar.button("Load Sample Data", use_container_width=True):
                with st.spinner("Loading sample dataset..."):
                    st.session_state.dataset = load_sample_datasets()
                    if not st.session_state.dataset.empty:
                        st.session_state.dataset_loaded = True
                        st.session_state.current_machine_index = 0
                        st.session_state.data_points = 0
                        st.session_state.history_data = []
                        st.session_state.probability_history = []
                        st.sidebar.success(f"Loaded {len(st.session_state.dataset)} records")
                    else:
                        st.sidebar.error("Failed to load sample data")
        
        else:  # Upload File
            uploaded_file = st.sidebar.file_uploader(
                "Upload machine data",
                type=['csv', 'xlsx'],
                help="Upload CSV or Excel file with machine sensor data",
                key="file_uploader"
            )
            
            if uploaded_file is not None:
                with st.spinner("Processing uploaded file..."):
                    st.session_state.dataset = load_external_data(uploaded_file)
                    if st.session_state.dataset is not None and not st.session_state.dataset.empty:
                        st.session_state.dataset_loaded = True
                        st.session_state.current_machine_index = 0
                        st.session_state.data_points = 0
                        st.session_state.history_data = []
                        st.session_state.probability_history = []
                        st.sidebar.success(f"Loaded {len(st.session_state.dataset)} records")
                    else:
                        st.sidebar.error("Failed to process uploaded file")
        
        # Machine selection (only show if dataset is loaded)
        if st.session_state.dataset_loaded and st.session_state.dataset is not None:
            st.sidebar.markdown("---")
            st.sidebar.subheader("Machine Selection")
            
            # Get unique machine IDs safely
            if 'machine_id' in st.session_state.dataset.columns:
                machine_ids = st.session_state.dataset['machine_id'].unique()
                selected_machine = st.sidebar.selectbox(
                    "Select Machine to Monitor:",
                    machine_ids,
                    key="machine_select"
                )
                st.session_state.selected_machine = selected_machine
                
                # Filter data for selected machine
                machine_data = st.session_state.dataset[st.session_state.dataset['machine_id'] == selected_machine]
                st.sidebar.info(f"{len(machine_data)} records for {selected_machine}")
            else:
                st.sidebar.warning("No machine_id column found. Using all data.")
                st.session_state.selected_machine = "All Machines"
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

    def add_alert(self, alert_type, message, severity, data=None):
        """Add alert to history"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity,
            'data': data
        }
        st.session_state.alert_history.append(alert)
        
        # Keep only last 50 alerts
        if len(st.session_state.alert_history) > 50:
            st.session_state.alert_history = st.session_state.alert_history[-50:]

    def should_update_probability(self):
        """Check if it's time to update the failure probability (every 2 minutes)"""
        if st.session_state.last_probability_update is None:
            return True
        
        current_time = datetime.now()
        time_since_update = current_time - st.session_state.last_probability_update
        return time_since_update.total_seconds() >= 120  # 2 minutes

    def update_data(self):
        """Update the current data from dataset with time-based probability updates"""
        if (st.session_state.simulation_started and 
            not st.session_state.simulation_paused and 
            st.session_state.dataset_loaded and 
            st.session_state.dataset is not None):
            
            # Get next data point from dataset
            if st.session_state.current_machine_index < len(st.session_state.dataset):
                current_data = st.session_state.dataset.iloc[st.session_state.current_machine_index].to_dict()
                current_data['timestamp'] = datetime.now()
                
                st.session_state.current_data = current_data
                st.session_state.current_machine_index += 1
                st.session_state.data_points += 1
                st.session_state.last_update = datetime.now()
                
                # Add to history
                st.session_state.history_data.append(current_data)
                if len(st.session_state.history_data) > 100:
                    st.session_state.history_data = st.session_state.history_data[-100:]
                
                # Update failure probability only every 2 minutes
                if self.should_update_probability():
                    prob, explanations, maintenance_actions = predict_failure_prob_from_sample(
                        current_data, self.model, self.scaler, st.session_state.history_data
                    )
                    st.session_state.failure_prob = prob * 100  # Convert to percentage
                    st.session_state.failure_explanations = explanations
                    st.session_state.maintenance_actions = maintenance_actions
                    st.session_state.last_probability_update = datetime.now()
                    
                    # Track probability history
                    st.session_state.probability_history.append(prob)
                    if len(st.session_state.probability_history) > 50:
                        st.session_state.probability_history = st.session_state.probability_history[-50:]
                    
                    # Schedule maintenance if needed (with consistency check)
                    if should_trigger_maintenance(prob, st.session_state.probability_history):
                        severity = 'CRITICAL' if prob > 0.7 else 'HIGH' if prob > 0.5 else 'MEDIUM'
                        machine_id = current_data.get('machine_id', 'Unknown_Machine')
                        maintenance = schedule_maintenance(
                            machine_id,
                            prob,
                            maintenance_actions,
                            severity
                        )
                        st.session_state.maintenance_schedule.append(maintenance)
                        
                        # Add alert
                        self.add_alert(
                            'Maintenance Scheduled',
                            f"Scheduled {severity} priority maintenance for {machine_id}",
                            severity.lower(),
                            current_data
                        )

    def render_maintenance_section(self):
        """Render maintenance scheduling and alerts section"""
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Maintenance Management")
        
        # Show next probability update time
        if st.session_state.simulation_started and st.session_state.last_probability_update:
            next_update = st.session_state.last_probability_update + timedelta(seconds=120)
            time_remaining = (next_update - datetime.now()).total_seconds()
            if time_remaining > 0:
                st.caption(f"Next probability update in: {int(time_remaining)} seconds")
        
        # Current maintenance actions
        if st.session_state.maintenance_actions:
            st.markdown("### Recommended Actions")
            for i, action in enumerate(st.session_state.maintenance_actions):
                st.markdown(f'<div class="action-item">{action}</div>', unsafe_allow_html=True)
        
        # Maintenance schedule
        if st.session_state.maintenance_schedule:
            st.markdown("### Maintenance Schedule")
            for schedule in st.session_state.maintenance_schedule[-5:]:  # Show last 5
                priority_color = {
                    'EMERGENCY': '#F44336',
                    'HIGH': '#FF9800',
                    'MEDIUM': '#FFC107',
                    'LOW': '#4CAF50'
                }.get(schedule['priority'], '#9E9E9E')
                
                st.markdown(f'''
                <div class="maintenance-schedule">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong>{schedule['machine_id']}</strong>
                        <span style="background: {priority_color}; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; font-size: 0.8rem;">
                            {schedule['priority']}
                        </span>
                    </div>
                    <div>Due: {schedule['due_date'].strftime('%Y-%m-%d %H:%M')}</div>
                    <div>Failure Risk: {schedule['failure_probability']:.1%}</div>
                    <div style="margin-top: 0.5rem;">
                        <strong>Actions:</strong>
                        <ul style="margin: 0.2rem 0;">
                            {''.join([f'<li>{action}</li>' for action in schedule['actions'][:2]])}
                        </ul>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No maintenance scheduled. All systems operating normally.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_alert_history(self):
        """Render alert history"""
        if st.session_state.alert_history:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Alert History")
            
            for alert in st.session_state.alert_history[-10:]:  # Show last 10 alerts
                severity_color = {
                    'critical': '#F44336',
                    'warning': '#FF9800',
                    'info': '#2196F3'
                }.get(alert['severity'], '#9E9E9E')
                
                st.markdown(f'''
                <div style="background: rgba{(*tuple(int(severity_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)), 0.1)}; 
                            border-left: 4px solid {severity_color}; 
                            padding: 0.8rem; margin: 0.5rem 0; border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between;">
                        <strong>{alert['type']}</strong>
                        <small>{alert['timestamp'].strftime('%H:%M:%S')}</small>
                    </div>
                    <div>{alert['message']}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def create_sensor_gauges(self):
        """Create sensor gauge charts"""
        if st.session_state.current_data is None:
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp_gauge = self.create_live_sensor_gauge(
                st.session_state.current_data.get('Air temperature [K]', 300) - 273.15,
                "Air Temp (°C)",
                min_val=20,
                max_val=50,
                unit="°C"
            )
            st.plotly_chart(temp_gauge, use_container_width=True, key="temp_gauge")
        
        with col2:
            rpm_gauge = self.create_live_sensor_gauge(
                st.session_state.current_data.get('Rotational speed [rpm]', 1500),
                "Rotation Speed",
                min_val=1000,
                max_val=2500,
                unit="RPM"
            )
            st.plotly_chart(rpm_gauge, use_container_width=True, key="rpm_gauge")
        
        with col3:
            torque_gauge = self.create_live_sensor_gauge(
                st.session_state.current_data.get('Torque [Nm]', 40),
                "Torque",
                min_val=10,
                max_val=80,
                unit="Nm"
            )
            st.plotly_chart(torque_gauge, use_container_width=True, key="torque_gauge")

    def create_comprehensive_trends(self):
        """Create comprehensive trend charts with smooth updates"""
        if len(st.session_state.history_data) <= 1:
            self.render_placeholder_trends()
            return
        
        # Convert history to DataFrame for easier plotting
        history_df = pd.DataFrame(st.session_state.history_data)
        
        # Add calculated metrics if not present
        if 'Power' not in history_df.columns:
            history_df['Power'] = (history_df['Rotational speed [rpm]'] * history_df['Torque [Nm]']) / 9.5488
        if 'Temp_Ratio' not in history_df.columns:
            history_df['Temp_Ratio'] = history_df['Process temperature [K]'] / history_df['Air temperature [K]']
        
        # Convert temperatures to Celsius for better readability
        history_df['Air Temp (°C)'] = history_df['Air temperature [K]'] - 273.15
        history_df['Process Temp (°C)'] = history_df['Process temperature [K]'] - 273.15
        
        # Create subplots for comprehensive overview
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature Trends (°C)',
                'Rotational Speed (RPM)',
                'Torque (Nm)',
                'Tool Wear (minutes)',
                'Power Output (W)',
                'Temperature Ratio'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # 1. Temperature Trends
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Air Temp (°C)'],
                mode='lines',
                name='Air Temp',
                line=dict(color='#1f77b4', width=2),
                legendgroup='temp'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Process Temp (°C)'],
                mode='lines',
                name='Process Temp',
                line=dict(color='#ff7f0e', width=2),
                legendgroup='temp'
            ),
            row=1, col=1
        )
        
        # Add temperature thresholds
        fig.add_hline(y=42, line_dash="dash", line_color="red", row=1, col=1,
                     annotation_text="Critical Temp", annotation_position="top right")
        fig.add_hline(y=37, line_dash="dot", line_color="orange", row=1, col=1,
                     annotation_text="Warning Temp", annotation_position="top right")
        
        # 2. Rotational Speed
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Rotational speed [rpm]'],
                mode='lines',
                name='RPM',
                line=dict(color='#2ca02c', width=2),
                fill='tozeroy',
                fillcolor='rgba(44, 160, 44, 0.1)'
            ),
            row=1, col=2
        )
        
        # Add RPM thresholds
        fig.add_hline(y=2500, line_dash="dash", line_color="red", row=1, col=2,
                     annotation_text="Critical RPM", annotation_position="top right")
        fig.add_hline(y=2200, line_dash="dot", line_color="orange", row=1, col=2,
                     annotation_text="Warning RPM", annotation_position="top right")
        
        # 3. Torque
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Torque [Nm]'],
                mode='lines',
                name='Torque',
                line=dict(color='#d62728', width=2)
            ),
            row=2, col=1
        )
        
        # Add torque thresholds
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1,
                     annotation_text="Critical Torque", annotation_position="top right")
        fig.add_hline(y=55, line_dash="dot", line_color="orange", row=2, col=1,
                     annotation_text="Warning Torque", annotation_position="top right")
        
        # 4. Tool Wear
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Tool wear [min]'],
                mode='lines+markers',
                name='Tool Wear',
                line=dict(color='#9467bd', width=3),
                marker=dict(size=4)
            ),
            row=2, col=2
        )
        
        # Add tool wear thresholds
        fig.add_hline(y=180, line_dash="dash", line_color="red", row=2, col=2,
                     annotation_text="Critical Wear", annotation_position="top right")
        fig.add_hline(y=120, line_dash="dot", line_color="orange", row=2, col=2,
                     annotation_text="Warning Wear", annotation_position="top right")
        
        # 5. Power Output
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Power'],
                mode='lines',
                name='Power',
                line=dict(color='#8c564b', width=2),
                fill='tozeroy',
                fillcolor='rgba(140, 86, 75, 0.1)'
            ),
            row=3, col=1
        )
        
        # Add power thresholds
        fig.add_hline(y=12000, line_dash="dash", line_color="red", row=3, col=1,
                     annotation_text="Critical Power", annotation_position="top right")
        fig.add_hline(y=8000, line_dash="dot", line_color="orange", row=3, col=1,
                     annotation_text="Warning Power", annotation_position="top right")
        
        # 6. Temperature Ratio
        fig.add_trace(
            go.Scatter(
                x=history_df.index, 
                y=history_df['Temp_Ratio'],
                mode='lines',
                name='Temp Ratio',
                line=dict(color='#e377c2', width=2)
            ),
            row=3, col=2
        )
        
        # Add ratio thresholds
        fig.add_hline(y=1.08, line_dash="dash", line_color="red", row=3, col=2,
                     annotation_text="High Ratio", annotation_position="top right")
        fig.add_hline(y=1.02, line_dash="dot", line_color="orange", row=3, col=2,
                     annotation_text="Normal Ratio", annotation_position="top right")
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="RPM", row=1, col=2)
        fig.update_yaxes(title_text="Torque (Nm)", row=2, col=1)
        fig.update_yaxes(title_text="Wear (min)", row=2, col=2)
        fig.update_yaxes(title_text="Power (W)", row=3, col=1)
        fig.update_yaxes(title_text="Ratio", row=3, col=2)
        
        # Update x-axis labels
        for i in [1, 2, 3]:
            for j in [1, 2]:
                fig.update_xaxes(title_text="Time Sequence", row=i, col=j)
        
        # Use a consistent key for the chart to prevent re-rendering
        st.plotly_chart(fig, use_container_width=True, key="trend_charts")
        
        # Add summary statistics
        st.markdown("### Sensor Statistics Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_temp = history_df['Air Temp (°C)'].iloc[-1] if len(history_df) > 0 else 0
            max_temp = history_df['Air Temp (°C)'].max() if len(history_df) > 0 else 0
            st.metric(
                "Air Temperature", 
                f"{current_temp:.1f}°C",
                f"Max: {max_temp:.1f}°C"
            )
        
        with col2:
            current_rpm = history_df['Rotational speed [rpm]'].iloc[-1] if len(history_df) > 0 else 0
            max_rpm = history_df['Rotational speed [rpm]'].max() if len(history_df) > 0 else 0
            st.metric(
                "Rotational Speed", 
                f"{current_rpm:.0f} RPM",
                f"Max: {max_rpm:.0f} RPM"
            )
        
        with col3:
            current_torque = history_df['Torque [Nm]'].iloc[-1] if len(history_df) > 0 else 0
            max_torque = history_df['Torque [Nm]'].max() if len(history_df) > 0 else 0
            st.metric(
                "Torque", 
                f"{current_torque:.1f} Nm",
                f"Max: {max_torque:.1f} Nm"
            )
        
        with col4:
            current_wear = history_df['Tool wear [min]'].iloc[-1] if len(history_df) > 0 else 0
            max_wear = history_df['Tool wear [min]'].max() if len(history_df) > 0 else 0
            st.metric(
                "Tool Wear", 
                f"{current_wear:.0f} min",
                f"Max: {max_wear:.0f} min"
            )

    def render_placeholder_trends(self):
        """Show placeholder trend charts when no data is available"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature Trends (°C)',
                'Rotational Speed (RPM)',
                'Torque (Nm)',
                'Tool Wear (minutes)',
                'Power Output (W)',
                'Temperature Ratio'
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )
        
        # Add empty traces with instructional text
        for i in range(1, 4):
            for j in range(1, 3):
                fig.add_annotation(
                    text="Start simulation to see data",
                    xref=f"x{i}{j}", yref=f"y{i}{j}",
                    x=0.5, y=0.5, xanchor="center", yanchor="middle",
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key="placeholder_trends")

    def render_correlation_analysis(self):
        """Render correlation analysis between sensors"""
        if len(st.session_state.history_data) > 10:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Sensor Correlation Analysis")
            
            history_df = pd.DataFrame(st.session_state.history_data)
            
            # Calculate correlations
            correlation_data = history_df[[
                'Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
            ]].corr()
            
            # Create correlation heatmap
            fig = px.imshow(
                correlation_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Sensor Correlation Matrix"
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="correlation_chart")
            
            # Add correlation insights
            st.markdown("### Correlation Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                temp_corr = correlation_data.loc['Air temperature [K]', 'Process temperature [K]']
                st.info(f"**Temperature Correlation**: {temp_corr:.2f}")
                
                if temp_corr > 0.7:
                    st.write("✓ Strong correlation between air and process temperatures")
                elif temp_corr > 0.3:
                    st.write("○ Moderate temperature correlation")
                else:
                    st.write("✗ Weak temperature relationship")
            
            with col2:
                power_torque_corr = correlation_data.loc['Rotational speed [rpm]', 'Torque [Nm]']
                st.info(f"**Power Factors Correlation**: {power_torque_corr:.2f}")
                
                if abs(power_torque_corr) > 0.5:
                    st.write("✓ Strong relationship between speed and torque")
                else:
                    st.write("○ Independent power factors")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def render_historical_trends(self):
        """Render longer-term historical trends"""
        if len(st.session_state.history_data) > 20:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("Historical Performance Trends")
            
            history_df = pd.DataFrame(st.session_state.history_data)
            
            # Create performance indicators over time
            fig = go.Figure()
            
            # Add efficiency indicator (inverse of temperature ratio)
            efficiency = 1 / (history_df['Process temperature [K]'] / history_df['Air temperature [K]'])
            fig.add_trace(go.Scatter(
                x=history_df.index,
                y=efficiency,
                mode='lines',
                name='Thermal Efficiency',
                line=dict(color='#00b894', width=3)
            ))
            
            # Add wear rate (derivative of tool wear)
            if len(history_df) > 2:
                wear_rate = history_df['Tool wear [min]'].diff().fillna(0)
                fig.add_trace(go.Scatter(
                    x=history_df.index,
                    y=wear_rate,
                    mode='lines',
                    name='Wear Rate (min/cycle)',
                    line=dict(color='#e17055', width=2),
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title="Machine Performance Indicators",
                yaxis=dict(title="Thermal Efficiency"),
                yaxis2=dict(
                    title="Wear Rate (min/cycle)",
                    overlaying='y',
                    side='right'
                ),
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True, key="historical_chart")
            
            # Performance summary
            st.markdown("### Performance Summary")
            avg_efficiency = efficiency.mean()
            avg_wear_rate = wear_rate.mean() if len(history_df) > 2 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if avg_efficiency > 0.95:
                    st.success(f"**Efficiency**: {avg_efficiency:.2f} (Excellent)")
                elif avg_efficiency > 0.90:
                    st.warning(f"**Efficiency**: {avg_efficiency:.2f} (Good)")
                else:
                    st.error(f"**Efficiency**: {avg_efficiency:.2f} (Poor)")
            
            with col2:
                if avg_wear_rate < 0.5:
                    st.success(f"**Wear Rate**: {avg_wear_rate:.2f} min/cycle (Low)")
                elif avg_wear_rate < 1.0:
                    st.warning(f"**Wear Rate**: {avg_wear_rate:.2f} min/cycle (Normal)")
                else:
                    st.error(f"**Wear Rate**: {avg_wear_rate:.2f} min/cycle (High)")
            
            with col3:
                total_cycles = len(history_df)
                st.info(f"**Total Cycles**: {total_cycles}")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def run_dashboard(self):
       # Header
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.title("SmartPredict AI Dashboard")            
        with col2:
            st.metric("Data Points", f"{st.session_state.data_points}")
            
        with col3:
            status_text = "RUNNING" if st.session_state.simulation_started else "STOPPED"
            status_color = "#4CAF50" if st.session_state.simulation_started else "#F44336"
            st.markdown(
                f'<div style="display: flex; align-items: center; justify-content: center; padding: 0.5rem; border-radius: 10px; background: {status_color}20; border: 1px solid {status_color}40;">'
                f'<span class="status-indicator status-running"></span>'
                f'<strong>{status_text}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )
        
            

        # Data loading section
        self.load_data_section()

        # Simulation controls
        st.sidebar.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.sidebar.header("Simulation Controls")
        
        # Show model load status
        st.sidebar.subheader("Model Status")
        for msg_type, msg_text in self.load_msgs:
            if msg_type == "success":
                st.sidebar.success(msg_text)
            elif msg_type == "warn":
                st.sidebar.warning(msg_text)
            elif msg_type == "error":
                st.sidebar.error(msg_text)
            else:
                st.sidebar.info(msg_text)
        
        # Start/Stop buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Start Simulation", use_container_width=True, key="start_btn"):
                if st.session_state.dataset_loaded:
                    st.session_state.simulation_started = True
                    st.session_state.simulation_paused = False
                    st.session_state.data_points = 0
                    st.session_state.current_machine_index = 0
                    st.session_state.history_data = []
                    st.session_state.maintenance_schedule = []
                    st.session_state.alert_history = []
                    st.session_state.probability_history = []
                    st.session_state.last_probability_update = None
                else:
                    st.sidebar.error("Please load a dataset first")
                
        with col2:
            if st.button("Stop Simulation", use_container_width=True, key="stop_btn"):
                st.session_state.simulation_started = False
                st.session_state.simulation_paused = False
        
        if st.session_state.dataset_loaded:
            simulation_speed = st.sidebar.slider("Data Speed (records/sec)", 1, 10, 3, key="speed")
            progress = st.session_state.current_machine_index / len(st.session_state.dataset) if st.session_state.dataset_loaded else 0
            st.sidebar.progress(progress, text=f"Progress: {st.session_state.current_machine_index}/{len(st.session_state.dataset)}")
            
            # Show probability update info
            if st.session_state.last_probability_update:
                st.sidebar.caption(f"Last probability update: {st.session_state.last_probability_update.strftime('%H:%M:%S')}")
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Main dashboard layout
        if st.session_state.dataset_loaded:
            # Create layout columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Sensor gauges and trends
                if st.session_state.current_data is not None:
                    # Sensor gauges with smooth updates
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Live Sensor Readings")
                    self.create_sensor_gauges()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Comprehensive trends with smooth updates
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Comprehensive Sensor Trends")
                    self.create_comprehensive_trends()
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add additional analysis sections
                    if len(st.session_state.history_data) > 10:
                        self.render_correlation_analysis()
                        self.render_historical_trends()
                        
                else:
                    self.render_stopped_view()
            
            with col2:
                # Failure probability and explanations
                if st.session_state.current_data is not None:
                    # Failure probability with smooth updates
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.subheader("Failure Risk Assessment")
                    
                    # Show last update time
                    if st.session_state.last_probability_update:
                        st.caption(f"Last updated: {st.session_state.last_probability_update.strftime('%H:%M:%S')}")
                    
                    failure_gauge = self.create_gauge_chart(
                        st.session_state.failure_prob, 
                        "Failure Probability", 
                        min_val=0, 
                        max_val=100, 
                        threshold=50,
                        unit="%"
                    )
                    st.plotly_chart(failure_gauge, use_container_width=True, key="failure_gauge")
                    
                    if st.session_state.failure_prob > 70:
                        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                        st.error(f"CRITICAL: High failure risk ({st.session_state.failure_prob:.1f}%)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif st.session_state.failure_prob > 50:
                        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                        st.warning(f"WARNING: Elevated failure risk ({st.session_state.failure_prob:.1f}%)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-card">', unsafe_allow_html=True)
                        st.success(f"NORMAL: Low failure risk ({st.session_state.failure_prob:.1f}%)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Explanations
                    self.render_explanation_section()
                
                # Maintenance section
                self.render_maintenance_section()
                
            # Alert history at the bottom
            self.render_alert_history()
                
        else:
            st.info("Please load a dataset from the sidebar to begin monitoring.")
            

        # Update data continuously when simulation is running
        if st.session_state.simulation_started and not st.session_state.simulation_paused:
            self.update_data()
            time.sleep(1.0 / st.session_state.get('speed', 3))
            st.rerun()

    def render_explanation_section(self):
        """Render failure explanation section"""
        if not st.session_state.failure_explanations:
            return
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Risk Factor Analysis")
        
        for factor, description, severity in st.session_state.failure_explanations:
            css_class = "critical-item" if severity == "critical" else "warning-item"
            icon_color = "#F44336" if severity == "critical" else "#FF9800"
            
            st.markdown(f'''
            <div class="{css_class}">
                <div style="display: flex; align-items: center; margin-bottom: 0.3rem;">
                    <span style="font-size: 1.2rem; margin-right: 0.5rem; color: {icon_color}">●</span>
                    <strong>{factor}</strong>
                </div>
                <div style="color: rgba(94, 94, 94, 0.9);">{description}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def create_gauge_chart(self, value, title, min_val=0, max_val=100, threshold=70, unit=""):
        """Create a beautiful gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            number = {'suffix': unit, 'font': {'size': 20, 'color': '#5E5E5E'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 16, 'color': '#5E5E5E'}},
            gauge = {
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#4FC3F7", 'thickness': 0.75},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 2,
                'bordercolor': "rgba(94, 94, 94, 0.3)",
                'steps': [
                    {'range': [min_val, threshold*0.7], 'color': 'rgba(76, 175, 80, 0.6)'},
                    {'range': [threshold*0.7, threshold], 'color': 'rgba(255, 152, 0, 0.6)'},
                    {'range': [threshold, max_val], 'color': 'rgba(244, 67, 54, 0.6)'}],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.8,
                    'value': threshold}}))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#5E5E5E", 'family': "Inter"},
            height=250,
            margin=dict(l=30, r=30, t=50, b=20)
        )
        return fig

    def create_live_sensor_gauge(self, value, title, min_val, max_val, unit=""):
        """Create a gauge for live sensor readings"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            number = {'suffix': unit, 'font': {'size': 18, 'color': '#5E5E5E'}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title, 'font': {'size': 14, 'color': '#5E5E5E'}},
            gauge = {
                'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#5E5E5E"},
                'bar': {'color': "#4FC3F7"},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 2,
                'bordercolor': "rgba(94, 94, 94, 0.3)"}))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': "#5E5E5E", 'family': "Inter"},
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def render_stopped_view(self):
        """Render the simulation stopped view"""
        st.markdown("""
        <div class="simulation-stopped-container">
            <h2>Simulation Ready</h2>
            <p style="font-size: 1.2rem; color: rgba(94,94,94,0.8); margin: 1rem 0;">
                Click 'Start Simulation' in the sidebar to begin monitoring
            </p>
        </div>
        """, unsafe_allow_html=True)

# Run the dashboard
if __name__ == "__main__":
    dashboard = PredictiveMaintenanceDashboard()
    dashboard.run_dashboard()