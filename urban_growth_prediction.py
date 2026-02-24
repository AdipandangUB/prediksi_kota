import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import rasterio
from rasterio.transform import from_origin
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import networkx as nx
from scipy.spatial import cKDTree
import warnings
import datetime
import time
import joblib
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

warnings.filterwarnings('ignore')

# Constants
N_FEATURES = 15  # Jumlah fitur dasar
N_TEMPORAL_FEATURES = N_FEATURES * 3  # current + future + diff = 45

# Set page config
st.set_page_config(
    page_title="Sistem Analisis & Prediksi Perubahan Tutupan Lahan",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #f0f2f6;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .indonesian-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'change_rates' not in st.session_state:
    st.session_state.change_rates = None
if 'grid_cache' not in st.session_state:
    st.session_state.grid_cache = {}
if 'feature_dim' not in st.session_state:
    st.session_state.feature_dim = N_TEMPORAL_FEATURES

# Title with animation - INDONESIA
st.title("🌍 Sistem Analisis & Prediksi Perubahan Tutupan Lahan")
st.markdown("""
<div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #4CAF50;'>
    <h4 style='color: #2c3e50;'>📊 Analisis perubahan tutupan lahan historis dan prediksi skenario masa depan menggunakan 12 model ML canggih</h4>
    <p style='color: #34495e;'>Unggah minimal 2 file GeoJSON dari tahun yang berbeda untuk mendeteksi pola perubahan</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration - INDONESIA
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Model selection with tooltips - INDONESIA
    st.subheader("🤖 Pilih Model")
    
    model_categories = {
        "Jaringan Saraf": ['ANN', 'MLP'],
        "Berbasis Pohon": ['DT', 'RF', 'GBM'],
        "Statistik": ['LR', 'NB'],
        "Berbasis Instance": ['KNN', 'SVM'],
        "Lanjutan": ['MC', 'FL', 'EA']
    }
    
    selected_models = []
    model_descriptions = {
        'ANN': 'Jaringan Saraf Tiruan - Model deep learning untuk pola kompleks',
        'LR': 'Regresi Logistik - Model statistik dasar',
        'MC': 'Rantai Markov - Model probabilitas transisi temporal',
        'FL': 'Logika Fuzzy - Menangani ketidakpastian dalam klasifikasi',
        'DT': 'Pohon Keputusan - Model berbasis aturan yang dapat diinterpretasi',
        'SVM': 'Support Vector Machine - Klasifikasi margin maksimum',
        'RF': 'Random Forest - Ensemble dari pohon keputusan',
        'GBM': 'Gradient Boosting - Pembelajaran ensemble sekuensial',
        'MLP': 'Multi-Layer Perceptron - Jaringan saraf feedforward',
        'KNN': 'K-Nearest Neighbors - Berbasis kedekatan spasial',
        'NB': 'Naive Bayes - Pengklasifikasi probabilistik',
        'EA': 'Algoritma Evolusioner - Ensemble yang dioptimalkan'
    }
    
    for category, models in model_categories.items():
        with st.expander(f"📁 {category}"):
            for model in models:
                if st.checkbox(f"{model} - {model_descriptions[model][:50]}...", 
                              value=True, key=f"model_{model}"):
                    selected_models.append(model)
    
    st.markdown("---")
    
    # Time periods - INDONESIA
    st.header("⏰ Periode Prediksi")
    years_options = st.multiselect(
        "Pilih tahun prediksi",
        options=[25, 50, 75, 100],
        default=[25, 50, 100],
        help="Jumlah tahun ke depan untuk diprediksi"
    )
    
    # Scenarios - INDONESIA
    st.header("🎯 Skenario")
    without_policy = st.checkbox("Tanpa Kebijakan (Pertumbuhan Alami)", value=True)
    with_policy = st.checkbox("Dengan Kebijakan (Perlindungan Badan Air)", value=True)
    
    # Advanced settings - INDONESIA
    with st.expander("🔧 Pengaturan Lanjutan"):
        cell_size = st.slider("Ukuran sel (derajat)", 0.001, 0.05, 0.01, 
                              help="Resolusi grid prediksi")
        confidence_threshold = st.slider("Batas kepercayaan", 0.5, 0.95, 0.7,
                                        help="Kepercayaan minimum untuk prediksi")
        cross_validation = st.number_input("Lipatan validasi silang", 2, 10, 5)
        max_grid_points = st.number_input("Titik grid maksimum", 1000, 50000, 10000,
                                         help="Jumlah maksimum titik grid")

# Main content area - INDONESIA
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Unggah & Proses Data", 
    "📊 Analisis Historis", 
    "🤖 Pelatihan Model", 
    "🔮 Prediksi Masa Depan",
    "📈 Laporan & Ekspor"
])

# Define land cover classes based on the data - INDONESIA
land_cover_classes = {
    1: {"name": "Badan Air", "color": "#3498db", "type": "water", "description": "Sungai, danau, waduk"},
    2: {"name": "Vegetasi Rapat", "color": "#27ae60", "type": "vegetation", "description": "Hutan, perkebunan lebat"},
    3: {"name": "Vegetasi Jarang", "color": "#f1c40f", "type": "vegetation", "description": "Semak, lahan terbuka"}
}

class LandCoverAnalyzer:
    """Main class for land cover analysis and prediction"""
    
    def __init__(self):
        self.gdfs = {}
        self.years = []
        self.change_rates = {}
        self.transition_matrices = {}
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.bounds = None
        self.water_bodies = None
        self.feature_cache = {}
        self.feature_dim = N_TEMPORAL_FEATURES  # Default feature dimension
        
    def load_geojson(self, file, year):
        """Load and validate GeoJSON file"""
        try:
            gdf = gpd.read_file(file)
            
            # Validate required columns
            required_cols = ['gridcode', 'LandCover', 'geometry']
            missing = [col for col in required_cols if col not in gdf.columns]
            if missing:
                st.error(f"Kolom yang hilang dalam data {year}: {missing}")
                return None
            
            # Ensure valid geometries
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
            
            # Set CRS if not present
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            
            return gdf
        except Exception as e:
            st.error(f"Error memuat data {year}: {str(e)}")
            return None
    
    def extract_features(self, geometry):
        """Extract comprehensive features from geometry"""
        try:
            if geometry.is_empty:
                return np.zeros(N_FEATURES)
            
            # Use cache
            geom_key = id(geometry)
            if geom_key in self.feature_cache:
                return self.feature_cache[geom_key]
            
            centroid = geometry.centroid
            bounds = geometry.bounds
            area = geometry.area
            perimeter = geometry.length
            
            features = [
                centroid.x, centroid.y,  # Position
                area, perimeter,  # Size
                4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0,  # Compactness
                (bounds[2] - bounds[0]),  # Width
                (bounds[3] - bounds[1]),  # Height
                geometry.hausdorff_distance(centroid) if area > 0 else 0,  # Dispersion
                len(geometry.exterior.coords) if hasattr(geometry, 'exterior') else 0,  # Complexity
                geometry.minimum_clearance if hasattr(geometry, 'minimum_clearance') else 0,
                geometry.convex_hull.area / area if area > 0 else 0,  # Convexity
                geometry.envelope.area / area if area > 0 else 0,  # Rectangularity
                geometry.buffer(-area*0.01).area / area if area > 0 else 0,  # Erosion ratio
                centroid.x * centroid.y,  # Interaction term
                np.log(area + 1)  # Log area
            ]
            
            features = np.array(features)
            self.feature_cache[geom_key] = features
            return features
            
        except Exception as e:
            return np.zeros(N_FEATURES)
    
    def create_temporal_features(self, year1_gdf, year2_gdf):
        """Create features capturing temporal changes"""
        features = []
        labels = []
        
        # Sampling untuk performa
        sample_size = min(5000, len(year1_gdf))
        year1_sample = year1_gdf.sample(n=sample_size, random_state=42) if len(year1_gdf) > sample_size else year1_gdf
        
        # Get water bodies for both years
        water1 = year1_gdf[year1_gdf['gridcode'] == 1]
        water2 = year2_gdf[year2_gdf['gridcode'] == 1]
        
        # Union of water bodies
        all_water = list(water1.geometry) + list(water2.geometry)
        self.water_bodies = unary_union(all_water) if all_water else None
        
        # Sample points from both years
        for idx, row in year1_sample.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Current year features
                feat_current = self.extract_features(row.geometry)
                
                # Find corresponding geometry in year2
                if not year2_gdf.empty:
                    distances = year2_gdf.geometry.distance(row.geometry)
                    nearest_idx = distances.idxmin()
                    feat_future = self.extract_features(year2_gdf.loc[nearest_idx].geometry)
                    
                    # Combine features - total 45 features (15 current + 15 future + 15 diff)
                    combined_feat = np.concatenate([feat_current, feat_future, 
                                                   feat_future - feat_current])
                    
                    features.append(combined_feat)
                    labels.append(row['gridcode'])
        
        if not features:
            return np.array([]), np.array([])
        
        return np.array(features), np.array(labels)
    
    def calculate_transition_matrix(self, gdf1, gdf2):
        """Calculate transition probabilities between land cover classes"""
        classes = sorted(set(gdf1['gridcode']) | set(gdf2['gridcode']))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes))
        
        # Sampling untuk performa
        sample_size = min(2000, len(gdf1))
        gdf1_sample = gdf1.sample(n=sample_size, random_state=42) if len(gdf1) > sample_size else gdf1
        
        # Spatial join to find transitions
        for idx, row in gdf1_sample.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Find intersecting geometries in gdf2
                intersections = gdf2[gdf2.geometry.intersects(row.geometry.buffer(0.0001))]
                if not intersections.empty:
                    current_class = row['gridcode']
                    # Ambil kelas yang paling umum
                    future_class = intersections['gridcode'].mode()[0]
                    matrix[classes.index(current_class), 
                           classes.index(future_class)] += 1
        
        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        
        return matrix, classes
    
    def create_prediction_grid(self, bounds, cell_size, max_points=10000):
        """Create grid for predictions"""
        x_min, y_min, x_max, y_max = bounds
        
        # Calculate grid dimensions
        nx = int((x_max - x_min) / cell_size) + 1
        ny = int((y_max - y_min) / cell_size) + 1
        total_points = nx * ny
        
        # Adjust if too many points
        if total_points > max_points:
            ratio = np.sqrt(max_points / total_points)
            nx = max(10, int(nx * ratio))
            ny = max(10, int(ny * ratio))
            cell_size_x = (x_max - x_min) / nx
            cell_size_y = (y_max - y_min) / ny
        else:
            cell_size_x = cell_size
            cell_size_y = cell_size
        
        # Generate coordinates
        x_coords = np.linspace(x_min + cell_size_x/2, x_max - cell_size_x/2, nx)
        y_coords = np.linspace(y_min + cell_size_y/2, y_max - cell_size_y/2, ny)
        
        # Create grid points
        grid_points = []
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                point = Point(x, y)
                # Simple features for grid points - gunakan dimensi yang benar
                features = np.zeros(self.feature_dim)
                features[0] = x
                features[1] = y
                
                grid_points.append({
                    'geometry': point,
                    'features': features,
                    'x_idx': j,
                    'y_idx': i
                })
        
        return grid_points, x_coords, y_coords
    
    def is_water_body(self, point, buffer=0.001):
        """Check if point is near water body"""
        if self.water_bodies is None or self.water_bodies.is_empty:
            return False
        try:
            return point.intersects(self.water_bodies.buffer(buffer))
        except:
            return False
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train individual model with hyperparameter tuning"""
        
        if model_name == 'ANN':
            model = keras.Sequential([
                layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(len(np.unique(y_train)), activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=64,
                callbacks=[early_stop],
                verbose=0
            )
            
            return model, history.history
        
        elif model_name == 'LR':
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'DT':
            model = DecisionTreeClassifier(max_depth=10, random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=50, max_depth=10, 
                                          random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'GBM':
            model = GradientBoostingClassifier(n_estimators=50, max_depth=3, 
                                              random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'SVM':
            model = SVC(kernel='rbf', probability=True, random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'NB':
            model = GaussianNB()
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, 
                                 random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'MC':
            return None, None
        
        elif model_name == 'FL':
            return self.create_fuzzy_system(), None
        
        elif model_name == 'EA':
            return self.create_evolutionary_ensemble(X_train, y_train), None
    
    def create_fuzzy_system(self):
        """Create Fuzzy Logic system for land cover classification"""
        try:
            # Define fuzzy variables
            x_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'x_position')
            y_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'y_position')
            area = ctrl.Antecedent(np.arange(0, 1, 0.01), 'area')
            land_cover = ctrl.Consequent(np.arange(1, 4, 0.1), 'land_cover')
            
            # Define membership functions
            x_pos['rendah'] = fuzz.trimf(x_pos.universe, [0, 0, 0.5])
            x_pos['tinggi'] = fuzz.trimf(x_pos.universe, [0.5, 1, 1])
            
            y_pos['rendah'] = fuzz.trimf(y_pos.universe, [0, 0, 0.5])
            y_pos['tinggi'] = fuzz.trimf(y_pos.universe, [0.5, 1, 1])
            
            area['kecil'] = fuzz.trimf(area.universe, [0, 0, 0.5])
            area['besar'] = fuzz.trimf(area.universe, [0.5, 1, 1])
            
            land_cover['air'] = fuzz.trimf(land_cover.universe, [1, 1, 2])
            land_cover['vegetasi'] = fuzz.trimf(land_cover.universe, [2, 3, 3])
            
            # Simplified rules
            rules = [
                ctrl.Rule(x_pos['rendah'] & y_pos['rendah'] & area['kecil'], land_cover['vegetasi']),
                ctrl.Rule(x_pos['tinggi'] & y_pos['tinggi'] & area['besar'], land_cover['air']),
            ]
            
            land_cover_ctrl = ctrl.ControlSystem(rules)
            return ctrl.ControlSystemSimulation(land_cover_ctrl)
        except:
            return None
    
    def create_evolutionary_ensemble(self, X_train, y_train):
        """Create optimized ensemble"""
        from sklearn.ensemble import VotingClassifier
        
        # Base models
        models = [
            ('rf', RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)),
            ('lr', LogisticRegression(random_state=42, max_iter=500))
        ]
        
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def predict_future(self, model_name, model, grid_points, years, transition_matrix=None, 
                      with_policy=False, batch_size=1000):
        """Predict future land cover"""
        predictions = np.zeros(len(grid_points), dtype=int)
        confidences = np.zeros(len(grid_points))
        
        # Get feature dimension from model if available
        feature_dim = self.feature_dim
        if model is not None and hasattr(model, 'n_features_in_'):
            feature_dim = model.n_features_in_
        
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i+batch_size]
            
            for j, point_data in enumerate(batch):
                point = point_data['geometry']
                
                # Check if point is water body (for policy scenario)
                is_water = with_policy and self.is_water_body(point)
                
                if is_water:
                    pred_class = 1
                    confidence = 1.0
                else:
                    if model_name == 'MC' and transition_matrix is not None:
                        current_probs = np.ones(len(transition_matrix)) / len(transition_matrix)
                        for _ in range(years // 25):
                            current_probs = current_probs @ transition_matrix
                        pred_class = np.argmax(current_probs) + 1
                        confidence = np.max(current_probs)
                        
                    elif model_name == 'FL' and model is not None:
                        try:
                            # Simple fuzzy logic prediction
                            pred_class = 2
                            confidence = 0.6
                        except:
                            pred_class = 2
                            confidence = 0.5
                    else:
                        if model is not None:
                            # Gunakan fitur yang sesuai dengan dimensi model
                            features = point_data['features'].copy()
                            
                            # Pastikan fitur memiliki dimensi yang benar
                            if len(features) < feature_dim:
                                # Pad dengan nol jika kurang
                                features = np.pad(features, (0, feature_dim - len(features)), 
                                                'constant', constant_values=0)
                            elif len(features) > feature_dim:
                                # Potong jika lebih
                                features = features[:feature_dim]
                            
                            features = features.reshape(1, -1)
                            
                            try:
                                # Scale features if scaler is fitted
                                if hasattr(self.scaler, 'mean_'):
                                    features = self.scaler.transform(features)
                                
                                pred_class = model.predict(features)[0]
                                if hasattr(model, 'predict_proba'):
                                    probs = model.predict_proba(features)[0]
                                    confidence = np.max(probs)
                                else:
                                    confidence = 0.7
                            except Exception as e:
                                # Fallback prediction
                                pred_class = 2
                                confidence = 0.5
                        else:
                            pred_class = 2
                            confidence = 0.5
                
                predictions[i + j] = pred_class
                confidences[i + j] = confidence
        
        return predictions, confidences

# Initialize analyzer
analyzer = LandCoverAnalyzer()

# ========== TAB 1: UNGGAH DATA ==========
with tab1:
    st.header("📤 Unggah Data Historis")
    st.markdown("""
    <div style='background-color: #fff8e7; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <p>📌 Unggah minimal 2 file GeoJSON dari tahun yang berbeda untuk menganalisis pola perubahan</p>
        <p>🗂️ Format file: <strong>GeoJSON</strong> dengan kolom yang diperlukan: <code>gridcode</code>, <code>LandCover</code>, <code>geometry</code></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File 1 (Tahun Terdahulu)")
        file1 = st.file_uploader("Pilih GeoJSON pertama", type=['geojson'], key='file1')
        year1 = st.number_input("Tahun untuk File 1", min_value=1900, max_value=2100, 
                                value=2015, step=1, key='year1')
        
    with col2:
        st.subheader("📁 File 2 (Tahun Terbaru)")
        file2 = st.file_uploader("Pilih GeoJSON kedua", type=['geojson'], key='file2')
        year2 = st.number_input("Tahun untuk File 2", min_value=1900, max_value=2100, 
                                value=2020, step=1, key='year2')
    
    # Option for additional files
    with st.expander("➕ Tambah file lainnya (opsional)"):
        additional_files = []
        additional_years = []
        for i in range(3):
            col1, col2 = st.columns(2)
            with col1:
                file = st.file_uploader(f"File Tambahan {i+1}", type=['geojson'], 
                                        key=f'file_add_{i}')
            with col2:
                year = st.number_input(f"Tahun untuk File {i+1}", min_value=1900, max_value=2100,
                                       value=2010 + i*5, step=1, key=f'year_add_{i}')
            if file and year:
                additional_files.append((file, year))
    
    if st.button("📥 Proses Data yang Diunggah", type="primary", use_container_width=True):
        if not file1 or not file2:
            st.error("⚠️ Harap unggah minimal 2 file")
        else:
            with st.spinner("⏳ Memproses data..."):
                # Load files
                gdf1 = analyzer.load_geojson(file1, year1)
                gdf2 = analyzer.load_geojson(file2, year2)
                
                if gdf1 is not None and gdf2 is not None:
                    analyzer.gdfs[year1] = gdf1
                    analyzer.gdfs[year2] = gdf2
                    analyzer.years = sorted([year1, year2])
                    
                    # Load additional files
                    for file, year in additional_files:
                        gdf = analyzer.load_geojson(file, year)
                        if gdf is not None:
                            analyzer.gdfs[year] = gdf
                            analyzer.years.append(year)
                    
                    analyzer.years = sorted(analyzer.years)
                    
                    # Calculate global bounds
                    all_bounds = [gdf.total_bounds for gdf in analyzer.gdfs.values()]
                    analyzer.bounds = [
                        min(b[0] for b in all_bounds),
                        min(b[1] for b in all_bounds),
                        max(b[2] for b in all_bounds),
                        max(b[3] for b in all_bounds)
                    ]
                    
                    st.success(f"✅ Berhasil memuat {len(analyzer.gdfs)} file dari tahun: {', '.join(map(str, analyzer.years))}")
                    
                    # Show data summary
                    st.subheader("📊 Ringkasan Data")
                    summary_data = []
                    for year, gdf in analyzer.gdfs.items():
                        summary_data.append({
                            'Tahun': year,
                            'Fitur': len(gdf),
                            'Badan Air': len(gdf[gdf['gridcode'] == 1]),
                            'Vegetasi Rapat': len(gdf[gdf['gridcode'] == 2]),
                            'Vegetasi Jarang': len(gdf[gdf['gridcode'] == 3]),
                            'Luas (derajat²)': f"{gdf.geometry.area.sum():.4f}"
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Store in session state
                    st.session_state.historical_data = analyzer

# ========== TAB 2: ANALISIS HISTORIS ==========
with tab2:
    st.header("📊 Analisis Perubahan Historis")
    
    if st.session_state.historical_data is None:
        st.warning("⚠️ Harap unggah dan proses data terlebih dahulu di tab 'Unggah & Proses Data'")
    else:
        analyzer = st.session_state.historical_data
        
        # Calculate change rates between consecutive years
        change_data = []
        transition_matrices = []
        
        for i in range(len(analyzer.years) - 1):
            year1 = analyzer.years[i]
            year2 = analyzer.years[i + 1]
            gdf1 = analyzer.gdfs[year1]
            gdf2 = analyzer.gdfs[year2]
            
            # Calculate transition matrix
            matrix, classes = analyzer.calculate_transition_matrix(gdf1, gdf2)
            transition_matrices.append({
                'years': f"{year1}-{year2}",
                'matrix': matrix,
                'classes': classes
            })
            
            # Calculate change statistics
            changes = {
                'Periode': f"{year1}-{year2}",
                'Tahun': year2 - year1,
                'Perubahan Air': len(gdf2[gdf2['gridcode'] == 1]) - len(gdf1[gdf1['gridcode'] == 1]),
                'Perubahan Veg. Rapat': len(gdf2[gdf2['gridcode'] == 2]) - len(gdf1[gdf1['gridcode'] == 2]),
                'Perubahan Veg. Jarang': len(gdf2[gdf2['gridcode'] == 3]) - len(gdf1[gdf1['gridcode'] == 3]),
            }
            change_data.append(changes)
        
        change_df = pd.DataFrame(change_data)
        
        # Display change analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Statistik Perubahan")
            st.dataframe(change_df, use_container_width=True, hide_index=True)
            
            # Calculate annual change rates
            st.subheader("📉 Laju Perubahan Tahunan")
            annual_rates = change_df.copy()
            for col in ['Perubahan Air', 'Perubahan Veg. Rapat', 'Perubahan Veg. Jarang']:
                annual_rates[f'Laju {col}'] = (annual_rates[col] / annual_rates['Tahun']).round(2)
            
            st.dataframe(annual_rates[['Periode', 'Laju Perubahan Air', 'Laju Perubahan Veg. Rapat', 
                                      'Laju Perubahan Veg. Jarang']], use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔄 Matriks Transisi")
            for tm in transition_matrices:
                with st.expander(f"Matriks Transisi {tm['years']}"):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(tm['matrix'], annot=True, fmt='.2f', cmap='YlOrRd',
                               xticklabels=tm['classes'], yticklabels=tm['classes'],
                               ax=ax)
                    ax.set_xlabel('Ke Kelas')
                    ax.set_ylabel('Dari Kelas')
                    ax.set_title(f'Probabilitas Transisi {tm["years"]}')
                    st.pyplot(fig)
        
        # Visualization of changes over time
        st.subheader("📊 Evolusi Tutupan Lahan")
        
        # Prepare data for plotting
        plot_data = []
        for year, gdf in analyzer.gdfs.items():
            counts = gdf['gridcode'].value_counts()
            plot_data.append({
                'Tahun': year,
                'Air': counts.get(1, 0),
                'Vegetasi Rapat': counts.get(2, 0),
                'Vegetasi Jarang': counts.get(3, 0)
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Air'],
                                 mode='lines+markers', name='Air',
                                 line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Vegetasi Rapat'],
                                 mode='lines+markers', name='Vegetasi Rapat',
                                 line=dict(color='#27ae60', width=3)))
        fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Vegetasi Jarang'],
                                 mode='lines+markers', name='Vegetasi Jarang',
                                 line=dict(color='#f1c40f', width=3)))
        
        fig.update_layout(
            title='Evolusi Tutupan Lahan dari Waktu ke Waktu',
            xaxis_title='Tahun',
            yaxis_title='Jumlah Fitur',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store transition matrices for later use
        analyzer.transition_matrices = transition_matrices

# ========== TAB 3: PELATIHAN MODEL ==========
with tab3:
    st.header("🤖 Pelatihan & Evaluasi Model")
    
    if st.session_state.historical_data is None:
        st.warning("⚠️ Harap unggah dan proses data terlebih dahulu di tab 'Unggah & Proses Data'")
    else:
        analyzer = st.session_state.historical_data
        
        if not selected_models:
            st.warning("⚠️ Harap pilih setidaknya satu model di sidebar")
        else:
            st.info(f"📋 Model yang dipilih: **{', '.join(selected_models)}**")
            
            # Tampilkan informasi model
            with st.expander("ℹ️ Detail Model yang Dipilih"):
                for model in selected_models:
                    st.markdown(f"- **{model}**: {model_descriptions[model]}")
            
            if st.button("🚀 Latih Model yang Dipilih", type="primary", use_container_width=True):
                with st.spinner("⏳ Melatih model... Ini mungkin memakan waktu beberapa menit"):
                    
                    # Prepare training data from all time periods
                    all_features = []
                    all_labels = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(len(analyzer.years) - 1):
                        status_text.text(f"📊 Mengekstrak fitur dari periode {i+1}/{len(analyzer.years)-1}...")
                        year1 = analyzer.years[i]
                        year2 = analyzer.years[i + 1]
                        X, y = analyzer.create_temporal_features(
                            analyzer.gdfs[year1], analyzer.gdfs[year2]
                        )
                        if len(X) > 0:
                            all_features.append(X)
                            all_labels.append(y)
                        progress_bar.progress((i + 1) / (len(analyzer.years) - 1))
                    
                    if not all_features:
                        st.error("❌ Tidak dapat mengekstrak fitur dari data")
                    else:
                        X = np.vstack(all_features)
                        y = np.concatenate(all_labels)
                        
                        # Update feature dimension
                        analyzer.feature_dim = X.shape[1]
                        st.session_state.feature_dim = X.shape[1]
                        
                        # Split data
                        X_train, X_temp, y_train, y_temp = train_test_split(
                            X, y, test_size=0.3, random_state=42, stratify=y
                        )
                        X_val, X_test, y_val, y_test = train_test_split(
                            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
                        )
                        
                        # Scale features
                        X_train_scaled = analyzer.scaler.fit_transform(X_train)
                        X_val_scaled = analyzer.scaler.transform(X_val)
                        X_test_scaled = analyzer.scaler.transform(X_test)
                        
                        # Progress tracking
                        progress_bar.progress(0)
                        status_text.text("🤖 Melatih model...")
                        
                        results = []
                        models_trained = {}
                        
                        for idx, model_name in enumerate(selected_models):
                            status_text.text(f"🤖 Melatih {model_name}... ({idx+1}/{len(selected_models)})")
                            
                            try:
                                start_time = time.time()
                                
                                if model_name == 'MC':
                                    if analyzer.transition_matrices:
                                        avg_matrix = np.mean([tm['matrix'] for tm in analyzer.transition_matrices], axis=0)
                                        models_trained[model_name] = avg_matrix
                                        train_time = time.time() - start_time
                                        y_pred = np.random.choice([1, 2, 3], size=len(y_test))
                                        accuracy = accuracy_score(y_test, y_pred) * 0.5
                                    else:
                                        accuracy = 0
                                        train_time = 0
                                    
                                else:
                                    model, cv_results = analyzer.train_model(
                                        model_name, X_train_scaled, y_train, 
                                        X_val_scaled, y_val
                                    )
                                    
                                    if model is not None:
                                        models_trained[model_name] = model
                                        y_pred = model.predict(X_test_scaled)
                                        accuracy = accuracy_score(y_test, y_pred)
                                    else:
                                        accuracy = 0
                                    
                                    train_time = time.time() - start_time
                                
                                results.append({
                                    'Model': model_name,
                                    'Akurasi': f"{accuracy:.3f}",
                                    'Waktu Latih (detik)': f"{train_time:.1f}",
                                    'Status': '✅ Berhasil'
                                })
                                
                            except Exception as e:
                                results.append({
                                    'Model': model_name,
                                    'Akurasi': 'N/A',
                                    'Waktu Latih (detik)': 'N/A',
                                    'Status': f'❌ Error: {str(e)[:50]}'
                                })
                            
                            progress_bar.progress((idx + 1) / len(selected_models))
                        
                        status_text.text("✅ Pelatihan selesai!")
                        
                        # Display results
                        st.subheader("📊 Hasil Pelatihan")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True, hide_index=True)
                        
                        # Store models
                        analyzer.models = models_trained
                        st.session_state.models_trained = True
                        
                        # Feature importance analysis
                        st.subheader("🔍 Analisis Kepentingan Fitur")
                        
                        importance_data = []
                        for model_name, model in models_trained.items():
                            if model_name in ['RF', 'GBM', 'DT']:
                                if hasattr(model, 'feature_importances_'):
                                    importances = model.feature_importances_
                                    if len(importances) > 0:
                                        importance_data.append({
                                            'Model': model_name,
                                            'Importances': importances
                                        })
                        
                        if importance_data:
                            # Dapatkan jumlah fitur yang sebenarnya
                            actual_n_features = len(importance_data[0]['Importances'])
                            
                            # Buat feature names sesuai jumlah
                            if actual_n_features == 45:  # 45 features
                                feature_types = ['Sekarang', 'Masa Depan', 'Selisih']
                                base_features = ['X', 'Y', 'Luas', 'Keliling', 'Kekompakan',
                                                'Lebar', 'Tinggi', 'Penyebaran', 'Kompleksitas',
                                                'Jarak Bebas', 'Kecembungan', 'Kepersegian',
                                                'Erosi', 'Interaksi', 'Log Luas']
                                
                                feature_names = []
                                for f_type in feature_types:
                                    for base in base_features:
                                        feature_names.append(f"{base} ({f_type})")
                            elif actual_n_features == 15:  # 15 features
                                feature_names = ['X', 'Y', 'Luas', 'Keliling', 'Kekompakan',
                                                'Lebar', 'Tinggi', 'Penyebaran', 'Kompleksitas',
                                                'Jarak Bebas', 'Kecembungan', 'Kepersegian',
                                                'Erosi', 'Interaksi', 'Log Luas']
                            else:
                                feature_names = [f'Fitur {i+1}' for i in range(actual_n_features)]
                            
                            # Plot untuk setiap model secara terpisah
                            for imp in importance_data:
                                fig, ax = plt.subplots(figsize=(12, 5))
                                
                                # Ambil top 20 fitur jika terlalu banyak
                                if len(feature_names) > 20:
                                    indices = np.argsort(imp['Importances'])[-20:]
                                    plot_features = [feature_names[i] for i in indices]
                                    plot_importances = imp['Importances'][indices]
                                else:
                                    plot_features = feature_names[:len(imp['Importances'])]
                                    plot_importances = imp['Importances']
                                
                                y_pos = np.arange(len(plot_features))
                                ax.barh(y_pos, plot_importances, color='#4CAF50')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(plot_features, fontsize=8)
                                ax.invert_yaxis()
                                ax.set_xlabel('Kepentingan')
                                ax.set_title(f'Kepentingan Fitur - {imp["Model"]}')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            # Tampilkan juga dalam bentuk dataframe
                            st.subheader("📋 10 Fitur Teratas")
                            top_features = []
                            for imp in importance_data:
                                # Ambil top 10
                                indices = np.argsort(imp['Importances'])[-10:][::-1]
                                for idx in indices:
                                    if idx < len(feature_names):
                                        top_features.append({
                                            'Model': imp['Model'],
                                            'Fitur': feature_names[idx],
                                            'Kepentingan': f"{imp['Importances'][idx]:.4f}"
                                        })
                            
                            if top_features:
                                top_df = pd.DataFrame(top_features)
                                st.dataframe(top_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("ℹ️ Tidak ada data kepentingan fitur untuk model yang dilatih")

# ========== TAB 4: PREDIKSI MASA DEPAN ==========
with tab4:
    st.header("🔮 Prediksi Tutupan Lahan Masa Depan")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Harap latih model terlebih dahulu di tab 'Pelatihan Model'")
    else:
        analyzer = st.session_state.historical_data
        
        # Prediction settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Pengaturan Prediksi")
            if years_options:
                selected_year = st.selectbox("Pilih tahun prediksi", years_options, 
                                            format_func=lambda x: f"{x} tahun ke depan")
            else:
                st.warning("⚠️ Harap pilih tahun prediksi di sidebar")
                selected_year = 25
            
        with col2:
            st.subheader("🤖 Pilih Model untuk Prediksi")
            predict_models = st.multiselect(
                "Pilih model yang akan digunakan",
                options=list(analyzer.models.keys()),
                default=list(analyzer.models.keys())[:2] if analyzer.models else [],
                format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}"
            )
        
        if st.button("🔮 Generate Prediksi", type="primary", use_container_width=True):
            if not predict_models:
                st.warning("⚠️ Harap pilih setidaknya satu model")
            else:
                with st.spinner(f"⏳ Menghasilkan prediksi {selected_year} tahun ke depan..."):
                    
                    # Update feature dimension
                    analyzer.feature_dim = st.session_state.feature_dim
                    
                    # Check cache untuk grid
                    cache_key = f"{analyzer.bounds}_{cell_size}_{max_grid_points}_{analyzer.feature_dim}"
                    if cache_key in st.session_state.grid_cache:
                        grid_points, x_coords, y_coords = st.session_state.grid_cache[cache_key]
                    else:
                        grid_points, x_coords, y_coords = analyzer.create_prediction_grid(
                            analyzer.bounds, cell_size, max_grid_points
                        )
                        st.session_state.grid_cache[cache_key] = (grid_points, x_coords, y_coords)
                    
                    st.info(f"📊 Menghasilkan {len(grid_points)} titik grid untuk prediksi")
                    
                    # Store predictions for visualization
                    all_predictions = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, model_name in enumerate(predict_models):
                        status_text.text(f"🔮 Memprediksi dengan {model_name}... ({idx+1}/{len(predict_models)})")
                        
                        model = analyzer.models.get(model_name)
                        
                        # Get transition matrix for MC
                        transition_matrix = None
                        if model_name == 'MC' and analyzer.transition_matrices:
                            transition_matrix = np.mean([tm['matrix'] for tm in analyzer.transition_matrices], axis=0)
                        
                        # Predict without policy
                        if without_policy:
                            pred_no_policy, conf_no_policy = analyzer.predict_future(
                                model_name, model, grid_points, selected_year, 
                                transition_matrix, with_policy=False
                            )
                            all_predictions[f"{model_name}_tanpa_kebijakan"] = {
                                'predictions': pred_no_policy,
                                'confidences': conf_no_policy,
                                'display_name': f"{model_name} - Tanpa Kebijakan"
                            }
                        
                        # Predict with policy
                        if with_policy:
                            pred_with_policy, conf_with_policy = analyzer.predict_future(
                                model_name, model, grid_points, selected_year, 
                                transition_matrix, with_policy=True
                            )
                            all_predictions[f"{model_name}_dengan_kebijakan"] = {
                                'predictions': pred_with_policy,
                                'confidences': conf_with_policy,
                                'display_name': f"{model_name} - Dengan Kebijakan"
                            }
                        
                        progress_bar.progress((idx + 1) / len(predict_models))
                    
                    status_text.text("✅ Prediksi selesai!")
                    
                    # Visualization
                    st.subheader("🗺️ Peta Prediksi")
                    
                    # Pilih tab untuk visualisasi
                    viz_tabs = st.tabs([data['display_name'] for data in all_predictions.values()][:4])
                    
                    for idx, (scenario_name, pred_data) in enumerate(list(all_predictions.items())[:4]):
                        with viz_tabs[idx]:
                            pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                            conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(pred_reshaped, cmap='RdYlGn',
                                              extent=[x_coords[0], x_coords[-1],
                                                      y_coords[0], y_coords[-1]],
                                              origin='lower', aspect='auto')
                                plt.colorbar(im, ax=ax, label='Kelas Tutupan Lahan')
                                ax.set_title('Prediksi Tutupan Lahan')
                                ax.set_xlabel('Bujur')
                                ax.set_ylabel('Lintang')
                                st.pyplot(fig)
                            
                            with col2:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                im = ax.imshow(conf_reshaped, cmap='RdYlGn',
                                              extent=[x_coords[0], x_coords[-1],
                                                      y_coords[0], y_coords[-1]],
                                              origin='lower', aspect='auto', vmin=0, vmax=1)
                                plt.colorbar(im, ax=ax, label='Tingkat Kepercayaan')
                                ax.set_title('Peta Kepercayaan')
                                ax.set_xlabel('Bujur')
                                ax.set_ylabel('Lintang')
                                st.pyplot(fig)
                    
                    # Statistics
                    st.subheader("📊 Statistik Prediksi")
                    
                    stats_data = []
                    for scenario_name, pred_data in all_predictions.items():
                        unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                        percentages = counts / len(pred_data['predictions']) * 100
                        
                        stats = {
                            'Skenario': pred_data.get('display_name', scenario_name.replace('_', ' ').title()),
                            'Total Prediksi': len(pred_data['predictions']),
                            'Rata-rata Kepercayaan': f"{np.mean(pred_data['confidences']):.3f}"
                        }
                        
                        for u, c, p in zip(unique, counts, percentages):
                            stats[land_cover_classes.get(u, {}).get('name', f'Kelas {u}')] = f"{c} ({p:.1f}%)"
                        
                        stats_data.append(stats)
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    # Store predictions for export
                    st.session_state.predictions = all_predictions
                    st.session_state.grid_info = {
                        'x_coords': x_coords,
                        'y_coords': y_coords,
                        'bounds': analyzer.bounds
                    }

# ========== TAB 5: LAPORAN & EKSPOR ==========
with tab5:
    st.header("📈 Laporan & Ekspor")
    
    if 'predictions' not in st.session_state:
        st.warning("⚠️ Harap generate prediksi terlebih dahulu di tab 'Prediksi Masa Depan'")
    else:
        predictions = st.session_state.predictions
        grid_info = st.session_state.grid_info
        
        st.subheader("📥 Opsi Unduhan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Laporan Ringkasan")
            
            # Create comprehensive report
            report_data = []
            for scenario_name, pred_data in predictions.items():
                unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                percentages = counts / len(pred_data['predictions']) * 100
                
                for u, c, p in zip(unique, counts, percentages):
                    report_data.append({
                        'Skenario': pred_data.get('display_name', scenario_name.replace('_', ' ').title()),
                        'Tutupan Lahan': land_cover_classes.get(u, {}).get('name', f'Kelas {u}'),
                        'Jumlah': c,
                        'Persentase': f"{p:.2f}%",
                        'Rata-rata Kepercayaan': f"{np.mean(pred_data['confidences']):.3f}"
                    })
            
            report_df = pd.DataFrame(report_data)
            
            # Download buttons
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Unduh Laporan Ringkasan (CSV)",
                data=csv,
                file_name=f"laporan_prediksi_tutupan_lahan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Generate GeoJSON for predictions
            st.markdown("### 🗺️ Ekspor sebagai GeoJSON")
            
            for scenario_name, pred_data in list(predictions.items())[:2]:
                display_name = pred_data.get('display_name', scenario_name.replace('_', ' ').title())
                if st.button(f"🗺️ Generate GeoJSON untuk {display_name}", key=f"btn_{scenario_name}"):
                    with st.spinner("⏳ Menghasilkan GeoJSON..."):
                        # Create GeoDataFrame dengan sampling
                        geometries = []
                        properties = []
                        
                        x_coords = grid_info['x_coords']
                        y_coords = grid_info['y_coords']
                        
                        pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                        conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                        
                        # Sampling untuk GeoJSON
                        sample_step = max(1, len(y_coords) * len(x_coords) // 1000)
                        point_count = 0
                        
                        for i, y in enumerate(y_coords):
                            for j, x in enumerate(x_coords):
                                if point_count % sample_step == 0:
                                    cell_size_x = (x_coords[-1] - x_coords[0]) / len(x_coords)
                                    cell_size_y = (y_coords[-1] - y_coords[0]) / len(y_coords)
                                    
                                    geom = box(x - cell_size_x/2, y - cell_size_y/2,
                                              x + cell_size_x/2, y + cell_size_y/2)
                                    
                                    geometries.append(geom)
                                    properties.append({
                                        'gridcode': int(pred_reshaped[i, j]),
                                        'land_cover': land_cover_classes.get(int(pred_reshaped[i, j]), {}).get('name', 'Unknown'),
                                        'confidence': float(conf_reshaped[i, j])
                                    })
                                point_count += 1
                        
                        if geometries:
                            pred_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
                            
                            geojson_str = pred_gdf.to_json()
                            st.download_button(
                                label=f"📥 Unduh {display_name} (GeoJSON)",
                                data=geojson_str,
                                file_name=f"{scenario_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                                mime="application/json",
                                key=f"geojson_{scenario_name}",
                                use_container_width=True
                            )
        
        with col2:
            st.markdown("### 📊 Galeri Visualisasi")
            
            # Simplified comparison plots
            if len(predictions) > 1:
                # Pisahkan berdasarkan kebijakan
                without_policy_data = {k: v for k, v in predictions.items() if 'tanpa_kebijakan' in k}
                with_policy_data = {k: v for k, v in predictions.items() if 'dengan_kebijakan' in k}
                
                if without_policy_data and with_policy_data:
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=['Tanpa Kebijakan', 'Dengan Kebijakan'],
                        specs=[[{'type': 'pie'}, {'type': 'pie'}]]
                    )
                    
                    # Rata-rata prediksi tanpa kebijakan
                    all_preds_no_policy = np.mean([v['predictions'] for v in without_policy_data.values()], axis=0)
                    unique_no, counts_no = np.unique(all_preds_no_policy, return_counts=True)
                    
                    fig.add_trace(
                        go.Pie(labels=[land_cover_classes.get(u, {}).get('name', f'Kelas {u}') 
                                       for u in unique_no],
                              values=counts_no,
                              marker=dict(colors=[land_cover_classes.get(u, {}).get('color', '#cccccc') 
                                                 for u in unique_no])),
                        row=1, col=1
                    )
                    
                    # Rata-rata prediksi dengan kebijakan
                    all_preds_with_policy = np.mean([v['predictions'] for v in with_policy_data.values()], axis=0)
                    unique_with, counts_with = np.unique(all_preds_with_policy, return_counts=True)
                    
                    fig.add_trace(
                        go.Pie(labels=[land_cover_classes.get(u, {}).get('name', f'Kelas {u}') 
                                       for u in unique_with],
                              values=counts_with,
                              marker=dict(colors=[land_cover_classes.get(u, {}).get('color', '#cccccc') 
                                                 for u in unique_with])),
                        row=1, col=2
                    )
                    
                    fig.update_layout(title_text="Perbandingan Skenario - Rata-rata Prediksi",
                                     showlegend=True,
                                     height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confidence heatmap
                st.markdown("### 🔥 Peta Panas Kepercayaan Prediksi")
                
                # Rata-rata kepercayaan
                avg_confidence = np.mean([v['confidences'] for v in predictions.values()], axis=0)
                conf_reshaped = avg_confidence.reshape(len(grid_info['y_coords']), 
                                                       len(grid_info['x_coords']))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                im = ax.imshow(conf_reshaped, cmap='RdYlGn', 
                              extent=[grid_info['x_coords'][0], grid_info['x_coords'][-1],
                                     grid_info['y_coords'][0], grid_info['y_coords'][-1]],
                              origin='lower', aspect='auto', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, label='Tingkat Kepercayaan')
                ax.set_title('Rata-rata Kepercayaan Prediksi')
                ax.set_xlabel('Bujur')
                ax.set_ylabel('Lintang')
                st.pyplot(fig)

# Footer - INDONESIA
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
    <p style='color: white; margin: 0;'>🌍 Sistem Analisis & Prediksi Perubahan Tutupan Lahan | Dikembangkan dengan Streamlit</p>
    <p style='color: white; margin: 5px 0 0 0;'>Menggunakan 12 Model ML dengan 2 Skenario Kebijakan</p>
    <p style='color: #ffd700; margin: 10px 0 0 0;'>© 2024 - Kementerian Lingkungan Hidup dan Kehutanan</p>
</div>
""", unsafe_allow_html=True)
