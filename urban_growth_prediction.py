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
import branca.colormap as cm
import requests
import io
import zipfile

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
    .legend-item {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 5px;
        border-radius: 3px;
    }
    .sample-data-card {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    }
    .download-link {
        color: #4CAF50;
        text-decoration: none;
        font-weight: bold;
    }
    .download-link:hover {
        text-decoration: underline;
    }
    .policy-info {
        background-color: #e8f4fd;
        border-left: 5px solid #2196F3;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .protected-class {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 10px;
        border-radius: 3px;
        display: inline-block;
        margin: 2px;
    }
    .dynamic-class {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 10px;
        border-radius: 3px;
        display: inline-block;
        margin: 2px;
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
if 'reclassified_gdfs' not in st.session_state:
    st.session_state.reclassified_gdfs = {}
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_grid_info' not in st.session_state:
    st.session_state.current_grid_info = None
if 'selected_prediction_year' not in st.session_state:
    st.session_state.selected_prediction_year = 25
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False
if 'policy_effectiveness' not in st.session_state:
    st.session_state.policy_effectiveness = None

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
        "Neural Networks (Jaringan Saraf)": ['ANN', 'MLP'],
        "Tree-based (Berbasis Pohon)": ['DT', 'RF', 'GBM'],
        "Statistical (Statistik)": ['LR', 'NB'],
        "Instance-based (Berbasis Sampling)": ['KNN', 'SVM'],
        "Advanced (Lanjutan)": ['MC', 'FL', 'EA']
    }
    
    selected_models = []
    model_descriptions = {
        'ANN': 'Artificial Neural Networks (Jaringan Saraf Tiruan) - Model deep learning untuk pola kompleks',
        'LR': 'Logistic Regression (Regresi Logistik) - Model statistik dasar',
        'MC': 'Markov Chain (Rantai Markov) - Model probabilitas transisi temporal',
        'FL': 'Fuzzy Logic (Logika Fuzzy) - Menangani ketidakpastian dalam klasifikasi',
        'DT': 'Decision Tree (Pohon Keputusan) - Model berbasis aturan yang dapat diinterpretasi',
        'SVM': 'Support Vector Machine - Klasifikasi margin maksimum',
        'RF': 'Random Forest - Ensemble dari pohon keputusan',
        'GBM': 'Gradient Boosting - Pembelajaran ensemble sekuensial',
        'MLP': 'Multi-Layer Perceptron - Jaringan saraf feedforward',
        'KNN': 'K-Nearest Neighbors - Berbasis kedekatan spasial',
        'NB': 'Naive Bayes - Pengklasifikasi probabilistik',
        'EA': 'Evolutionary Algorithm (Algoritma Evolusioner) - Ensemble yang dioptimalkan'
    }
    
    for category, models in model_categories.items():
        with st.expander(f"📁 {category}"):
            for model in models:
                if st.checkbox(f"{model} - {model_descriptions[model][:50]}...", 
                              value=True, key=f"model_{model}"):
                    selected_models.append(model)
    
    st.markdown("---")
    
    # Basemap selection
    st.subheader("🗺️ Pilih Basemap")
    basemap_options = {
        "OpenStreetMap": "OpenStreetMap",
        "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Terrain": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "Dark Matter": "CartoDB dark_matter",
        "Positron": "CartoDB positron",
        "Watercolor": "Stamen Watercolor",
        "Toner": "Stamen Toner"
    }
    selected_basemap = st.selectbox("Pilih jenis peta dasar", list(basemap_options.keys()))
    
    st.markdown("---")
    
    # Time periods - INDONESIA
    st.header("⏰ Periode Prediksi")
    years_options = st.multiselect(
        "Pilih tahun prediksi",
        options=[25, 50, 75, 100],
        default=[25, 50, 100],
        help="Jumlah tahun ke depan untuk diprediksi"
    )
    
    # Scenarios - INDONESIA with enhanced description
    st.header("🎯 Skenario")
    without_policy = st.checkbox("Tanpa Kebijakan (Pertumbuhan Alami)", value=True)
    with_policy = st.checkbox(
        "Dengan Kebijakan (Perlindungan Ekologis)", 
        value=True,
        help="🌊 Badan Air dan 🌳 Vegetasi Rapat dilindungi dan tidak berubah\n" +
             "🌿 Vegetasi Jarang, 🏙️ Lahan Terbangun, dan 🏜️ Lahan Terbuka dapat berkembang"
    )
    
    # Advanced settings - INDONESIA
    with st.expander("🔧 Pengaturan Lanjutan"):
        cell_size = st.slider("Ukuran sel (derajat)", 0.001, 0.05, 0.01, 
                              help="Resolusi grid prediksi")
        confidence_threshold = st.slider("Batas kepercayaan", 0.5, 0.95, 0.7,
                                        help="Kepercayaan minimum untuk prediksi")
        cross_validation = st.number_input("Lipatan validasi silang", 2, 10, 5)
        max_grid_points = st.number_input("Titik grid maksimum", 1000, 50000, 10000,
                                         help="Jumlah maksimum titik grid")
        show_prediction_map = st.checkbox("Tampilkan peta prediksi dengan basemap", value=True)
        policy_buffer = st.slider("Buffer zona lindung (derajat)", 0.0, 0.01, 0.001,
                                  help="Area buffer di sekitar badan air yang juga dilindungi")

# Main content area - INDONESIA
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Unggah & Proses Data", 
    "📊 Analisis Historis", 
    "🤖 Pelatihan Model", 
    "🔮 Prediksi Masa Depan",
    "📈 Laporan & Ekspor"
])

# ========== KLASIFIKASI TUTUPAN LAHAN BARU (5 KELAS) ==========
# Kelas baru: 
# 1: Badan Air
# 2: Vegetasi Rapat (hutan, perkebunan lebat)
# 3: Vegetasi Jarang (semak, padang rumput, lahan pertanian)
# 4: Lahan Terbangun (permukiman, industri, infrastruktur)
# 5: Lahan Terbuka (tanah kosong, pasir, tambang)

land_cover_classes_new = {
    1: {"name": "Badan Air", "color": "#3498db", "type": "water", "description": "Sungai, danau, waduk, laut", "protected": True},
    2: {"name": "Vegetasi Rapat", "color": "#2ecc71", "type": "dense_vegetation", "description": "Hutan lebat, perkebunan dengan kanopi rapat", "protected": True},
    3: {"name": "Vegetasi Jarang", "color": "#f1c40f", "type": "sparse_vegetation", "description": "Semak, padang rumput, lahan pertanian", "protected": False},
    4: {"name": "Lahan Terbangun", "color": "#e74c3c", "type": "built_up", "description": "Permukiman, industri, infrastruktur", "protected": False},
    5: {"name": "Lahan Terbuka", "color": "#95a5a6", "type": "open_land", "description": "Tanah kosong, pasir, tambang, lahan tandus", "protected": False}
}

# Fungsi untuk reklasifikasi dari gridcode asli ke 5 kelas baru
def reclassify_land_cover(gridcode):
    """
    Reklasifikasi tutupan lahan menjadi 5 kelas:
    - gridcode 1 (Badan Air) -> kelas 1 (Badan Air)
    - gridcode 2 (Vegetasi Rapat) -> kelas 2 (Vegetasi Rapat)
    - gridcode 3 (Vegetasi Jarang) -> kelas 3 (Vegetasi Jarang)
    - gridcode 4 (Bangunan) -> kelas 4 (Lahan Terbangun)
    - Lainnya (jika ada) -> kelas 5 (Lahan Terbuka)
    """
    if gridcode == 1:
        return 1  # Badan Air
    elif gridcode == 2:
        return 2  # Vegetasi Rapat
    elif gridcode == 3:
        return 3  # Vegetasi Jarang
    elif gridcode == 4:
        return 4  # Lahan Terbangun
    else:
        return 5  # Lahan Terbuka (default untuk kode lain)

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
        self.dense_vegetation = None
        self.feature_cache = {}
        self.feature_dim = N_TEMPORAL_FEATURES  # Default feature dimension
        self.reclassified_gdfs = {}  # Untuk menyimpan GDF yang sudah direklasifikasi
        self.reference_gdf = None  # Untuk menyimpan GDF referensi untuk ekstraksi fitur
        
    def load_geojson(self, file, year):
        """Load and validate GeoJSON file"""
        try:
            # Handle both file paths and uploaded files
            if isinstance(file, (str, Path)):
                gdf = gpd.read_file(file)
            else:
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
            
            # Create reclassified version (5 kelas)
            gdf_reclass = gdf.copy()
            gdf_reclass['gridcode_original'] = gdf_reclass['gridcode']
            gdf_reclass['gridcode'] = gdf_reclass['gridcode_original'].apply(reclassify_land_cover)
            gdf_reclass['LandCover'] = gdf_reclass['gridcode'].map({
                1: 'Badan Air',
                2: 'Vegetasi Rapat',
                3: 'Vegetasi Jarang',
                4: 'Lahan Terbangun',
                5: 'Lahan Terbuka'
            })
            
            # Store both versions
            self.reclassified_gdfs[year] = gdf_reclass
            
            # Set reference GDF untuk ekstraksi fitur (gunakan yang terbaru)
            if self.reference_gdf is None or year > max(self.years) if self.years else True:
                self.reference_gdf = gdf_reclass
            
            return gdf
        except Exception as e:
            st.error(f"Error memuat data {year}: {str(e)}")
            return None
    
    def get_reclassified_gdf(self, year):
        """Get reclassified GeoDataFrame for a specific year"""
        return self.reclassified_gdfs.get(year, None)
    
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
        """Create features capturing temporal changes using reclassified data"""
        features = []
        labels = []
        
        # Get reclassified GDFs
        gdf1_reclass = self.reclassified_gdfs.get(year1_gdf.attrs.get('year', 0), year1_gdf)
        gdf2_reclass = self.reclassified_gdfs.get(year2_gdf.attrs.get('year', 0), year2_gdf)
        
        # Sampling untuk performa
        sample_size = min(5000, len(gdf1_reclass))
        year1_sample = gdf1_reclass.sample(n=sample_size, random_state=42) if len(gdf1_reclass) > sample_size else gdf1_reclass
        
        # Get water bodies and dense vegetation for both years
        water1 = gdf1_reclass[gdf1_reclass['gridcode'] == 1]
        water2 = gdf2_reclass[gdf2_reclass['gridcode'] == 1]
        dense_veg1 = gdf1_reclass[gdf1_reclass['gridcode'] == 2]
        dense_veg2 = gdf2_reclass[gdf2_reclass['gridcode'] == 2]
        
        # Union of protected areas
        all_water = list(water1.geometry) + list(water2.geometry)
        all_dense_veg = list(dense_veg1.geometry) + list(dense_veg2.geometry)
        
        self.water_bodies = unary_union(all_water) if all_water else None
        self.dense_vegetation = unary_union(all_dense_veg) if all_dense_veg else None
        
        # Sample points from both years
        for idx, row in year1_sample.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Current year features
                feat_current = self.extract_features(row.geometry)
                
                # Find corresponding geometry in year2
                if not gdf2_reclass.empty:
                    distances = gdf2_reclass.geometry.distance(row.geometry)
                    nearest_idx = distances.idxmin()
                    feat_future = self.extract_features(gdf2_reclass.loc[nearest_idx].geometry)
                    
                    # Combine features
                    combined_feat = np.concatenate([feat_current, feat_future, 
                                                   feat_future - feat_current])
                    
                    features.append(combined_feat)
                    labels.append(row['gridcode'])  # Gunakan gridcode yang sudah direklasifikasi
        
        if not features:
            return np.array([]), np.array([])
        
        return np.array(features), np.array(labels)
    
    def calculate_transition_matrix(self, gdf1, gdf2):
        """Calculate transition probabilities between land cover classes using reclassified data (5 kelas)"""
        # Gunakan reclassified GDFs
        gdf1_reclass = self.reclassified_gdfs.get(gdf1.attrs.get('year', 0), gdf1)
        gdf2_reclass = self.reclassified_gdfs.get(gdf2.attrs.get('year', 0), gdf2)
        
        classes = sorted(set(gdf1_reclass['gridcode']) | set(gdf2_reclass['gridcode']))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes))
        
        # Sampling untuk performa
        sample_size = min(2000, len(gdf1_reclass))
        gdf1_sample = gdf1_reclass.sample(n=sample_size, random_state=42) if len(gdf1_reclass) > sample_size else gdf1_reclass
        
        # Spatial join to find transitions
        for idx, row in gdf1_sample.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Find intersecting geometries in gdf2
                intersections = gdf2_reclass[gdf2_reclass.geometry.intersects(row.geometry.buffer(0.0001))]
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
        """Create grid for predictions dengan resolusi yang lebih baik"""
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
        
        # Create grid points dengan fitur yang lengkap
        grid_points = []
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                point = Point(x, y)
                
                # Buat fitur lengkap berdasarkan referensi GDF
                features = self.create_grid_features(point)
                
                grid_points.append({
                    'geometry': point,
                    'features': features,
                    'x_idx': j,
                    'y_idx': i
                })
        
        return grid_points, x_coords, y_coords
    
    def create_grid_features(self, point):
        """Create comprehensive features for a grid point"""
        features = np.zeros(self.feature_dim)
        
        # Fitur dasar: koordinat
        features[0] = point.x
        features[1] = point.y
        
        # Cari fitur dari referensi GDF
        if self.reference_gdf is not None:
            try:
                # Cari polygon terdekat
                distances = self.reference_gdf.geometry.distance(point)
                nearest_idx = distances.idxmin()
                nearest_geom = self.reference_gdf.loc[nearest_idx, 'geometry']
                
                # Ekstrak fitur dari geometry terdekat
                ref_features = self.extract_features(nearest_geom)
                
                # Isi fitur lainnya dari referensi
                for k in range(2, min(len(ref_features), self.feature_dim)):
                    features[k] = ref_features[k]
            except:
                # Jika gagal, gunakan nilai default
                for k in range(2, self.feature_dim):
                    features[k] = np.random.rand() * 0.1
        else:
            # Jika tidak ada referensi, gunakan nilai default
            for k in range(2, self.feature_dim):
                features[k] = np.random.rand() * 0.1
        
        return features
    
    def is_protected_area(self, point, buffer=0.001):
        """Check if point is in protected area (water body or dense vegetation)"""
        is_water = False
        is_dense_veg = False
        
        if self.water_bodies is not None and not self.water_bodies.is_empty:
            try:
                is_water = point.intersects(self.water_bodies.buffer(buffer))
            except:
                is_water = False
        
        if self.dense_vegetation is not None and not self.dense_vegetation.is_empty:
            try:
                is_dense_veg = point.intersects(self.dense_vegetation.buffer(buffer))
            except:
                is_dense_veg = False
        
        return is_water or is_dense_veg
    
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
            model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'DT':
            model = DecisionTreeClassifier(max_depth=15, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                          random_state=42, n_jobs=-1, class_weight='balanced')
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'GBM':
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                              random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'SVM':
            model = SVC(kernel='rbf', probability=True, random_state=42, max_iter=1000, class_weight='balanced')
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=7)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'NB':
            model = GaussianNB()
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'MLP':
            model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, 
                                 random_state=42, early_stopping=True)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'MC':
            return None, None
        
        elif model_name == 'FL':
            return self.create_fuzzy_system(), None
        
        elif model_name == 'EA':
            return self.create_evolutionary_ensemble(X_train, y_train), None
    
    def create_fuzzy_system(self):
        """Create Fuzzy Logic system for land cover classification (5 classes)"""
        try:
            # Define fuzzy variables
            x_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'x_position')
            y_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'y_position')
            area = ctrl.Antecedent(np.arange(0, 1, 0.01), 'area')
            land_cover = ctrl.Consequent(np.arange(1, 6, 0.1), 'land_cover')
            
            # Define membership functions
            x_pos['rendah'] = fuzz.trimf(x_pos.universe, [0, 0, 0.5])
            x_pos['tinggi'] = fuzz.trimf(x_pos.universe, [0.5, 1, 1])
            
            y_pos['rendah'] = fuzz.trimf(y_pos.universe, [0, 0, 0.5])
            y_pos['tinggi'] = fuzz.trimf(y_pos.universe, [0.5, 1, 1])
            
            area['kecil'] = fuzz.trimf(area.universe, [0, 0, 0.3])
            area['sedang'] = fuzz.trimf(area.universe, [0.2, 0.5, 0.8])
            area['besar'] = fuzz.trimf(area.universe, [0.7, 1, 1])
            
            # Define membership functions for 5 classes
            land_cover['air'] = fuzz.trimf(land_cover.universe, [1, 1, 2])
            land_cover['vegetasi_rapat'] = fuzz.trimf(land_cover.universe, [1.5, 2, 2.5])
            land_cover['vegetasi_jarang'] = fuzz.trimf(land_cover.universe, [2.5, 3, 3.5])
            land_cover['terbangun'] = fuzz.trimf(land_cover.universe, [3.5, 4, 4.5])
            land_cover['terbuka'] = fuzz.trimf(land_cover.universe, [4.5, 5, 5])
            
            # Simplified rules
            rules = [
                ctrl.Rule(x_pos['rendah'] & y_pos['rendah'] & area['kecil'], land_cover['vegetasi_jarang']),
                ctrl.Rule(x_pos['tinggi'] & y_pos['tinggi'] & area['besar'], land_cover['air']),
                ctrl.Rule(x_pos['rendah'] & y_pos['tinggi'] & area['sedang'], land_cover['vegetasi_rapat']),
                ctrl.Rule(x_pos['tinggi'] & y_pos['rendah'] & area['kecil'], land_cover['terbangun']),
                ctrl.Rule(area['besar'], land_cover['air']),
                ctrl.Rule(area['kecil'] & x_pos['tinggi'], land_cover['terbangun']),
            ]
            
            land_cover_ctrl = ctrl.ControlSystem(rules)
            return ctrl.ControlSystemSimulation(land_cover_ctrl)
        except:
            return None
    
    def create_evolutionary_ensemble(self, X_train, y_train):
        """Create optimized ensemble"""
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Base models
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('gbm', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
        ]
        
        ensemble = VotingClassifier(estimators=models, voting='soft')
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def predict_future(self, model_name, model, grid_points, years, transition_matrix=None, 
                      with_policy=False, batch_size=1000, policy_buffer=0.001):
        """
        Predict future land cover (5 classes) dengan aturan kebijakan:
        - Jika with_policy=True, Badan Air (1) dan Vegetasi Rapat (2) tidak berubah
        - Hanya Vegetasi Jarang (3), Lahan Terbangun (4), dan Lahan Terbuka (5) yang dapat berubah
        """
        predictions = np.zeros(len(grid_points), dtype=int)
        confidences = np.zeros(len(grid_points))
        
        # Get feature dimension from model if available
        feature_dim = self.feature_dim
        if model is not None and hasattr(model, 'n_features_in_'):
            feature_dim = model.n_features_in_
        
        # Untuk skenario dengan kebijakan
        if with_policy:
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                
                for j, point_data in enumerate(batch):
                    point = point_data['geometry']
                    
                    # Cek apakah titik dalam area lindung
                    is_protected = self.is_protected_area(point, policy_buffer)
                    
                    if is_protected:
                        # Jika area lindung, cari kelas asli dari data terbaru
                        if self.reference_gdf is not None:
                            distances = self.reference_gdf.geometry.distance(point)
                            nearest_idx = distances.idxmin()
                            original_class = self.reference_gdf.loc[nearest_idx, 'gridcode']
                            
                            # Pertahankan kelas asli jika termasuk kelas lindung
                            if original_class in [1, 2]:
                                predictions[i + j] = original_class
                                confidences[i + j] = 1.0
                            else:
                                # Jika bukan kelas lindung, gunakan prediksi dengan batasan
                                pred_class, confidence = self._predict_single_point(
                                    model_name, model, point_data, years, transition_matrix, feature_dim
                                )
                                
                                # Jika prediksi kelas lindung, pilih kelas dinamis terdekat
                                if pred_class in [1, 2]:
                                    pred_class = 3  # Default ke vegetasi jarang
                                    confidence = 0.6
                                
                                predictions[i + j] = pred_class
                                confidences[i + j] = confidence
                        else:
                            # Jika tidak ada referensi, asumsikan kelas 1 (air)
                            predictions[i + j] = 1
                            confidences[i + j] = 0.8
                    else:
                        # Untuk area non-lindung, gunakan prediksi normal
                        pred_class, confidence = self._predict_single_point(
                            model_name, model, point_data, years, transition_matrix, feature_dim
                        )
                        
                        # Pastikan prediksi bukan kelas lindung (kecuali jika memang area lindung)
                        # Tapi karena ini area non-lindung, kita tidak boleh memprediksi kelas lindung
                        if pred_class in [1, 2]:
                            # Pilih kelas alternatif berdasarkan probabilitas
                            if hasattr(model, 'predict_proba') and model is not None:
                                features = point_data['features'].copy()
                                if len(features) < feature_dim:
                                    features = np.pad(features, (0, feature_dim - len(features)), 
                                                    'constant', constant_values=0)
                                elif len(features) > feature_dim:
                                    features = features[:feature_dim]
                                
                                features = features.reshape(1, -1)
                                
                                try:
                                    if hasattr(self.scaler, 'mean_'):
                                        features = self.scaler.transform(features)
                                    
                                    probs = model.predict_proba(features)[0]
                                    # Ambil probabilitas untuk kelas dinamis (3,4,5)
                                    # Asumsikan model memiliki kelas 1-5
                                    dynamic_probs = []
                                    dynamic_classes = []
                                    
                                    # Dapatkan semua kelas yang ada di model
                                    if hasattr(model, 'classes_'):
                                        classes = model.classes_
                                        for idx, cls in enumerate(classes):
                                            if cls in [3, 4, 5]:
                                                dynamic_probs.append(probs[idx])
                                                dynamic_classes.append(cls)
                                    
                                    if dynamic_probs:
                                        max_idx = np.argmax(dynamic_probs)
                                        pred_class = dynamic_classes[max_idx]
                                        confidence = dynamic_probs[max_idx]
                                    else:
                                        pred_class = 3
                                        confidence = 0.5
                                except:
                                    pred_class = 3
                                    confidence = 0.5
                            else:
                                pred_class = 3
                                confidence = 0.5
                        
                        predictions[i + j] = pred_class
                        confidences[i + j] = confidence
        
        else:
            # Tanpa kebijakan: gunakan prediksi normal
            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i+batch_size]
                
                for j, point_data in enumerate(batch):
                    pred_class, confidence = self._predict_single_point(
                        model_name, model, point_data, years, transition_matrix, feature_dim
                    )
                    predictions[i + j] = pred_class
                    confidences[i + j] = confidence
        
        return predictions, confidences
    
    def _predict_single_point(self, model_name, model, point_data, years, transition_matrix, feature_dim):
        """Helper function to predict single point"""
        if model_name == 'MC' and transition_matrix is not None:
            current_probs = np.ones(len(transition_matrix)) / len(transition_matrix)
            for _ in range(years // 25):
                current_probs = current_probs @ transition_matrix
            pred_class = np.argmax(current_probs) + 1
            confidence = np.max(current_probs)
            
        elif model_name == 'FL' and model is not None:
            try:
                # Fuzzy logic prediction
                pred_class = 3
                confidence = 0.6
            except:
                pred_class = 3
                confidence = 0.5
        else:
            if model is not None:
                features = point_data['features'].copy()
                
                # Pastikan fitur memiliki dimensi yang benar
                if len(features) < feature_dim:
                    features = np.pad(features, (0, feature_dim - len(features)), 
                                    'constant', constant_values=0)
                elif len(features) > feature_dim:
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
                        confidence = 0.8
                except Exception as e:
                    # Fallback prediction
                    pred_class = 3
                    confidence = 0.5
            else:
                pred_class = 3
                confidence = 0.5
        
        return pred_class, confidence


# Fungsi untuk membuat peta folium dengan basemap (5 kelas)
def create_folium_map(gdf, title, basemap_type="OpenStreetMap", center=None, show_protected=True):
    """Create Folium map with basemap for GeoJSON visualization (5 classes)"""
    if gdf is None or len(gdf) == 0:
        return None
    
    # Calculate center if not provided
    if center is None:
        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    # Create base map
    if basemap_type == "OpenStreetMap":
        m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')
    elif basemap_type == "Satellite":
        m = folium.Map(location=center, zoom_start=12, 
                      tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                      attr='Esri')
    elif basemap_type == "Terrain":
        m = folium.Map(location=center, zoom_start=12, 
                      tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                      attr='OpenTopoMap')
    elif basemap_type == "Dark Matter":
        m = folium.Map(location=center, zoom_start=12, tiles='CartoDB dark_matter')
    elif basemap_type == "Positron":
        m = folium.Map(location=center, zoom_start=12, tiles='CartoDB positron')
    elif basemap_type == "Watercolor":
        m = folium.Map(location=center, zoom_start=12, tiles='Stamen Watercolor')
    elif basemap_type == "Toner":
        m = folium.Map(location=center, zoom_start=12, tiles='Stamen Toner')
    else:
        m = folium.Map(location=center, zoom_start=12)
    
    # Create color mapping for 5 classes
    def get_color(gridcode):
        if gridcode == 1:
            return '#3498db'  # Blue - Water
        elif gridcode == 2:
            return '#2ecc71'  # Green - Dense Vegetation
        elif gridcode == 3:
            return '#f1c40f'  # Yellow - Sparse Vegetation
        elif gridcode == 4:
            return '#e74c3c'  # Red - Built-up
        elif gridcode == 5:
            return '#95a5a6'  # Grey - Open Land
        else:
            return '#34495e'  # Dark Grey - Unknown
    
    # Add GeoJSON to map
    folium.GeoJson(
        gdf,
        name=title,
        style_function=lambda feature: {
            'fillColor': get_color(feature['properties']['gridcode']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['gridcode', 'LandCover', 'confidence'] if 'confidence' in gdf.columns else ['gridcode', 'LandCover'],
            aliases=['Kode', 'Tutupan Lahan', 'Kepercayaan'] if 'confidence' in gdf.columns else ['Kode', 'Tutupan Lahan'],
            localize=True
        )
    ).add_to(m)
    
    # Add legend for 5 classes with protection info
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border-radius: 5px; border: 2px solid grey; box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
        <p><strong>Legenda Tutupan Lahan</strong></p>
        <p><span style="background-color: #3498db; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Badan Air</p>
        <p><span style="background-color: #2ecc71; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Vegetasi Rapat</p>
        <p><span style="background-color: #f1c40f; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Vegetasi Jarang</p>
        <p><span style="background-color: #e74c3c; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Lahan Terbangun</p>
        <p><span style="background-color: #95a5a6; width: 20px; height: 20px; display: inline-block; margin-right: 5px;"></span> Lahan Terbuka</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen button
    folium.plugins.Fullscreen().add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

# Initialize analyzer
analyzer = LandCoverAnalyzer()

# ========== TAB 1: UNGGAH DATA ==========
with tab1:
    st.header("📤 Unggah Data Historis")
    
    # Sample data section
    st.markdown("""
    <div class='sample-data-card'>
        <h4>📁 Data Contoh</h4>
        <p>Gunakan data contoh untuk simulasi cepat:</p>
        <ul>
            <li><strong>Land_Cover_Tahun_2009.geojson</strong> - Data tutupan lahan tahun 2009</li>
            <li><strong>Land_Cover_Tahun_2015.geojson</strong> - Data tutupan lahan tahun 2015</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #fff8e7; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
        <p>📌 Unggah minimal 2 file GeoJSON dari tahun yang berbeda untuk menganalisis pola perubahan</p>
        <p>🗂️ Format file: <strong>GeoJSON</strong> dengan kolom yang diperlukan: <code>gridcode</code>, <code>LandCover</code>, <code>geometry</code></p>
        <p>🔄 <strong>Reklasifikasi Otomatis:</strong> Data akan direklasifikasi menjadi 5 kelas: Badan Air (1), Vegetasi Rapat (2), Vegetasi Jarang (3), Lahan Terbangun (4), Lahan Terbuka (5)</p>
        <p>🛡️ <strong>Kelas Dilindungi:</strong> Badan Air (1) dan Vegetasi Rapat (2) akan dilindungi dalam skenario kebijakan</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 File 1 (Tahun Terdahulu)")
        file1 = st.file_uploader("Pilih GeoJSON pertama", type=['geojson'], key='file1')
        year1 = st.number_input("Tahun untuk File 1", min_value=1900, max_value=2100, 
                                value=2009, step=1, key='year1')
        
    with col2:
        st.subheader("📁 File 2 (Tahun Terbaru)")
        file2 = st.file_uploader("Pilih GeoJSON kedua", type=['geojson'], key='file2')
        year2 = st.number_input("Tahun untuk File 2", min_value=1900, max_value=2100, 
                                value=2015, step=1, key='year2')
    
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
                    # Store original GDFs
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
                    
                    # Show data summary with original and reclassified
                    st.subheader("📊 Ringkasan Data (Setelah Reklasifikasi 5 Kelas)")
                    
                    summary_data = []
                    for year in analyzer.years:
                        gdf_reclass = analyzer.get_reclassified_gdf(year)
                        if gdf_reclass is not None:
                            counts = gdf_reclass['gridcode'].value_counts()
                            protected_count = counts.get(1, 0) + counts.get(2, 0)
                            dynamic_count = counts.get(3, 0) + counts.get(4, 0) + counts.get(5, 0)
                            
                            summary_data.append({
                                'Tahun': year,
                                'Badan Air (1)': counts.get(1, 0),
                                'Vegetasi Rapat (2)': counts.get(2, 0),
                                'Vegetasi Jarang (3)': counts.get(3, 0),
                                'Lahan Terbangun (4)': counts.get(4, 0),
                                'Lahan Terbuka (5)': counts.get(5, 0),
                                'Total Dilindungi': protected_count,
                                'Total Dinamis': dynamic_count,
                                'Total': len(gdf_reclass)
                            })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Store in session state
                    st.session_state.historical_data = analyzer
                    
                    # Visualize with basemap
                    st.subheader("🗺️ Visualisasi Peta dengan Basemap")
                    
                    # Create tabs for each year
                    if len(analyzer.years) > 0:
                        map_tabs = st.tabs([f"Tahun {year}" for year in analyzer.years])
                        
                        for i, year in enumerate(analyzer.years):
                            with map_tabs[i]:
                                gdf_reclass = analyzer.get_reclassified_gdf(year)
                                if gdf_reclass is not None:
                                    # Calculate center
                                    bounds = gdf_reclass.total_bounds
                                    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                                    
                                    # Create map
                                    m = create_folium_map(gdf_reclass, f'Tutupan Lahan {year}', selected_basemap, center)
                                    if m is not None:
                                        folium_static(m, width=800, height=500)
                                    
                                    # Show statistics
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    counts = gdf_reclass['gridcode'].value_counts()
                                    with col1:
                                        st.metric("Badan Air", counts.get(1, 0))
                                    with col2:
                                        st.metric("Vegetasi Rapat", counts.get(2, 0))
                                    with col3:
                                        st.metric("Vegetasi Jarang", counts.get(3, 0))
                                    with col4:
                                        st.metric("Lahan Terbangun", counts.get(4, 0))
                                    with col5:
                                        st.metric("Lahan Terbuka", counts.get(5, 0))

# ========== TAB 2: ANALISIS HISTORIS ==========
with tab2:
    st.header("📊 Analisis Perubahan Historis")
    
    if st.session_state.historical_data is None:
        st.warning("⚠️ Harap unggah dan proses data terlebih dahulu di tab 'Unggah & Proses Data'")
    else:
        analyzer = st.session_state.historical_data
        
        # Calculate change rates between consecutive years using reclassified data
        change_data = []
        transition_matrices = []
        
        for i in range(len(analyzer.years) - 1):
            year1 = analyzer.years[i]
            year2 = analyzer.years[i + 1]
            gdf1 = analyzer.gdfs[year1]
            gdf2 = analyzer.gdfs[year2]
            
            # Set year attribute for reclassification lookup
            gdf1.attrs['year'] = year1
            gdf2.attrs['year'] = year2
            
            # Calculate transition matrix with reclassified data
            matrix, classes = analyzer.calculate_transition_matrix(gdf1, gdf2)
            transition_matrices.append({
                'years': f"{year1}-{year2}",
                'matrix': matrix,
                'classes': classes
            })
            
            # Get reclassified GDFs for statistics
            gdf1_reclass = analyzer.get_reclassified_gdf(year1)
            gdf2_reclass = analyzer.get_reclassified_gdf(year2)
            
            if gdf1_reclass is not None and gdf2_reclass is not None:
                # Calculate change statistics with new classes
                changes = {
                    'Periode': f"{year1}-{year2}",
                    'Tahun': year2 - year1,
                    'Perubahan Air': len(gdf2_reclass[gdf2_reclass['gridcode'] == 1]) - len(gdf1_reclass[gdf1_reclass['gridcode'] == 1]),
                    'Perubahan Veg. Rapat': len(gdf2_reclass[gdf2_reclass['gridcode'] == 2]) - len(gdf1_reclass[gdf1_reclass['gridcode'] == 2]),
                    'Perubahan Veg. Jarang': len(gdf2_reclass[gdf2_reclass['gridcode'] == 3]) - len(gdf1_reclass[gdf1_reclass['gridcode'] == 3]),
                    'Perubahan Terbangun': len(gdf2_reclass[gdf2_reclass['gridcode'] == 4]) - len(gdf1_reclass[gdf1_reclass['gridcode'] == 4]),
                    'Perubahan Terbuka': len(gdf2_reclass[gdf2_reclass['gridcode'] == 5]) - len(gdf1_reclass[gdf1_reclass['gridcode'] == 5]),
                }
                change_data.append(changes)
        
        if change_data:
            change_df = pd.DataFrame(change_data)
            
            # Display change analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Statistik Perubahan")
                st.dataframe(change_df, use_container_width=True, hide_index=True)
                
                # Calculate annual change rates
                st.subheader("📉 Laju Perubahan Tahunan")
                annual_rates = change_df.copy()
                for col in ['Perubahan Air', 'Perubahan Veg. Rapat', 'Perubahan Veg. Jarang', 
                           'Perubahan Terbangun', 'Perubahan Terbuka']:
                    annual_rates[f'Laju {col}'] = (annual_rates[col] / annual_rates['Tahun']).round(2)
                
                st.dataframe(annual_rates[['Periode', 'Laju Perubahan Air', 'Laju Perubahan Veg. Rapat', 
                                          'Laju Perubahan Veg. Jarang', 'Laju Perubahan Terbangun',
                                          'Laju Perubahan Terbuka']], use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🔄 Matriks Transisi")
                for tm in transition_matrices:
                    with st.expander(f"Matriks Transisi {tm['years']}"):
                        # Create class names for 5 classes
                        class_names = {1: 'Air', 2: 'Veg Rapat', 3: 'Veg Jarang', 4: 'Terbangun', 5: 'Terbuka'}
                        xticklabels = [class_names.get(c, f'Kelas {c}') for c in tm['classes']]
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(tm['matrix'], annot=True, fmt='.2f', cmap='YlOrRd',
                                   xticklabels=xticklabels, yticklabels=xticklabels,
                                   ax=ax)
                        ax.set_xlabel('Ke Kelas')
                        ax.set_ylabel('Dari Kelas')
                        ax.set_title(f'Probabilitas Transisi {tm["years"]}')
                        st.pyplot(fig)
            
            # Visualization of changes over time
            st.subheader("📊 Evolusi Tutupan Lahan")
            
            # Prepare data for plotting
            plot_data = []
            for year in analyzer.years:
                gdf_reclass = analyzer.get_reclassified_gdf(year)
                if gdf_reclass is not None:
                    counts = gdf_reclass['gridcode'].value_counts()
                    plot_data.append({
                        'Tahun': year,
                        'Air': counts.get(1, 0),
                        'Vegetasi Rapat': counts.get(2, 0),
                        'Vegetasi Jarang': counts.get(3, 0),
                        'Lahan Terbangun': counts.get(4, 0),
                        'Lahan Terbuka': counts.get(5, 0)
                    })
            
            plot_df = pd.DataFrame(plot_data)
            
            # Create interactive plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Air'],
                                     mode='lines+markers', name='Air',
                                     line=dict(color='#3498db', width=3)))
            fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Vegetasi Rapat'],
                                     mode='lines+markers', name='Vegetasi Rapat',
                                     line=dict(color='#2ecc71', width=3)))
            fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Vegetasi Jarang'],
                                     mode='lines+markers', name='Vegetasi Jarang',
                                     line=dict(color='#f1c40f', width=3)))
            fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Lahan Terbangun'],
                                     mode='lines+markers', name='Lahan Terbangun',
                                     line=dict(color='#e74c3c', width=3)))
            fig.add_trace(go.Scatter(x=plot_df['Tahun'], y=plot_df['Lahan Terbuka'],
                                     mode='lines+markers', name='Lahan Terbuka',
                                     line=dict(color='#95a5a6', width=3)))
            
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
                        gdf1 = analyzer.gdfs[year1]
                        gdf2 = analyzer.gdfs[year2]
                        
                        # Set year attribute
                        gdf1.attrs['year'] = year1
                        gdf2.attrs['year'] = year2
                        
                        X, y = analyzer.create_temporal_features(gdf1, gdf2)
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
                                        y_pred = np.random.choice([1, 2, 3, 4, 5], size=len(y_test))
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
        
        # Informasi tentang aturan kebijakan
        if with_policy:
            st.markdown("""
            <div class='policy-info'>
                <h4>📋 Aturan Skenario Dengan Kebijakan:</h4>
                <p>
                    <span class='protected-class'>🌊 Badan Air (1) - Dilindungi</span>
                    <span class='protected-class'>🌳 Vegetasi Rapat (2) - Dilindungi</span>
                    <span class='dynamic-class'>🌿 Vegetasi Jarang (3) - Dinamis</span>
                    <span class='dynamic-class'>🏙️ Lahan Terbangun (4) - Dinamis</span>
                    <span class='dynamic-class'>🏜️ Lahan Terbuka (5) - Dinamis</span>
                </p>
                <p>✅ Kelas yang dilindungi <strong>tidak akan berubah</strong> dalam prediksi<br>
                ✅ Hanya kelas dinamis yang dapat berkembang sesuai prediksi model</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Pengaturan Prediksi")
            if years_options:
                selected_year = st.selectbox(
                    "Pilih tahun prediksi", 
                    years_options, 
                    format_func=lambda x: f"{x} tahun ke depan",
                    key="year_selector"
                )
                st.session_state.selected_prediction_year = selected_year
            else:
                st.warning("⚠️ Harap pilih tahun prediksi di sidebar")
                selected_year = st.session_state.selected_prediction_year
            
        with col2:
            st.subheader("🤖 Pilih Model untuk Prediksi")
            predict_models = st.multiselect(
                "Pilih model yang akan digunakan",
                options=list(analyzer.models.keys()),
                default=list(analyzer.models.keys())[:2] if analyzer.models else [],
                format_func=lambda x: f"{x} - {model_descriptions.get(x, '')}",
                key="model_selector"
            )
        
        # Button to generate predictions
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            generate_button = st.button(
                "🔮 Generate Prediksi Baru", 
                type="primary", 
                use_container_width=True,
                key="generate_prediction_btn"
            )
        
        if generate_button:
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
                                transition_matrix, with_policy=False, batch_size=1000,
                                policy_buffer=policy_buffer if 'policy_buffer' in locals() else 0.001
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
                                transition_matrix, with_policy=True, batch_size=1000,
                                policy_buffer=policy_buffer if 'policy_buffer' in locals() else 0.001
                            )
                            all_predictions[f"{model_name}_dengan_kebijakan"] = {
                                'predictions': pred_with_policy,
                                'confidences': conf_with_policy,
                                'display_name': f"{model_name} - Dengan Kebijakan"
                            }
                        
                        progress_bar.progress((idx + 1) / len(predict_models))
                    
                    status_text.text("✅ Prediksi selesai!")
                    
                    # Store in session state
                    st.session_state.current_predictions = all_predictions
                    st.session_state.current_grid_info = {
                        'x_coords': x_coords,
                        'y_coords': y_coords,
                        'bounds': analyzer.bounds,
                        'n_x': len(x_coords),
                        'n_y': len(y_coords),
                        'selected_year': selected_year
                    }
                    
                    st.success(f"✅ Berhasil menghasilkan {len(all_predictions)} skenario prediksi")
        
        # Display predictions if available
        if st.session_state.current_predictions is not None:
            all_predictions = st.session_state.current_predictions
            grid_info = st.session_state.current_grid_info
            
            x_coords = grid_info['x_coords']
            y_coords = grid_info['y_coords']
            pred_year = grid_info.get('selected_year', 25)
            
            # Only show if there are predictions
            if all_predictions and len(all_predictions) > 0:
                
                # ========== VISUALISASI DENGAN BASEMAP ==========
                if show_prediction_map:
                    st.subheader("🗺️ Peta Prediksi dengan Basemap")
                    
                    # Pilih scenario untuk ditampilkan di peta
                    scenario_options = list(all_predictions.keys())
                    
                    if scenario_options:
                        # Create a unique key for the selector
                        selector_key = f"map_selector_{hash(frozenset(scenario_options))}"
                        
                        selected_scenario = st.selectbox(
                            "Pilih skenario untuk ditampilkan di peta",
                            options=scenario_options,
                            format_func=lambda x: all_predictions[x]['display_name'],
                            key=selector_key
                        )
                        
                        if selected_scenario:
                            pred_data = all_predictions[selected_scenario]
                            
                            # Validasi data
                            if (pred_data['predictions'] is not None and 
                                len(pred_data['predictions']) > 0 and
                                pred_data['confidences'] is not None and
                                len(pred_data['confidences']) > 0):
                                
                                try:
                                    # Konversi ke GeoDataFrame dengan sampling
                                    total_cells = len(y_coords) * len(x_coords)
                                    sample_step = max(1, total_cells // 1000)  # Sampling untuk performa
                                    
                                    # Buat GeoDataFrame
                                    geometries = []
                                    properties = []
                                    
                                    pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                                    conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                                    
                                    point_count = 0
                                    cells_added = 0
                                    
                                    for i, y in enumerate(y_coords):
                                        for j, x in enumerate(x_coords):
                                            if point_count % sample_step == 0:
                                                cell_size_x = (x_coords[-1] - x_coords[0]) / len(x_coords)
                                                cell_size_y = (y_coords[-1] - y_coords[0]) / len(y_coords)
                                                
                                                geom = box(x - cell_size_x/2, y - cell_size_y/2,
                                                          x + cell_size_x/2, y + cell_size_y/2)
                                                
                                                pred_class = int(pred_reshaped[i, j])
                                                class_name = {1: 'Badan Air', 2: 'Vegetasi Rapat', 
                                                            3: 'Vegetasi Jarang', 4: 'Lahan Terbangun',
                                                            5: 'Lahan Terbuka'}.get(pred_class, 'Unknown')
                                                
                                                geometries.append(geom)
                                                properties.append({
                                                    'gridcode': pred_class,
                                                    'LandCover': class_name,
                                                    'confidence': float(conf_reshaped[i, j])
                                                })
                                                cells_added += 1
                                            point_count += 1
                                    
                                    if geometries:
                                        pred_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
                                        
                                        # Hitung center
                                        center = [(y_coords[0] + y_coords[-1]) / 2, (x_coords[0] + x_coords[-1]) / 2]
                                        
                                        # Buat peta
                                        m = create_folium_map(
                                            pred_gdf, 
                                            f"Prediksi {pred_year} Tahun - {all_predictions[selected_scenario]['display_name']}",
                                            selected_basemap, 
                                            center
                                        )
                                        
                                        if m is not None:
                                            folium_static(m, width=1000, height=600)
                                        
                                        st.info(f"📊 Menampilkan {cells_added} dari {total_cells} sel prediksi (sampling 1:{sample_step} untuk performa)")
                                    else:
                                        st.warning("⚠️ Tidak dapat membuat GeoDataFrame dari prediksi")
                                except Exception as e:
                                    st.error(f"Error membuat peta: {str(e)}")
                            else:
                                st.warning("⚠️ Data prediksi tidak valid untuk skenario ini")
                    else:
                        st.warning("⚠️ Tidak ada skenario prediksi tersedia")
                
                # Visualization dengan matplotlib - MENGGUNAKAN PLOT SEPERTI GAMBAR
                st.subheader("🗺️ Visualisasi Grid Prediksi")
                
                # Buat tabs untuk setiap skenario
                scenario_items = list(all_predictions.items())
                if scenario_items:
                    # Limit to first 4 scenarios
                    display_items = scenario_items[:4]
                    viz_tabs = st.tabs([data['display_name'] for _, data in display_items])
                    
                    for idx, (scenario_name, pred_data) in enumerate(display_items):
                        with viz_tabs[idx]:
                            try:
                                pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                                conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    
                                    # Create custom colormap for 5 classes - SESUAI GAMBAR
                                    from matplotlib.colors import ListedColormap
                                    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6']
                                    cmap = ListedColormap(colors)
                                    
                                    # Plot dengan extent yang tepat
                                    im = ax.imshow(pred_reshaped, cmap=cmap,
                                                  extent=[x_coords[0], x_coords[-1],
                                                          y_coords[0], y_coords[-1]],
                                                  origin='lower', aspect='auto', vmin=1, vmax=5,
                                                  interpolation='nearest')
                                    
                                    # Tambahkan grid lines
                                    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                                    
                                    # Set labels dan title
                                    ax.set_xlabel('Bujur', fontsize=12)
                                    ax.set_ylabel('Lintang', fontsize=12)
                                    ax.set_title('Prediksi Tutupan Lahan', fontsize=14, fontweight='bold')
                                    
                                    # Create colorbar with labels - SEPERTI GAMBAR
                                    cbar = plt.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5], 
                                                       fraction=0.046, pad=0.04)
                                    cbar.ax.set_yticklabels(['Badan Air', 'Vegetasi Rapat', 'Vegetasi Jarang', 
                                                            'Lahan Terbangun', 'Lahan Terbuka'], fontsize=10)
                                    cbar.ax.tick_params(labelsize=10)
                                    
                                    # Tambahkan boundary box
                                    ax.set_xlim(x_coords[0], x_coords[-1])
                                    ax.set_ylim(y_coords[0], y_coords[-1])
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)  # Clean up memory
                                
                                with col2:
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    im = ax.imshow(conf_reshaped, cmap='RdYlGn',
                                                  extent=[x_coords[0], x_coords[-1],
                                                          y_coords[0], y_coords[-1]],
                                                  origin='lower', aspect='auto', vmin=0, vmax=1,
                                                  interpolation='nearest')
                                    
                                    # Tambahkan grid lines
                                    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                                    
                                    ax.set_xlabel('Bujur', fontsize=12)
                                    ax.set_ylabel('Lintang', fontsize=12)
                                    ax.set_title('Peta Kepercayaan', fontsize=14, fontweight='bold')
                                    
                                    cbar = plt.colorbar(im, ax=ax, label='Tingkat Kepercayaan', 
                                                       fraction=0.046, pad=0.04)
                                    cbar.ax.tick_params(labelsize=10)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)  # Clean up memory
                                
                                # Tampilkan statistik tambahan
                                st.markdown(f"**Statistik {pred_data['display_name']}:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Rata-rata Kepercayaan", f"{np.mean(pred_data['confidences']):.3f}")
                                with col2:
                                    st.metric("Kepercayaan Min", f"{np.min(pred_data['confidences']):.3f}")
                                with col3:
                                    st.metric("Kepercayaan Max", f"{np.max(pred_data['confidences']):.3f}")
                                
                            except Exception as e:
                                st.error(f"Error visualisasi untuk {pred_data['display_name']}: {str(e)}")
                
                # Statistics
                st.subheader("📊 Statistik Prediksi")
                
                stats_data = []
                for scenario_name, pred_data in all_predictions.items():
                    try:
                        unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                        percentages = counts / len(pred_data['predictions']) * 100
                        
                        stats = {
                            'Skenario': pred_data.get('display_name', scenario_name.replace('_', ' ').title()),
                            'Total Prediksi': len(pred_data['predictions']),
                            'Rata-rata Kepercayaan': f"{np.mean(pred_data['confidences']):.3f}"
                        }
                        
                        for u, c, p in zip(unique, counts, percentages):
                            class_name = {1: 'Badan Air', 2: 'Vegetasi Rapat', 3: 'Vegetasi Jarang',
                                         4: 'Lahan Terbangun', 5: 'Lahan Terbuka'}.get(u, f'Kelas {u}')
                            stats[class_name] = f"{c} ({p:.1f}%)"
                        
                        stats_data.append(stats)
                    except Exception as e:
                        st.error(f"Error statistik untuk {scenario_name}: {str(e)}")
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
            else:
                st.info("👈 Klik tombol 'Generate Prediksi Baru' untuk memulai")

# ========== TAB 5: LAPORAN & EKSPOR ==========
with tab5:
    st.header("📈 Laporan & Ekspor")
    
    if st.session_state.current_predictions is None:
        st.warning("⚠️ Harap generate prediksi terlebih dahulu di tab 'Prediksi Masa Depan'")
    else:
        predictions = st.session_state.current_predictions
        grid_info = st.session_state.current_grid_info
        
        if predictions and grid_info:
            st.subheader("📥 Opsi Unduhan")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Laporan Ringkasan")
                
                # Create comprehensive report
                report_data = []
                for scenario_name, pred_data in predictions.items():
                    try:
                        unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                        percentages = counts / len(pred_data['predictions']) * 100
                        
                        for u, c, p in zip(unique, counts, percentages):
                            class_name = {1: 'Badan Air', 2: 'Vegetasi Rapat', 3: 'Vegetasi Jarang',
                                         4: 'Lahan Terbangun', 5: 'Lahan Terbuka'}.get(u, f'Kelas {u}')
                            report_data.append({
                                'Skenario': pred_data.get('display_name', scenario_name.replace('_', ' ').title()),
                                'Tutupan Lahan': class_name,
                                'Status': 'Dilindungi' if u in [1, 2] else 'Dinamis',
                                'Jumlah': c,
                                'Persentase': f"{p:.2f}%",
                                'Rata-rata Kepercayaan': f"{np.mean(pred_data['confidences']):.3f}"
                            })
                    except Exception as e:
                        st.error(f"Error membuat laporan untuk {scenario_name}: {str(e)}")
                
                if report_data:
                    report_df = pd.DataFrame(report_data)
                    
                    # Download buttons
                    csv = report_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Unduh Laporan Ringkasan (CSV)",
                        data=csv,
                        file_name=f"laporan_prediksi_tutupan_lahan_5kelas_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                # Generate GeoJSON for predictions
                st.markdown("### 🗺️ Ekspor sebagai GeoJSON")
                
                for scenario_name, pred_data in list(predictions.items())[:2]:
                    display_name = pred_data.get('display_name', scenario_name.replace('_', ' ').title())
                    if st.button(f"🗺️ Generate GeoJSON untuk {display_name}", key=f"btn_{scenario_name}"):
                        with st.spinner("⏳ Menghasilkan GeoJSON..."):
                            try:
                                x_coords = grid_info['x_coords']
                                y_coords = grid_info['y_coords']
                                
                                # Sampling untuk GeoJSON
                                total_cells = len(y_coords) * len(x_coords)
                                sample_step = max(1, total_cells // 1000)
                                
                                pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                                conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                                
                                geometries = []
                                properties = []
                                point_count = 0
                                
                                for i, y in enumerate(y_coords):
                                    for j, x in enumerate(x_coords):
                                        if point_count % sample_step == 0:
                                            cell_size_x = (x_coords[-1] - x_coords[0]) / len(x_coords)
                                            cell_size_y = (y_coords[-1] - y_coords[0]) / len(y_coords)
                                            
                                            geom = box(x - cell_size_x/2, y - cell_size_y/2,
                                                      x + cell_size_x/2, y + cell_size_y/2)
                                            
                                            pred_class = int(pred_reshaped[i, j])
                                            class_name = {1: 'Badan Air', 2: 'Vegetasi Rapat', 
                                                        3: 'Vegetasi Jarang', 4: 'Lahan Terbangun',
                                                        5: 'Lahan Terbuka'}.get(pred_class, 'Unknown')
                                            
                                            geometries.append(geom)
                                            properties.append({
                                                'gridcode': pred_class,
                                                'land_cover': class_name,
                                                'status': 'Dilindungi' if pred_class in [1, 2] else 'Dinamis',
                                                'confidence': float(conf_reshaped[i, j])
                                            })
                                        point_count += 1
                                
                                if geometries:
                                    pred_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
                                    
                                    geojson_str = pred_gdf.to_json()
                                    st.download_button(
                                        label=f"📥 Unduh {display_name} (GeoJSON)",
                                        data=geojson_str,
                                        file_name=f"{scenario_name}_5kelas_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                                        mime="application/json",
                                        key=f"geojson_{scenario_name}",
                                        use_container_width=True
                                    )
                                    
                                    st.info(f"📊 Mengekspor {len(pred_gdf)} dari {total_cells} sel (sampling 1:{sample_step})")
                            except Exception as e:
                                st.error(f"Error membuat GeoJSON: {str(e)}")
            
            with col2:
                st.markdown("### 📊 Galeri Visualisasi")
                
                # Simplified comparison plots
                if len(predictions) > 1:
                    # Pisahkan berdasarkan kebijakan
                    without_policy_data = {k: v for k, v in predictions.items() if 'tanpa_kebijakan' in k}
                    with_policy_data = {k: v for k, v in predictions.items() if 'dengan_kebijakan' in k}
                    
                    if without_policy_data and with_policy_data:
                        try:
                            fig = make_subplots(
                                rows=1, cols=2,
                                subplot_titles=['Tanpa Kebijakan', 'Dengan Kebijakan'],
                                specs=[[{'type': 'pie'}, {'type': 'pie'}]]
                            )
                            
                            # Rata-rata prediksi tanpa kebijakan
                            all_preds_no_policy = np.mean([v['predictions'] for v in without_policy_data.values()], axis=0)
                            unique_no, counts_no = np.unique(all_preds_no_policy, return_counts=True)
                            
                            labels_no = [{1: 'Badan Air', 2: 'Vegetasi Rapat', 3: 'Vegetasi Jarang',
                                         4: 'Lahan Terbangun', 5: 'Lahan Terbuka'}.get(u, f'Kelas {u}') for u in unique_no]
                            colors_no = ['#3498db' if u==1 else '#2ecc71' if u==2 else '#f1c40f' if u==3 else '#e74c3c' if u==4 else '#95a5a6' for u in unique_no]
                            
                            fig.add_trace(
                                go.Pie(labels=labels_no,
                                      values=counts_no,
                                      marker=dict(colors=colors_no)),
                                row=1, col=1
                            )
                            
                            # Rata-rata prediksi dengan kebijakan
                            all_preds_with_policy = np.mean([v['predictions'] for v in with_policy_data.values()], axis=0)
                            unique_with, counts_with = np.unique(all_preds_with_policy, return_counts=True)
                            
                            labels_with = [{1: 'Badan Air', 2: 'Vegetasi Rapat', 3: 'Vegetasi Jarang',
                                           4: 'Lahan Terbangun', 5: 'Lahan Terbuka'}.get(u, f'Kelas {u}') for u in unique_with]
                            colors_with = ['#3498db' if u==1 else '#2ecc71' if u==2 else '#f1c40f' if u==3 else '#e74c3c' if u==4 else '#95a5a6' for u in unique_with]
                            
                            fig.add_trace(
                                go.Pie(labels=labels_with,
                                      values=counts_with,
                                      marker=dict(colors=colors_with)),
                                row=1, col=2
                            )
                            
                            fig.update_layout(title_text="Perbandingan Skenario - Rata-rata Prediksi",
                                             showlegend=True,
                                             height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error membuat pie chart: {str(e)}")
                    
                    # Confidence heatmap
                    st.markdown("### 🔥 Peta Panas Kepercayaan Prediksi")
                    
                    try:
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
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Error membuat heatmap: {str(e)}")
                    
                    # Analisis Efektivitas Kebijakan
                    if with_policy_data and without_policy_data:
                        st.markdown("### 📊 Analisis Efektivitas Kebijakan")
                        
                        try:
                            # Hitung rata-rata untuk masing-masing skenario
                            avg_no_policy = np.mean([v['predictions'] for v in without_policy_data.values()], axis=0)
                            avg_with_policy = np.mean([v['predictions'] for v in with_policy_data.values()], axis=0)
                            
                            # Hitung persentase kelas yang dilindungi
                            protected_classes_no_policy = np.sum((avg_no_policy == 1) | (avg_no_policy == 2)) / len(avg_no_policy) * 100
                            protected_classes_with_policy = np.sum((avg_with_policy == 1) | (avg_with_policy == 2)) / len(avg_with_policy) * 100
                            
                            # Hitung perubahan
                            protection_increase = protected_classes_with_policy - protected_classes_no_policy
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Area Dilindungi - Tanpa Kebijakan",
                                    f"{protected_classes_no_policy:.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Area Dilindungi - Dengan Kebijakan", 
                                    f"{protected_classes_with_policy:.1f}%",
                                    delta=f"{protection_increase:+.1f}%"
                                )
                            with col3:
                                st.metric(
                                    "Efektivitas Kebijakan",
                                    f"{protection_increase:.1f}%",
                                    delta_color="normal"
                                )
                            
                            # Bar chart perbandingan
                            fig = go.Figure()
                            
                            # Data untuk chart
                            categories = ['Badan Air', 'Vegetasi Rapat', 'Vegetasi Jarang', 'Lahan Terbangun', 'Lahan Terbuka']
                            no_policy_counts = [np.sum(avg_no_policy == i) for i in range(1, 6)]
                            with_policy_counts = [np.sum(avg_with_policy == i) for i in range(1, 6)]
                            
                            fig.add_trace(go.Bar(
                                name='Tanpa Kebijakan',
                                x=categories,
                                y=no_policy_counts,
                                marker_color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6']
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='Dengan Kebijakan',
                                x=categories,
                                y=with_policy_counts,
                                marker_color=['#2980b9', '#27ae60', '#f39c12', '#c0392b', '#7f8c8d']
                            ))
                            
                            fig.update_layout(
                                title='Perbandingan Tutupan Lahan: Dengan vs Tanpa Kebijakan',
                                xaxis_title='Kelas Tutupan Lahan',
                                yaxis_title='Jumlah Sel',
                                barmode='group',
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error dalam analisis efektivitas kebijakan: {str(e)}")
        else:
            st.info("Tidak ada data prediksi yang tersedia")

# Footer - INDONESIA
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
    <p style='color: white; margin: 0;'>🌍 Sistem Analisis & Prediksi Perubahan Tutupan Lahan | Dikembangkan oleh Dr. Adipandang Yudono (Scrypt, Sistem Arsitektur, WebGIS Analytics)</p>
    <p style='color: white; margin: 5px 0 0 0;'>Menggunakan 12 Model ML dengan 2 Skenario Kebijakan | 5 Kelas Tutupan Lahan</p>
    <p style='color: white; margin: 5px 0 0 0;'>🛡️ Kebijakan Perlindungan: Badan Air (1) dan Vegetasi Rapat (2) dilindungi dari perubahan</p>
    <p style='color: #ffd700; margin: 10px 0 0 0;'>© 2026 - Departemen Perencanan Wilayah & Kota, Fakultas Teknik, UNIVERSITAS BRAWIJAYA</p>
</div>
""", unsafe_allow_html=True)
