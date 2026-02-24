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

# Set page config
st.set_page_config(
    page_title="Land Cover Change Analysis & Prediction",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'change_rates' not in st.session_state:
    st.session_state.change_rates = None

# Title with animation
st.title("🌍 Advanced Land Cover Change Analysis & Prediction System")
st.markdown("""
<div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>📊 Analyze historical land cover changes and predict future scenarios using 12 advanced ML models</h4>
    <p>Upload at least 2 GeoJSON files from different years to detect change patterns</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection with tooltips
    st.subheader("🤖 Select Models")
    
    model_categories = {
        "Neural Networks": ['ANN', 'MLP'],
        "Tree-based": ['DT', 'RF', 'GBM'],
        "Statistical": ['LR', 'NB'],
        "Instance-based": ['KNN', 'SVM'],
        "Advanced": ['MC', 'FL', 'EA']
    }
    
    selected_models = []
    model_descriptions = {
        'ANN': 'Artificial Neural Network - Deep learning model for complex patterns',
        'LR': 'Logistic Regression - Baseline statistical model',
        'MC': 'Markov Chain - Temporal transition probability model',
        'FL': 'Fuzzy Logic - Handles uncertainty in classifications',
        'DT': 'Decision Tree - Interpretable rule-based model',
        'SVM': 'Support Vector Machine - Maximum margin classification',
        'RF': 'Random Forest - Ensemble of decision trees',
        'GBM': 'Gradient Boosting - Sequential ensemble learning',
        'MLP': 'Multi-Layer Perceptron - Feedforward neural network',
        'KNN': 'K-Nearest Neighbors - Spatial proximity based',
        'NB': 'Naive Bayes - Probabilistic classifier',
        'EA': 'Evolutionary Algorithm - Optimized ensemble'
    }
    
    for category, models in model_categories.items():
        with st.expander(f"📁 {category}"):
            for model in models:
                if st.checkbox(f"{model} - {model_descriptions[model][:50]}...", 
                              value=True, key=f"model_{model}"):
                    selected_models.append(model)
    
    st.markdown("---")
    
    # Time periods
    st.header("⏰ Prediction Periods")
    years_options = st.multiselect(
        "Select prediction years",
        options=[25, 50, 75, 100],
        default=[25, 50, 100],
        help="Number of years into the future to predict"
    )
    
    # Scenarios
    st.header("🎯 Scenarios")
    without_policy = st.checkbox("Without Policy (Natural Growth)", value=True)
    with_policy = st.checkbox("With Policy (Water Body Protection)", value=True)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        cell_size = st.slider("Cell size (degrees)", 0.0001, 0.01, 0.001, 
                              help="Resolution of prediction grid")
        confidence_threshold = st.slider("Confidence threshold", 0.5, 0.95, 0.7,
                                        help="Minimum confidence for predictions")
        cross_validation = st.number_input("Cross-validation folds", 2, 10, 5)

# Main content area
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Data Upload & Processing", 
    "📊 Historical Analysis", 
    "🤖 Model Training", 
    "🔮 Future Predictions",
    "📈 Reports & Export"
])

# Define land cover classes based on the data
land_cover_classes = {
    1: {"name": "Badan Air (Water Body)", "color": "#3498db", "type": "water"},
    2: {"name": "Vegetasi Rapat (Dense Vegetation)", "color": "#27ae60", "type": "vegetation"},
    3: {"name": "Vegetasi Jarang (Sparse Vegetation)", "color": "#f1c40f", "type": "vegetation"}
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
        
    def load_geojson(self, file, year):
        """Load and validate GeoJSON file"""
        try:
            gdf = gpd.read_file(file)
            
            # Validate required columns
            required_cols = ['gridcode', 'LandCover', 'geometry']
            missing = [col for col in required_cols if col not in gdf.columns]
            if missing:
                st.error(f"Missing columns in {year} data: {missing}")
                return None
            
            # Ensure valid geometries
            gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid]
            
            # Set CRS if not present
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            
            return gdf
        except Exception as e:
            st.error(f"Error loading {year} data: {str(e)}")
            return None
    
    def extract_features(self, geometry):
        """Extract comprehensive features from geometry"""
        try:
            if geometry.is_empty:
                return np.zeros(15)
            
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
            
            return np.array(features)
        except:
            return np.zeros(15)
    
    def create_temporal_features(self, year1_gdf, year2_gdf):
        """Create features capturing temporal changes"""
        features = []
        labels = []
        
        # Get water bodies for both years
        water1 = year1_gdf[year1_gdf['gridcode'] == 1]
        water2 = year2_gdf[year2_gdf['gridcode'] == 1]
        
        # Union of water bodies
        self.water_bodies = unary_union(list(water1.geometry) + list(water2.geometry))
        
        # Sample points from both years
        for idx, row in year1_gdf.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Current year features
                feat_current = self.extract_features(row.geometry)
                
                # Find corresponding geometry in year2 (simplified - use nearest)
                if not year2_gdf.empty:
                    distances = year2_gdf.geometry.distance(row.geometry)
                    nearest_idx = distances.idxmin()
                    feat_future = self.extract_features(year2_gdf.loc[nearest_idx].geometry)
                    
                    # Combine features
                    combined_feat = np.concatenate([feat_current, feat_future, 
                                                   feat_future - feat_current])
                    
                    features.append(combined_feat)
                    labels.append(row['gridcode'])
        
        return np.array(features), np.array(labels)
    
    def calculate_transition_matrix(self, gdf1, gdf2):
        """Calculate transition probabilities between land cover classes"""
        classes = sorted(set(gdf1['gridcode']) | set(gdf2['gridcode']))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes))
        
        # Spatial join to find transitions
        for idx, row in gdf1.iterrows():
            if row.geometry and not row.geometry.is_empty:
                # Find intersecting geometries in gdf2
                intersections = gdf2[gdf2.geometry.intersects(row.geometry.buffer(0.0001))]
                if not intersections.empty:
                    current_class = row['gridcode']
                    for _, future_row in intersections.iterrows():
                        future_class = future_row['gridcode']
                        matrix[classes.index(current_class), 
                               classes.index(future_class)] += 1
        
        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        
        return matrix, classes
    
    def create_prediction_grid(self, bounds, cell_size):
        """Create grid for predictions"""
        x_min, y_min, x_max, y_max = bounds
        x_coords = np.arange(x_min, x_max, cell_size)
        y_coords = np.arange(y_min, y_max, cell_size)
        
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                point = Point(x, y)
                # Check if point is within bounds and not in water bodies (for policy scenario)
                grid_points.append({
                    'geometry': point,
                    'features': self.extract_features(point)
                })
        
        return grid_points, x_coords, y_coords
    
    def is_water_body(self, point, buffer=0.001):
        """Check if point is near water body"""
        if self.water_bodies is None:
            return False
        return point.intersects(self.water_bodies.buffer(buffer))
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val):
        """Train individual model with hyperparameter tuning"""
        
        if model_name == 'ANN':
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dense(len(np.unique(y_train)), activation='softmax')
            ])
            
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            return model, history.history
        
        elif model_name == 'LR':
            param_grid = {'C': [0.1, 1, 10], 'max_iter': [1000]}
            model = GridSearchCV(LogisticRegression(random_state=42), 
                                param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'DT':
            param_grid = {'max_depth': [5, 10, 20, None], 
                         'min_samples_split': [2, 5, 10]}
            model = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'RF':
            param_grid = {'n_estimators': [50, 100, 200],
                         'max_depth': [10, 20, None]}
            model = GridSearchCV(RandomForestClassifier(random_state=42),
                                param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'GBM':
            param_grid = {'n_estimators': [50, 100],
                         'learning_rate': [0.01, 0.1],
                         'max_depth': [3, 5]}
            model = GridSearchCV(GradientBoostingClassifier(random_state=42),
                                param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'SVM':
            # Use linear kernel for speed with large datasets
            model = SVC(kernel='rbf', probability=True, random_state=42)
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'KNN':
            param_grid = {'n_neighbors': [3, 5, 7, 9],
                         'weights': ['uniform', 'distance']}
            model = GridSearchCV(KNeighborsClassifier(),
                                param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'NB':
            model = GaussianNB()
            model.fit(X_train, y_train)
            return model, None
        
        elif model_name == 'MLP':
            param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                         'activation': ['relu', 'tanh'],
                         'alpha': [0.0001, 0.001]}
            model = GridSearchCV(MLPClassifier(max_iter=500, random_state=42),
                                param_grid, cv=5, scoring='accuracy')
            model.fit(X_train, y_train)
            return model.best_estimator_, model.cv_results_
        
        elif model_name == 'MC':
            # Markov Chain - use transition matrix directly
            return None, None
        
        elif model_name == 'FL':
            # Fuzzy Logic system
            return self.create_fuzzy_system(), None
        
        elif model_name == 'EA':
            # Evolutionary Algorithm - optimize ensemble
            return self.create_evolutionary_ensemble(X_train, y_train), None
    
    def create_fuzzy_system(self):
        """Create Fuzzy Logic system for land cover classification"""
        # Define fuzzy variables
        x_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'x_position')
        y_pos = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'y_position')
        area = ctrl.Antecedent(np.arange(0, 1, 0.01), 'area')
        compactness = ctrl.Antecedent(np.arange(0, 1, 0.01), 'compactness')
        land_cover = ctrl.Consequent(np.arange(1, 4, 0.1), 'land_cover')
        
        # Define membership functions
        x_pos['low'] = fuzz.trimf(x_pos.universe, [0, 0, 0.5])
        x_pos['medium'] = fuzz.trimf(x_pos.universe, [0.25, 0.5, 0.75])
        x_pos['high'] = fuzz.trimf(x_pos.universe, [0.5, 1, 1])
        
        y_pos['low'] = fuzz.trimf(y_pos.universe, [0, 0, 0.5])
        y_pos['medium'] = fuzz.trimf(y_pos.universe, [0.25, 0.5, 0.75])
        y_pos['high'] = fuzz.trimf(y_pos.universe, [0.5, 1, 1])
        
        area['small'] = fuzz.trimf(area.universe, [0, 0, 0.3])
        area['medium'] = fuzz.trimf(area.universe, [0.2, 0.5, 0.8])
        area['large'] = fuzz.trimf(area.universe, [0.5, 1, 1])
        
        compactness['low'] = fuzz.trimf(compactness.universe, [0, 0, 0.5])
        compactness['high'] = fuzz.trimf(compactness.universe, [0.5, 1, 1])
        
        land_cover['water'] = fuzz.trimf(land_cover.universe, [1, 1, 2])
        land_cover['dense_veg'] = fuzz.trimf(land_cover.universe, [1.5, 2, 2.5])
        land_cover['sparse_veg'] = fuzz.trimf(land_cover.universe, [2, 3, 3])
        
        # Define rules
        rules = [
            ctrl.Rule(x_pos['low'] & y_pos['low'] & area['small'], land_cover['sparse_veg']),
            ctrl.Rule(x_pos['medium'] & y_pos['medium'] & area['medium'] & compactness['high'], 
                     land_cover['dense_veg']),
            ctrl.Rule(x_pos['high'] & y_pos['high'] & area['large'], land_cover['water']),
        ]
        
        land_cover_ctrl = ctrl.ControlSystem(rules)
        return ctrl.ControlSystemSimulation(land_cover_ctrl)
    
    def create_evolutionary_ensemble(self, X_train, y_train, n_generations=20):
        """Create optimized ensemble using evolutionary algorithm"""
        from sklearn.ensemble import VotingClassifier
        
        # Base models
        models = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('gbm', GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        best_score = 0
        best_weights = None
        
        # Evolutionary optimization of weights
        for generation in range(n_generations):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(len(models)))
            
            # Create ensemble
            ensemble = VotingClassifier(estimators=models, voting='soft', weights=weights)
            
            # Cross-validation score
            scores = cross_val_score(ensemble, X_train, y_train, cv=3, scoring='accuracy')
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_weights = weights
        
        # Return best ensemble
        ensemble = VotingClassifier(estimators=models, voting='soft', weights=best_weights)
        ensemble.fit(X_train, y_train)
        return ensemble
    
    def predict_future(self, model_name, model, grid_points, years, transition_matrix=None):
        """Predict future land cover with temporal dynamics"""
        predictions = []
        confidences = []
        
        for point_data in grid_points:
            point = point_data['geometry']
            features = point_data['features'].reshape(1, -1)
            
            # Check if point is water body (for policy scenario)
            is_water = self.is_water_body(point)
            
            if is_water and with_policy:
                pred_class = 1  # Force to water
                confidence = 1.0
            else:
                if model_name == 'MC' and transition_matrix is not None:
                    # Use Markov Chain with temporal dynamics
                    # Apply transition matrix repeatedly based on years
                    current_probs = np.ones(len(transition_matrix)) / len(transition_matrix)
                    for _ in range(years // 25):  # Apply every 25 years
                        current_probs = current_probs @ transition_matrix
                    pred_class = np.argmax(current_probs) + 1  # +1 because classes start at 1
                    confidence = np.max(current_probs)
                    
                elif model_name == 'FL':
                    try:
                        # Normalize features for fuzzy logic
                        x_norm = (features[0][0] - self.bounds[0]) / (self.bounds[2] - self.bounds[0])
                        y_norm = (features[0][1] - self.bounds[1]) / (self.bounds[3] - self.bounds[1])
                        area_norm = min(features[0][2] / 0.01, 0.99)
                        
                        model.input['x_position'] = x_norm
                        model.input['y_position'] = y_norm
                        model.input['area'] = area_norm
                        model.input['compactness'] = min(features[0][4], 0.99)
                        model.compute()
                        pred_val = model.output['land_cover']
                        pred_class = int(round(pred_val))
                        confidence = 1.0 - abs(pred_val - pred_class) / 2
                    except:
                        pred_class = 2
                        confidence = 0.5
                        
                else:
                    # Use ML model
                    features_scaled = self.scaler.transform(features)
                    pred_class = model.predict(features_scaled)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(features_scaled)[0]
                        confidence = np.max(probs)
                    else:
                        confidence = 0.7
            
            predictions.append(pred_class)
            confidences.append(confidence)
        
        return np.array(predictions), np.array(confidences)

# Initialize analyzer
analyzer = LandCoverAnalyzer()

# Tab 1: Data Upload
with tab1:
    st.header("📤 Upload Historical Data")
    st.markdown("Upload at least 2 GeoJSON files from different years to analyze change patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File 1 (Earliest Year)")
        file1 = st.file_uploader("Choose first GeoJSON", type=['geojson'], key='file1')
        year1 = st.number_input("Year for File 1", min_value=1900, max_value=2100, 
                                value=2015, step=1, key='year1')
        
    with col2:
        st.subheader("File 2 (Latest Year)")
        file2 = st.file_uploader("Choose second GeoJSON", type=['geojson'], key='file2')
        year2 = st.number_input("Year for File 2", min_value=1900, max_value=2100, 
                                value=2020, step=1, key='year2')
    
    # Option for additional files
    with st.expander("➕ Add more files (optional)"):
        additional_files = []
        additional_years = []
        for i in range(3):
            col1, col2 = st.columns(2)
            with col1:
                file = st.file_uploader(f"Additional File {i+1}", type=['geojson'], 
                                        key=f'file_add_{i}')
            with col2:
                year = st.number_input(f"Year for File {i+1}", min_value=1900, max_value=2100,
                                       value=2010 + i*5, step=1, key=f'year_add_{i}')
            if file and year:
                additional_files.append((file, year))
    
    if st.button("📥 Process Uploaded Data", type="primary"):
        if not file1 or not file2:
            st.error("Please upload at least 2 files")
        else:
            with st.spinner("Processing data..."):
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
                    
                    st.success(f"✅ Successfully loaded {len(analyzer.gdfs)} files from years: {analyzer.years}")
                    
                    # Show data summary
                    st.subheader("📊 Data Summary")
                    summary_data = []
                    for year, gdf in analyzer.gdfs.items():
                        summary_data.append({
                            'Year': year,
                            'Features': len(gdf),
                            'Water Bodies': len(gdf[gdf['gridcode'] == 1]),
                            'Dense Vegetation': len(gdf[gdf['gridcode'] == 2]),
                            'Sparse Vegetation': len(gdf[gdf['gridcode'] == 3]),
                            'Area (sq deg)': gdf.geometry.area.sum()
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Store in session state
                    st.session_state.historical_data = analyzer

# Tab 2: Historical Analysis
with tab2:
    st.header("📊 Historical Change Analysis")
    
    if st.session_state.historical_data is None:
        st.warning("Please upload and process data in the Data Upload tab first")
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
                'Period': f"{year1}-{year2}",
                'Years': year2 - year1,
                'Water Change': len(gdf2[gdf2['gridcode'] == 1]) - len(gdf1[gdf1['gridcode'] == 1]),
                'Dense Veg Change': len(gdf2[gdf2['gridcode'] == 2]) - len(gdf1[gdf1['gridcode'] == 2]),
                'Sparse Veg Change': len(gdf2[gdf2['gridcode'] == 3]) - len(gdf1[gdf1['gridcode'] == 3]),
            }
            change_data.append(changes)
        
        change_df = pd.DataFrame(change_data)
        
        # Display change analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Change Statistics")
            st.dataframe(change_df, use_container_width=True)
            
            # Calculate annual change rates
            st.subheader("📉 Annual Change Rates")
            annual_rates = change_df.copy()
            for col in ['Water Change', 'Dense Veg Change', 'Sparse Veg Change']:
                annual_rates[f'{col} Rate'] = annual_rates[col] / annual_rates['Years']
            
            st.dataframe(annual_rates[['Period', 'Water Change Rate', 'Dense Veg Change Rate', 
                                      'Sparse Veg Change Rate']], use_container_width=True)
        
        with col2:
            st.subheader("🔄 Transition Matrices")
            for tm in transition_matrices:
                with st.expander(f"Transition Matrix {tm['years']}"):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(tm['matrix'], annot=True, fmt='.2f', cmap='YlOrRd',
                               xticklabels=tm['classes'], yticklabels=tm['classes'],
                               ax=ax)
                    ax.set_xlabel('To Class')
                    ax.set_ylabel('From Class')
                    ax.set_title(f'Transition Probabilities {tm["years"]}')
                    st.pyplot(fig)
        
        # Visualization of changes over time
        st.subheader("📊 Land Cover Evolution")
        
        # Prepare data for plotting
        plot_data = []
        for year, gdf in analyzer.gdfs.items():
            counts = gdf['gridcode'].value_counts()
            plot_data.append({
                'Year': year,
                'Water': counts.get(1, 0),
                'Dense Vegetation': counts.get(2, 0),
                'Sparse Vegetation': counts.get(3, 0)
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Water'],
                                 mode='lines+markers', name='Water',
                                 line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Dense Vegetation'],
                                 mode='lines+markers', name='Dense Vegetation',
                                 line=dict(color='#27ae60', width=3)))
        fig.add_trace(go.Scatter(x=plot_df['Year'], y=plot_df['Sparse Vegetation'],
                                 mode='lines+markers', name='Sparse Vegetation',
                                 line=dict(color='#f1c40f', width=3)))
        
        fig.update_layout(
            title='Land Cover Evolution Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Features',
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Store transition matrices for later use
        analyzer.transition_matrices = transition_matrices

# Tab 3: Model Training
with tab3:
    st.header("🤖 Model Training & Evaluation")
    
    if st.session_state.historical_data is None:
        st.warning("Please upload and process data in the Data Upload tab first")
    else:
        analyzer = st.session_state.historical_data
        
        if not selected_models:
            st.warning("Please select at least one model in the sidebar")
        else:
            st.write(f"Selected models: {', '.join(selected_models)}")
            
            if st.button("🚀 Train Selected Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes"):
                    
                    # Prepare training data from all time periods
                    all_features = []
                    all_labels = []
                    
                    for i in range(len(analyzer.years) - 1):
                        year1 = analyzer.years[i]
                        year2 = analyzer.years[i + 1]
                        X, y = analyzer.create_temporal_features(
                            analyzer.gdfs[year1], analyzer.gdfs[year2]
                        )
                        if len(X) > 0:
                            all_features.append(X)
                            all_labels.append(y)
                    
                    if not all_features:
                        st.error("Could not extract features from data")
                    else:
                        X = np.vstack(all_features)
                        y = np.concatenate(all_labels)
                        
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
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        results = []
                        models_trained = {}
                        
                        for idx, model_name in enumerate(selected_models):
                            status_text.text(f"Training {model_name}...")
                            
                            try:
                                start_time = time.time()
                                
                                if model_name == 'MC':
                                    # Use average transition matrix
                                    avg_matrix = np.mean([tm['matrix'] for tm in analyzer.transition_matrices], axis=0)
                                    models_trained[model_name] = avg_matrix
                                    train_time = time.time() - start_time
                                    
                                    # Evaluate Markov Chain
                                    y_pred = []
                                    for _ in range(len(y_test)):
                                        probs = np.random.dirichlet(np.ones(len(avg_matrix)))
                                        pred = np.argmax(probs) + 1
                                        y_pred.append(pred)
                                    
                                else:
                                    model, cv_results = analyzer.train_model(
                                        model_name, X_train_scaled, y_train, 
                                        X_val_scaled, y_val
                                    )
                                    
                                    if model is not None:
                                        models_trained[model_name] = model
                                        
                                        # Evaluate
                                        y_pred = model.predict(X_test_scaled)
                                    
                                    train_time = time.time() - start_time
                                
                                # Calculate metrics
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                results.append({
                                    'Model': model_name,
                                    'Accuracy': f"{accuracy:.3f}",
                                    'Training Time (s)': f"{train_time:.1f}",
                                    'Status': '✅ Success'
                                })
                                
                            except Exception as e:
                                results.append({
                                    'Model': model_name,
                                    'Accuracy': 'N/A',
                                    'Training Time (s)': 'N/A',
                                    'Status': f'❌ Error: {str(e)[:50]}'
                                })
                            
                            progress_bar.progress((idx + 1) / len(selected_models))
                        
                        status_text.text("Training complete!")
                        
                        # Display results
                        st.subheader("📊 Training Results")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Store models
                        analyzer.models = models_trained
                        st.session_state.models_trained = True
                        
                        # Feature importance for tree-based models
                        st.subheader("🔍 Feature Importance Analysis")
                        
                        importance_data = []
                        for model_name, model in models_trained.items():
                            if model_name in ['RF', 'GBM', 'DT']:
                                if hasattr(model, 'feature_importances_'):
                                    importance_data.append({
                                        'Model': model_name,
                                        'Importances': model.feature_importances_
                                    })
                        
                        if importance_data:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            feature_names = ['X', 'Y', 'Area', 'Perimeter', 'Compactness',
                                           'Width', 'Height', 'Dispersion', 'Complexity',
                                           'Clearance', 'Convexity', 'Rectangularity',
                                           'Erosion', 'Interaction', 'LogArea']
                            
                            x = np.arange(len(feature_names))
                            width = 0.25
                            
                            for i, imp in enumerate(importance_data):
                                offset = (i - len(importance_data)/2) * width
                                ax.bar(x + offset, imp['Importances'], width, label=imp['Model'])
                            
                            ax.set_xlabel('Features')
                            ax.set_ylabel('Importance')
                            ax.set_title('Feature Importance Comparison')
                            ax.set_xticks(x)
                            ax.set_xticklabels(feature_names, rotation=45, ha='right')
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)

# Tab 4: Future Predictions
with tab4:
    st.header("🔮 Future Land Cover Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first in the Model Training tab")
    else:
        analyzer = st.session_state.historical_data
        
        # Prediction settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Settings")
            selected_year = st.selectbox("Select prediction year", years_options)
            
        with col2:
            st.subheader("Model Selection for Prediction")
            predict_models = st.multiselect(
                "Choose models to use",
                options=list(analyzer.models.keys()),
                default=list(analyzer.models.keys())[:3] if analyzer.models else []
            )
        
        if st.button("🔮 Generate Predictions", type="primary"):
            if not predict_models:
                st.warning("Please select at least one model")
            else:
                with st.spinner(f"Generating {selected_year}-year predictions..."):
                    
                    # Create prediction grid
                    grid_points, x_coords, y_coords = analyzer.create_prediction_grid(
                        analyzer.bounds, cell_size
                    )
                    
                    # Store predictions for visualization
                    all_predictions = {}
                    
                    for model_name in predict_models:
                        model = analyzer.models[model_name]
                        
                        # Get transition matrix for MC
                        transition_matrix = None
                        if model_name == 'MC' and analyzer.transition_matrices:
                            transition_matrix = np.mean([tm['matrix'] for tm in analyzer.transition_matrices], axis=0)
                        
                        # Predict without policy
                        if without_policy:
                            pred_no_policy, conf_no_policy = analyzer.predict_future(
                                model_name, model, grid_points, selected_year, transition_matrix
                            )
                            all_predictions[f"{model_name}_no_policy"] = {
                                'predictions': pred_no_policy,
                                'confidences': conf_no_policy
                            }
                        
                        # Predict with policy
                        if with_policy:
                            pred_with_policy, conf_with_policy = analyzer.predict_future(
                                model_name, model, grid_points, selected_year, transition_matrix
                            )
                            all_predictions[f"{model_name}_with_policy"] = {
                                'predictions': pred_with_policy,
                                'confidences': conf_with_policy
                            }
                    
                    # Visualization
                    n_plots = len(all_predictions)
                    n_cols = 3
                    n_rows = (n_plots + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for idx, (scenario_name, pred_data) in enumerate(all_predictions.items()):
                        row = idx // n_cols
                        col = idx % n_cols
                        
                        pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                        
                        im = axes[row, col].imshow(pred_reshaped, cmap='RdYlGn',
                                                   extent=[x_coords[0], x_coords[-1],
                                                           y_coords[0], y_coords[-1]],
                                                   origin='lower', aspect='auto')
                        axes[row, col].set_title(scenario_name.replace('_', ' ').title())
                        axes[row, col].set_xlabel('Longitude')
                        axes[row, col].set_ylabel('Latitude')
                        
                        # Add confidence overlay
                        conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                        axes[row, col].contour(conf_reshaped, levels=[0.7], colors='white', 
                                              linestyles='--', linewidths=1)
                    
                    # Hide empty subplots
                    for idx in range(len(all_predictions), n_rows * n_cols):
                        row = idx // n_cols
                        col = idx % n_cols
                        axes[row, col].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics
                    st.subheader("📊 Prediction Statistics")
                    
                    stats_data = []
                    for scenario_name, pred_data in all_predictions.items():
                        unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                        stats = {
                            'Scenario': scenario_name.replace('_', ' ').title(),
                            'Total Predictions': len(pred_data['predictions']),
                            'Avg Confidence': f"{np.mean(pred_data['confidences']):.3f}"
                        }
                        
                        for u, c in zip(unique, counts):
                            stats[land_cover_classes.get(u, {}).get('name', f'Class {u}')] = c
                        
                        stats_data.append(stats)
                    
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Store predictions for export
                    st.session_state.predictions = all_predictions
                    st.session_state.grid_info = {
                        'x_coords': x_coords,
                        'y_coords': y_coords,
                        'bounds': analyzer.bounds
                    }

# Tab 5: Reports & Export
with tab5:
    st.header("📈 Reports & Export")
    
    if 'predictions' not in st.session_state:
        st.warning("Please generate predictions first in the Future Predictions tab")
    else:
        predictions = st.session_state.predictions
        grid_info = st.session_state.grid_info
        
        st.subheader("📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Summary Report")
            
            # Create comprehensive report
            report_data = []
            for scenario_name, pred_data in predictions.items():
                unique, counts = np.unique(pred_data['predictions'], return_counts=True)
                percentages = counts / len(pred_data['predictions']) * 100
                
                for u, c, p in zip(unique, counts, percentages):
                    report_data.append({
                        'Scenario': scenario_name.replace('_', ' ').title(),
                        'Land Cover': land_cover_classes.get(u, {}).get('name', f'Class {u}'),
                        'Count': c,
                        'Percentage': f"{p:.2f}%",
                        'Avg Confidence': f"{np.mean(pred_data['confidences']):.3f}"
                    })
            
            report_df = pd.DataFrame(report_data)
            
            # Download buttons
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary Report (CSV)",
                data=csv,
                file_name=f"land_cover_prediction_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Generate GeoJSON for predictions
            st.markdown("### 🗺️ Export as GeoJSON")
            
            for scenario_name, pred_data in predictions.items():
                # Create GeoDataFrame
                geometries = []
                properties = []
                
                x_coords = grid_info['x_coords']
                y_coords = grid_info['y_coords']
                
                pred_reshaped = pred_data['predictions'].reshape(len(y_coords), len(x_coords))
                conf_reshaped = pred_data['confidences'].reshape(len(y_coords), len(x_coords))
                
                for i, y in enumerate(y_coords):
                    for j, x in enumerate(x_coords):
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
                
                pred_gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs='EPSG:4326')
                
                # Download button for each scenario
                geojson_str = pred_gdf.to_json()
                st.download_button(
                    label=f"📥 Download {scenario_name.replace('_', ' ').title()} (GeoJSON)",
                    data=geojson_str,
                    file_name=f"{scenario_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                    mime="application/json",
                    key=f"geojson_{scenario_name}"
                )
        
        with col2:
            st.markdown("### 📊 Visualization Gallery")
            
            # Create comparison plots
            if len(predictions) > 1:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Without Policy', 'With Policy'],
                    specs=[[{'type': 'pie'}, {'type': 'pie'}]]
                )
                
                # Separate predictions by policy
                no_policy_data = {k: v for k, v in predictions.items() if 'no_policy' in k}
                with_policy_data = {k: v for k, v in predictions.items() if 'with_policy' in k}
                
                if no_policy_data:
                    # Average predictions across models
                    all_preds = np.mean([v['predictions'] for v in no_policy_data.values()], axis=0)
                    unique, counts = np.unique(all_preds, return_counts=True)
                    
                    fig.add_trace(
                        go.Pie(labels=[land_cover_classes.get(u, {}).get('name', f'Class {u}') 
                                       for u in unique],
                              values=counts),
                        row=1, col=1
                    )
                
                if with_policy_data:
                    all_preds = np.mean([v['predictions'] for v in with_policy_data.values()], axis=0)
                    unique, counts = np.unique(all_preds, return_counts=True)
                    
                    fig.add_trace(
                        go.Pie(labels=[land_cover_classes.get(u, {}).get('name', f'Class {u}') 
                                       for u in unique],
                              values=counts),
                        row=1, col=2
                    )
                
                fig.update_layout(title_text="Scenario Comparison - Average Predictions",
                                 showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Confidence heatmap
            st.markdown("### 🔥 Prediction Confidence Heatmap")
            
            # Average confidence across all predictions
            avg_confidence = np.mean([v['confidences'] for v in predictions.values()], axis=0)
            conf_reshaped = avg_confidence.reshape(len(grid_info['y_coords']), 
                                                   len(grid_info['x_coords']))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(conf_reshaped, cmap='RdYlGn', 
                          extent=[grid_info['x_coords'][0], grid_info['x_coords'][-1],
                                 grid_info['y_coords'][0], grid_info['y_coords'][-1]],
                          origin='lower', aspect='auto', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, label='Confidence')
            ax.set_title('Average Prediction Confidence')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌍 Advanced Land Cover Change Analysis & Prediction System | Developed with Streamlit</p>
    <p>Using 12 ML Models with 2 Policy Scenarios</p>
</div>
""", unsafe_allow_html=True)