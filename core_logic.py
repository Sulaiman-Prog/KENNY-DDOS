import pandas as pd
import numpy as np
from scipy.stats import entropy
import logging
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class DDoSDetector:
    def __init__(self, entropy_threshold=0.6):
        self.threshold = entropy_threshold

    def calculate_entropy(self, data):
        """Tier 1: Statistical Filter (Entropy-based) [cite: 65, 92]"""
        value_counts = pd.Series(data).value_counts(normalize=True)
        return entropy(value_counts)

    def load_parquet_data(self, training_file, testing_file):
        """Phase 1: Data Preparation [cite: 60, 78]"""
        logging.info("Reading Parquet files...")
        train_df = pd.read_parquet(training_file)
        test_df = pd.read_parquet(testing_file)
        
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        df.columns = df.columns.str.strip()
        
        if 'Label' in df.columns:
            # Maps 'BENIGN' to 0 and everything else (attacks) to 1 [cite: 29, 39]
            df['Target'] = df['Label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        
        logging.info(f"Success! Merged dataset has {df.shape[0]} rows.")
        return df

def perform_feature_selection(df, n_features=45):
    """Phase 2: Random Forest Feature Importance Analysis [cite: 67, 101]"""
    logging.info(f"Selecting top {n_features} features using Random Forest...")
    
    X = df.drop(columns=['Target', 'Label'], errors='ignore')
    X = X.select_dtypes(include=[np.number]) 
    y = df['Target']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    top_features = importances['Feature'].head(n_features).tolist()
    logging.info("Feature Selection Complete.")
    return top_features, importances

def train_hybrid_system(df, top_features):
    """Phase 3 & 4: Implementation and Comparative Evaluation [cite: 70, 74, 83]"""
    logging.info("Preparing data for Hybrid MLP training...")
    
    X = df[top_features]
    y = df['Target']
    
    # MLP requires data scaling for better performance 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 70/15/15 Split logic (30% held out for Val/Test) [cite: 63, 98]
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Tier 2: MLP Module (Deep Pattern Recognition) [cite: 68, 102]
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), 
                        max_iter=500, 
                        activation='relu', 
                        solver='adam', 
                        random_state=42,
                        verbose=True) 
    
    logging.info("Training Hybrid Tier 2 (MLP)...")
    mlp.fit(X_train, y_train)
    
    # Evaluation Metrics [cite: 46, 111, 123]
    predictions = mlp.predict(X_test)
    print("\n" + "="*40)
    print("ENHANCED HYBRID MODEL PERFORMANCE")
    print("="*40)
    print(f"Accuracy Score: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Visualizing Confusion Matrix for the project defense
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    

    return mlp, scaler

# --- Execution Main Block ---
if __name__ == "__main__":
    detector = DDoSDetector()
    
    # Step 1: Ingestion
    TRAIN_FILE = "Syn-training.parquet"
    TEST_FILE = "Syn-testing.parquet"
    
    try:
        data = detector.load_parquet_data(TRAIN_FILE, TEST_FILE)
        
        # Step 2: Feature Selection (RF Module)
        top_cols, importance_df = perform_feature_selection(data)
        
        # Visualize Feature Importance
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))

        # Step 3 & 4: Hybrid MLP Training and Evaluation
        trained_model, data_scaler = train_hybrid_system(data, top_cols)
            
    except Exception as e:
        logging.error(f"Execution Error: {e}")