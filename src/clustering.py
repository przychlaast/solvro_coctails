import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from collections import Counter
from preprocessing import load_and_preprocess_data, encode_categorical_features

def load_processed_data(file_path):
    """Wczytuje przetworzone dane z pliku JSON"""
    return pd.read_json(file_path)

def kmeans_clustering(features, n_clusters_range=range(2, 16)):
    """Wykonuje klasteryzację K-means dla różnej liczby klastrów"""
    results = {}
    
    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        if n_clusters > 1:
            silhouette = silhouette_score(features, cluster_labels)
            db_score = davies_bouldin_score(features, cluster_labels)
        else:
            silhouette = 0
            db_score = 0
        
        results[n_clusters] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette': silhouette,
            'davies_bouldin': db_score
        }
        
        print(f"K-means z {n_clusters} klastrami:")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Score: {db_score:.4f}")
        print()
    
    return results

def plot_clustering_metrics(kmeans_results):
    """Wizualizuje metryki oceny klasteryzacji"""
    plt.figure(figsize=(10, 8))
    
    n_clusters = list(kmeans_results.keys())
    silhouette_scores = [result['silhouette'] for result in kmeans_results.values()]
    db_scores = [result['davies_bouldin'] for result in kmeans_results.values()]
    
    plt.subplot(2, 1, 1)
    plt.plot(n_clusters, silhouette_scores, 'o-', color='blue')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(n_clusters, db_scores, 'o-', color='red')
    plt.xlabel('Liczba klastrów')
    plt.ylabel('Davies-Bouldin Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../figures/clustering_metrics.png')
    plt.close()

def visualize_clusters_2d(features, labels, method_name):
    """Wizualizuje klastry w przestrzeni 2D po redukcji wymiarów"""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], color=color, 
                   marker='o', s=30, label=f'Klaster {label}')
    
    plt.title(f'Wizualizacja klastrów dla {method_name}')
    plt.xlabel('Pierwsza główna składowa')
    plt.ylabel('Druga główna składowa')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../figures/clusters_2d_{method_name}.png')
    plt.close()

def analyze_clusters(df, labels, feature_cols, method_name):
    """Analizuje charakterystyki klastrów"""
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    numeric_features = ['num_ingredients', 'num_alcoholic_ingredients', 'percent_alcoholic']
    if 'name_length' in df.columns:
        numeric_features.append('name_length')
    if 'num_tags' in df.columns:
        numeric_features.append('num_tags')
    
    cluster_stats = df_with_clusters.groupby('cluster')[numeric_features].agg(['mean', 'std'])
    print(f"Statystyki klastrów dla {method_name}:")
    print(cluster_stats)
    
    category_cols = [col for col in feature_cols if col.startswith('category_')]
    glass_cols = [col for col in feature_cols if col.startswith('glass_')]
    
    cluster_categories = {}
    cluster_glasses = {}
    
    for cluster_id in set(labels):
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        
        categories = []
        for _, row in cluster_df.iterrows():
            cat_cols = [col for col in category_cols if row[col] == 1]
            categories.extend([col.replace('category_', '') for col in cat_cols])
        
        glasses = []
        for _, row in cluster_df.iterrows():
            glass_c = [col for col in glass_cols if row[col] == 1]
            glasses.extend([col.replace('glass_', '') for col in glass_c])
        
        cluster_categories[cluster_id] = Counter(categories).most_common(3)
        cluster_glasses[cluster_id] = Counter(glasses).most_common(3)
    
    print(f"Najpopularniejsze kategorie koktajli w klastrach dla {method_name}:")
    for cluster_id, top_cats in cluster_categories.items():
        print(f"Klaster {cluster_id}: {top_cats}")
    
    print(f"Najpopularniejsze szklanki w klastrach dla {method_name}:")
    for cluster_id, top_glasses in cluster_glasses.items():
        print(f"  Klaster {cluster_id}: {top_glasses}")
    
    return cluster_stats, cluster_categories, cluster_glasses

def plot_cluster_characteristics(df, labels, method_name):
    """Tworzy heatmapę charakterystyk klastrów"""
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    unique_clusters = set(labels)
    
    numeric_features = ['num_ingredients', 'num_alcoholic_ingredients', 'percent_alcoholic']
    if 'name_length' in df.columns:
        numeric_features.append('name_length')
    if 'num_tags' in df.columns:
        numeric_features.append('num_tags')
    
    cluster_means = []
    for cluster in unique_clusters:
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster]
        means = cluster_df[numeric_features].mean()
        cluster_means.append(means)
    
    heatmap_df = pd.DataFrame(cluster_means, index=[f'Klaster {c}' for c in unique_clusters])
    
    normalized_df = (heatmap_df - heatmap_df.mean()) / heatmap_df.std()
    
    plt.figure(figsize=(12, len(unique_clusters) * 0.8))
    sns.heatmap(normalized_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Znormalizowane charakterystyki klastrów dla {method_name}')
    plt.tight_layout()
    plt.savefig(f'../figures/cluster_heatmap_{method_name}.png')
    plt.close()

if __name__ == "__main__":
    try:
        df = load_processed_data('../data/processed_cocktails.json')
    except FileNotFoundError:
        try:
            df = load_and_preprocess_data('../data/cocktail_dataset.json')
            df, _, _ = encode_categorical_features(df)
            
            df.to_json('../data/processed_cocktails.json', orient='records')
        except Exception as e:
            print(f"Błąd podczas przetwarzania danych: {e}")
            exit(1)
    
    numeric_features = ['num_ingredients', 'num_alcoholic_ingredients', 'percent_alcoholic']
    if 'name_length' in df.columns:
        numeric_features.append('name_length')
    if 'num_tags' in df.columns:
        numeric_features.append('num_tags')
    
    category_cols = [col for col in df.columns if col.startswith('category_')]
    glass_cols = [col for col in df.columns if col.startswith('glass_')]
    
    feature_cols = numeric_features + category_cols + glass_cols
    features = df[feature_cols].copy()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features[numeric_features] = scaler.fit_transform(features[numeric_features])
    
    print(f"Przygotowano dane o wymiarach: {features.shape}")
    
    
    kmeans_results = kmeans_clustering(features)
    plot_clustering_metrics(kmeans_results)

    best_n_clusters = max(kmeans_results, key=lambda k: kmeans_results[k]['silhouette'])
    print(f"Optymalna liczba klastrów według Silhouette Score: {best_n_clusters}")
    
    best_kmeans = kmeans_results[best_n_clusters]
    visualize_clusters_2d(features, best_kmeans['labels'], f'kmeans_{best_n_clusters}_clusters')
    analyze_clusters(df, best_kmeans['labels'], feature_cols, f'kmeans_{best_n_clusters}_clusters')
    plot_cluster_characteristics(df, best_kmeans['labels'], f'kmeans_{best_n_clusters}_clusters')
    
    