import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.colors as pc
import plotly.graph_objs as go
import plotly.express as px
from itertools import cycle
from natsort import natsorted
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean, cityblock, chebyshev, canberra, braycurtis, cosine, minkowski
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from google.colab import drive
from google.colab import files
from IPython.display import display

# Mount Google Drive
drive.mount('/content/drive')

# ==============================================================================
# 1. CONFIGURATION & PARAMETERS
# ==============================================================================

# Data Configuration
FORCE_RELOAD_TRAIN = False
mir_ranges = [(600, 1800), (2400, 4000)]
manual_label_map = {
    # Manually assign labels to drug groups if necessary
}

# Train and Test Data Paths
train_paths = [
    # Insert train data paths here
]

# Layer 1 Configuration: PCA + kNN
FORCE_RETRAIN_PCA = True
pca_manual_n_components = 10     # Set to None for auto-optimization
pca_manual_k_neighbors = 1       # Set to None for auto-optimization
pca_custom_metric = "euclidean"  # Set to None for auto-optimization

# Layer 2 Configuration: PLS + kNN
FORCE_RETRAIN_PLS = True
pls_manual_n_components = 10     # Set to None for auto-optimization
pls_manual_k_neighbors = 5       # Set to None for auto-optimization
pls_custom_metric = 'braycurtis' # Set to None for auto-optimization

# General Configuration
distance_metrics_list = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'braycurtis']
k_range_list = range(1, 11)
threshold_multiplier = 1.3


# ==============================================================================
# 2. FUNCTION DEFINITIONS: DATA PREPROCESSING
# ==============================================================================

def load_and_preprocess_data(folder_paths, mir_ranges=None, manual_label_map=None):
    all_data = {}
    raw_labels = []
    group_labels = []
    if mir_ranges is None: mir_ranges = [(650, 1750)]
    if manual_label_map is None: manual_label_map = {}

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        if not os.path.exists(folder_path): continue

        sorted_files = natsorted(os.listdir(folder_path))
        for file_name in sorted_files:
            if not file_name.lower().endswith(('.csv')): continue
            try:
                df = pd.read_csv(os.path.join(folder_path, file_name), header=None)
            except pd.errors.EmptyDataError: continue

            mask = np.zeros(len(df), dtype=bool)
            for start, end in mir_ranges:
                mask |= ((df[0] >= start) & (df[0] <= end))
            df = df[mask].copy()

            raw_labels.append(os.path.splitext(file_name)[0])
            group_labels.append(manual_label_map.get(folder_name, folder_name).lower())
            all_data[raw_labels[-1]] = df.set_index(0)[1]

    if not all_data: raise ValueError("No valid data found.")
    combined_df = pd.DataFrame(all_data)
    combined_df.index.name = "wavenumber"
    print(f"Data shape: {combined_df.shape}")
    return combined_df, raw_labels, group_labels

def preprocess_spectra(X, apply_snv=True, apply_deriv=True, window_length=11, polyorder=2, deriv_order=1):
    X_proc = X.copy()
    X_proc = np.nan_to_num(X_proc, nan=0.0)
    if apply_snv:
        means = np.mean(X_proc, axis=1, keepdims=True)
        stds = np.std(X_proc, axis=1, keepdims=True)
        stds[stds == 0] = 1e-10
        X_proc = (X_proc - means) / stds
    if apply_deriv:
        wl = min(window_length, X_proc.shape[1])
        if wl % 2 == 0: wl = max(3, wl - 1)
        X_proc = savgol_filter(X_proc, window_length=wl, polyorder=min(polyorder, wl - 1), deriv=deriv_order, axis=1)
    return np.nan_to_num(X_proc)


# ==============================================================================
# 3. FUNCTION DEFINITIONS: ML MODELS & DISTANCE METRICS
# ==============================================================================

class PCATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    def fit(self, X, y=None):
        self.pca.fit(X)
        return self
    def transform(self, X):
        return self.pca.transform(X)

class PLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)
    def fit(self, X, y):
        self.pls.fit(X, y)
        return self
    def transform(self, X):
        return self.pls.transform(X)

def detect_mode(manual_n_components, manual_k_neighbors, custom_metric):
    if manual_n_components is None and manual_k_neighbors is None and custom_metric is None: return 'optimal'
    elif manual_n_components is not None and manual_k_neighbors is None and custom_metric is None: return 'fixed_n'
    else: return 'manual'

def optimize_pca_knn(X_train, y_train, metrics=None, k_range=None):
    if metrics is None: metrics = ['euclidean', 'manhattan', 'chebyshev', 'cosine', 'braycurtis', 'canberra']
    if k_range is None: k_range = range(1, 11)
    pipe = Pipeline([('pca', PCATransformer()), ('knn', KNeighborsClassifier())])
    param_grid = {'pca__n_components': list(range(3, 7)), 'knn__n_neighbors': list(k_range), 'knn__metric': metrics}
    grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1, error_score='raise')
    try: grid.fit(X_train, y_train)
    except Exception as e:
        print(f"Optimization error: {e}")
        param_grid['pca__n_components'] = [3, 4]
        grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
    results = [{'n_components': p['pca__n_components'], 'k': p['knn__n_neighbors'], 'metric': p['knn__metric'], 'accuracy': grid.cv_results_['mean_test_score'][i]} for i, p in enumerate(grid.cv_results_['params'])]
    return grid.best_params_, results, grid

def optimize_pls_knn(X_train, y_train, metrics=None, k_range=None):
    if metrics is None: metrics = ['euclidean', 'manhattan', 'chebyshev']
    if k_range is None: k_range = range(1, 11)
    pipe = Pipeline([('pls', PLSTransformer()), ('knn', KNeighborsClassifier())])
    param_grid = {'pls__n_components': list(range(3, 7)), 'knn__n_neighbors': list(k_range), 'knn__metric': metrics}
    grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1, error_score='raise')
    try: grid.fit(X_train, y_train)
    except Exception as e:
        print(f"Optimization error: {e}")
        param_grid['pls__n_components'] = [3, 4]
        grid = GridSearchCV(pipe, param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
    results = [{'n_components': p['pls__n_components'], 'k': p['knn__n_neighbors'], 'metric': p['knn__metric'], 'accuracy': grid.cv_results_['mean_test_score'][i]} for i, p in enumerate(grid.cv_results_['params'])]
    return grid.best_params_, results, grid

def advanced_distance(u, v, metric='euclidean'):
    try:
        if metric == 'euclidean': return euclidean(u, v)
        elif metric == 'manhattan': return cityblock(u, v)
        elif metric == 'chebyshev': return chebyshev(u, v)
        elif metric == 'cosine': return cosine(u, v)
        elif metric == 'canberra': return canberra(u, v)
        elif metric == 'braycurtis': return braycurtis(u, v)
        elif metric == 'minkowski': return minkowski(u, v, p=3)
        else: return euclidean(u, v)
    except Exception as e:
        return np.nan

def calculate_class_distances(X_train_dr, y_train, X_test_dr, test_group_labels, encoder, metrics):
    class_names = encoder.classes_
    class_centroids = {}
    distances_dict = {cls: {metric: [] for metric in metrics} for cls in class_names}
    for cls in class_names:
        cls_idx = np.where(encoder.transform([cls])[0] == y_train)[0]
        class_centroids[cls] = np.mean(X_train_dr[cls_idx], axis=0) if len(cls_idx) > 0 else np.zeros(X_train_dr.shape[1])
    for i in range(len(X_test_dr)):
        true_class = test_group_labels[i]
        for metric in metrics:
            dist = advanced_distance(X_test_dr[i], class_centroids[true_class], metric=metric)
            if not np.isnan(dist) and not np.isinf(dist):
                distances_dict[true_class][metric].append(dist)
    return distances_dict

def compute_distance_thresholds(X_train_dr, y_train, encoder, metrics, multiplier=1.5, prediction=False):
    class_names = encoder.classes_
    centroids = {}
    thresholds = {cls: {} for cls in class_names}
    for cls in class_names:
        cls_idx = np.where(encoder.transform([cls])[0] == y_train)[0]
        X_cls = X_train_dr[cls_idx]
        centroid = np.mean(X_cls, axis=0)
        centroids[cls] = centroid
        for metric in metrics:
            dists = [advanced_distance(x, centroid, metric=metric) for x in X_cls]
            clean_dists = [d for d in dists if not (np.isnan(d) or np.isinf(d))]
            thresholds[cls][metric] = np.percentile(clean_dists, 100) * multiplier if clean_dists else float('inf')
    return centroids, thresholds, multiplier

def predict_with_unknown(knn_model, X_test_dr, encoder, centroids, thresholds, metric, prediction=False):
    y_pred_raw = knn_model.predict(X_test_dr)
    y_pred_labels = encoder.inverse_transform(y_pred_raw)
    final_labels, all_dists, all_thresholds = [], []
    for i, x in enumerate(X_test_dr):
        pred_label = y_pred_labels[i]
        centroid = centroids[pred_label]
        dist = advanced_distance(x, centroid, metric)
        threshold = thresholds[pred_label][metric]
        all_dists.append(dist)
        all_thresholds.append(threshold)
        if dist > threshold: final_labels.append('unknown')
        else: final_labels.append(pred_label)
    return final_labels, all_dists, all_thresholds

def calculate_classification_metrics(df):
    correct = (df['Actual Label'] == df['Predicted Label']).mean()
    unknown = (df['Predicted Label'] == 'unknown').mean()
    mistake = 1 - correct - unknown
    return pd.Series({'Correct': correct, 'Unknown': unknown, 'Misclassified': mistake, 'Count': len(df)})


# ==============================================================================
# 4. FUNCTION DEFINITIONS: VISUALIZATION & REPORTS
# ==============================================================================

def reduce_dimensionality(X_train, X_test=None, n_components=None, y_train=None, plot_params=None):
    method = plot_params.get('method', 'PLS') if plot_params else 'PLS'
    if n_components is None and plot_params: n_components = plot_params['manual_n_components']
    
    if method == 'PCA':
        pca = PCA(n_components=n_components)
        pca.fit(X_train)
        X_train_dr = pca.transform(X_train)
        if X_test is not None: return X_train_dr, pca.transform(X_test)
        return X_train_dr
    elif method == 'PLS':
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_train, y_train)
        X_train_dr = pls.transform(X_train)
        if X_test is not None: return X_train_dr, pls.transform(X_test)
        return X_train_dr

def plot_class_accuracy_per_metric(X_train, y_train, encoder, n_components, metrics, k_values, plot_params):
    method_name = plot_params.get('method', 'DR')
    X_train_dr = reduce_dimensionality(X_train, n_components=n_components, y_train=y_train, plot_params=plot_params)
    class_names = encoder.classes_
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    if n_metrics > 1: axes = axes.flatten()
    else: axes = [axes]

    for idx, metric in enumerate(metrics):
        if idx >= len(axes): break
        ax = axes[idx]
        ax.set_title(f'Distance: {metric.upper()}', fontsize=14)
        ax.set_xlabel('Number of neighbors (k)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xticks(k_values)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.05)

        class_accuracies = {cls: [] for cls in class_names}
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        line_styles = ['-', '--', '-.', ':']

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            for cls in class_names:
                cls_encoded = encoder.transform([cls])[0]
                cls_acc_scores = []
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for train_idx, test_idx in kf.split(X_train_dr, y_train):
                    X_train_fold, X_test_fold = X_train_dr[train_idx], X_train_dr[test_idx]
                    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
                    knn.fit(X_train_fold, y_train_fold)
                    cls_mask = (y_test_fold == cls_encoded)
                    if np.sum(cls_mask) > 0:
                        y_pred = knn.predict(X_test_fold[cls_mask])
                        acc = accuracy_score(y_test_fold[cls_mask], y_pred)
                        cls_acc_scores.append(acc)
                class_accuracies[cls].append(np.mean(cls_acc_scores) if cls_acc_scores else 0)

        for i, cls in enumerate(class_names):
            ax.plot(k_values, class_accuracies[cls], marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], color=colors[i], label=cls, markersize=6, linewidth=1.5)
        if idx == 0: ax.legend(loc='lower right', fontsize=9, ncol=2)

    for j in range(idx + 1, len(axes)): axes[j].axis('off')
    plt.suptitle(f'CLASS-WISE ACCURACY BY K ({method_name}={n_components})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_accuracy_at_optimal_n(optimization_results, best_n_components, k_range, plot_params):
    method_name = plot_params.get('method', 'DR')
    df = pd.DataFrame(optimization_results)
    df_optimal_n = df[df['n_components'] == best_n_components]
    if df_optimal_n.empty: return
    grouped = df_optimal_n.groupby(['metric', 'k'])['accuracy'].mean().reset_index()
    plt.figure(figsize=(12, 8))
    metrics = grouped['metric'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    line_styles = ['-', '--', '-.', ':']
    for i, metric in enumerate(metrics):
        metric_data = grouped[grouped['metric'] == metric]
        plt.plot(metric_data['k'], metric_data['accuracy'], marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], label=metric.upper(), markersize=8, linewidth=2)
    plt.xlabel('Number of neighbors (k)', fontsize=12)
    plt.ylabel('Average Accuracy', fontsize=12)
    plt.title(f'ACCURACY AT N_COMPONENTS ({best_n_components}) - {method_name}', fontsize=14)
    plt.xticks(list(k_range), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Distance')
    plt.tight_layout()
    plt.show()

def plot_distance_boxplot(distances_dict, class_names):
    metrics = list(distances_dict[class_names[0]].keys())
    n_metrics = len(metrics)
    ncols = min(2, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 6 * nrows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    for i, metric in enumerate(metrics):
        if i >= len(axes): break
        ax = axes[i] 
        data_to_plot, labels = [], []
        for cls in class_names:
            clean_data = [x for x in distances_dict[cls][metric] if not (np.isnan(x) or np.isinf(x))]
            if clean_data: data_to_plot.append(clean_data); labels.append(cls)
        if data_to_plot:
            ax.boxplot(data_to_plot, vert=False, patch_artist=True)
            ax.set_title(f'Distance: {metric.upper()}', fontsize=14)
            ax.set_yticklabels(labels, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
        else: ax.set_title(f'No data for {metric.upper()}')
    plt.suptitle('DISTANCE DISTRIBUTION BY CLASS AND METRIC', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_dr_3d(X_dr, y, raw_labels, encoder, plot_params, title_suffix="", components=[1, 2, 3], width=1400, height=1000, axis_ratio=(1.2, 1, 0.8)):
    method_name = plot_params.get('method', 'DR')
    title = f"{method_name} 3D - {title_suffix}"
    c1, c2, c3 = [c - 1 for c in components]
    class_names = encoder.classes_
    style_cycle = cycle([(color, symbol) for symbol in ['circle', 'cross', 'circle-open'] for color in pc.qualitative.D3])
    fig = go.Figure()
    for cls in class_names:
        idx = encoder.transform([cls])[0]
        indices = np.where(y == idx)[0]
        color_hex, marker_symbol = next(style_cycle)
        fig.add_trace(go.Scatter3d(
            x=X_dr[indices, c1], y=X_dr[indices, c2], z=X_dr[indices, c3],
            mode='markers', marker=dict(size=6, color=color_hex, symbol=marker_symbol),
            name=cls, showlegend=True, text=[raw_labels[i] for i in indices],
            hovertemplate="<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>"
        ))
    fig.update_layout(
        scene=dict(xaxis_title=f'{method_name} {components[0]}', yaxis_title=f'{method_name} {components[1]}', zaxis_title=f'{method_name} {components[2]}', aspectmode='manual', aspectratio=dict(x=axis_ratio[0], y=axis_ratio[1], z=axis_ratio[2])),
        width=width, height=height, title=title, margin=dict(l=100, r=20, b=0, t=40)
    )
    fig.show()

def plot_test_accuracy_at_optimal_n_with_threshold(X_train, y_train, X_test, y_test, test_group_labels, encoder, n_components, metrics, k_range, plot_params, threshold_multiplier=1.5):
    method_name = plot_params.get('method', 'DR')
    X_train_dr, X_test_dr = reduce_dimensionality(X_train, X_test, n_components, y_train, plot_params)
    plt.figure(figsize=(12, 8))
    markers, line_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'], ['-', '--', '-.', ':']
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for i, metric in enumerate(metrics):
        accuracies = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train_dr, y_train)
            centroids, thresholds, _ = compute_distance_thresholds(X_train_dr, y_train, encoder, [metric], multiplier=threshold_multiplier)
            y_pred_labels_with_unknown, _, _ = predict_with_unknown(knn, X_test_dr, encoder, centroids, thresholds, metric)
            correct = sum([1 for true, pred in zip(test_group_labels, y_pred_labels_with_unknown) if pred != 'unknown' and pred == true])
            accuracies.append(correct / len(y_test))
        plt.plot(k_range, accuracies, marker=markers[i % len(markers)], linestyle=line_styles[i % len(line_styles)], color=colors[i], label=metric.upper(), markersize=8, linewidth=2)
    plt.xlabel('Number of neighbors (k)')
    plt.ylabel('Test Accuracy (with Threshold)')
    plt.title(f'TEST ACCURACY (WITH THRESHOLD) AT N_COMPONENTS={n_components} - {method_name}')
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Distance')
    plt.tight_layout()
    plt.show()

def generate_report(results_df, method_name, best_k, metric):
    class_metrics = results_df.groupby('Actual Label', group_keys=False).apply(calculate_classification_metrics).reset_index()
    class_metrics.rename(columns={'Actual Label': 'Class'}, inplace=True)
    class_metrics['Display Class'] = class_metrics.apply(lambda row: f"{('Unknown' if str(row['Class']).strip().lower() == 'unknown' else row['Class'])} (n={row['Count']})", axis=1)

    fig, ax = plt.subplots(figsize=(12, max(6, len(class_metrics) * 0.4)))
    index = np.arange(len(class_metrics))
    bar_height = 0.6
    correct, mistake, unknown = class_metrics['Correct'].values, class_metrics['Misclassified'].values, class_metrics['Unknown'].values

    ax.barh(index, correct, height=bar_height, label='Correct', color='green', left=0)
    ax.barh(index, mistake, height=bar_height, label='Misclassified', color='purple', left=correct)
    ax.barh(index, unknown, height=bar_height, label='Unknown', color='red', left=correct + mistake)

    for i in range(len(index)):
        if correct[i] > 0.02: ax.text(correct[i]/2, i, f"{correct[i]*100:.1f}%", va='center', ha='center', color='white', fontsize=9)
        if mistake[i] > 0.02: ax.text(correct[i] + mistake[i]/2, i, f"{mistake[i]*100:.1f}%", va='center', ha='center', color='black', fontsize=9)
        if unknown[i] > 0.02: ax.text(correct[i] + mistake[i] + unknown[i]/2, i, f"{unknown[i]*100:.1f}%", va='center', ha='center', color='white', fontsize=9)

    ax.set(yticks=index, yticklabels=class_metrics['Display Class'])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Ratio')
    ax.set_title(f'[{method_name}] CLASSIFICATION RATIO BY CLASS (CORRECT / MISCLASSIFIED / UNKNOWN)\nK={best_k}, Metric={metric}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()

    report_df = class_metrics[['Class', 'Count']].copy()
    report_df['Correct (%)'] = class_metrics['Correct'] * 100
    report_df['Misclassified (%)'] = class_metrics['Misclassified'] * 100
    report_df['Unknown (%)'] = class_metrics['Unknown'] * 100
    report_df['Best K'], report_df['Metric'] = best_k, metric

    print(f"\n[COMPREHENSIVE REPORT {method_name}]:")
    display(report_df)

    detailed_path = f"/content/detailed_report_{method_name}.xlsx"
    summary_path = f"/content/accuracy_report_{method_name}.xlsx"
    results_df.to_excel(detailed_path, index=False)
    report_df.to_excel(summary_path, index=False)
    print(f"Reports successfully saved to system:\n- {detailed_path}\n- {summary_path}")
    files.download(detailed_path)
    files.download(summary_path)


# ==============================================================================
# 5. TRAINING (LAYER 1: PCA & 2: PLS)
# ==============================================================================

print("="*50)
print("LOADING TRAIN DATA")
print("="*50)
train_df, train_raw_labels, train_group_labels = load_and_preprocess_data(train_paths, mir_ranges, manual_label_map)
master_wavenumbers = train_df.index
X_train = train_df.T.to_numpy()
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_group_labels)
X_train_proc = preprocess_spectra(X_train, apply_snv=True, apply_deriv=True)
print(f"Loading complete. Train classes: {encoder.classes_}")

print("\n" + "="*50)
print("[LAYER 1] RUNNING TRAINING: PCA + kNN")
print("="*50)
mode_pca = detect_mode(pca_manual_n_components, pca_manual_k_neighbors, pca_custom_metric)

if mode_pca == 'optimal':
    print("\nOptimizing PCA and kNN simultaneously...")
    try:
        best_params_pca, pca_opt_results, grid_pca = optimize_pca_knn(X_train_proc, y_train, metrics=distance_metrics_list, k_range=k_range_list)
        pca_manual_n_components, pca_manual_k_neighbors, pca_custom_metric = best_params_pca['pca__n_components'], best_params_pca['knn__n_neighbors'], best_params_pca['knn__metric']
        print(f"==> OPTIMAL PCA PARAMETERS: n={pca_manual_n_components}, k={pca_manual_k_neighbors}, Metric={pca_custom_metric}")
    except Exception as e:
        print(f"Optimization error: {e}")
        pca_manual_n_components, pca_manual_k_neighbors, pca_custom_metric = 3, 5, 'euclidean'
        pca_opt_results = None
elif mode_pca == 'fixed_n':
    print("\nOptimizing k and metric with fixed n_components...")
    pipe = Pipeline([('pca', PCATransformer(n_components=pca_manual_n_components)), ('knn', KNeighborsClassifier())])
    grid_pca = GridSearchCV(pipe, {'knn__n_neighbors': list(k_range_list), 'knn__metric': distance_metrics_list}, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
    grid_pca.fit(X_train_proc, y_train)
    pca_manual_k_neighbors, pca_custom_metric = grid_pca.best_params_['knn__n_neighbors'], grid_pca.best_params_['knn__metric']
    pca_opt_results = [{'n_components': pca_manual_n_components, 'k': p['knn__n_neighbors'], 'metric': p['knn__metric'], 'accuracy': grid_pca.cv_results_['mean_test_score'][i]} for i, p in enumerate(grid_pca.cv_results_['params'])]
    print(f"==> OPTIMAL PCA PARAMETERS (n={pca_manual_n_components}): k={pca_manual_k_neighbors}, Metric={pca_custom_metric}")
else:
    print(f"\nUsing manual parameters: n={pca_manual_n_components}, k={pca_manual_k_neighbors}, Metric={pca_custom_metric}")
    pca_opt_results = None

pca_transformer = PCATransformer(n_components=pca_manual_n_components)
pca_transformer.fit(X_train_proc)
X_train_pca = pca_transformer.transform(X_train_proc)

pca_knn_model = KNeighborsClassifier(n_neighbors=pca_manual_k_neighbors, metric=pca_custom_metric)
pca_knn_model.fit(X_train_pca, y_train)

plot_params_pca = {
    'method': 'PCA', 'best_n_components': pca_manual_n_components,
    'optimization_results': pca_opt_results, 'manual_n_components': pca_manual_n_components,
    'custom_metric': pca_custom_metric, 'distance_metrics': distance_metrics_list,
    'k_range': k_range_list, 'k_values': list(k_range_list), 'mode': mode_pca
}

print("\n[PLOTTING TRAINING SET GRAPHS - PCA]")
plot_dr_3d(X_train_pca, y_train, train_raw_labels, encoder, plot_params_pca, title_suffix="Training Set (PCA)")
if pca_opt_results is not None:
    plot_accuracy_at_optimal_n(pca_opt_results, pca_manual_n_components, k_range_list, plot_params_pca)
plot_class_accuracy_per_metric(X_train_proc, y_train, encoder, pca_manual_n_components, [pca_custom_metric], k_range_list, plot_params_pca)
print("PCA Training Complete.")

print("\n" + "="*50)
print("[LAYER 2] RUNNING TRAINING: PLS + kNN")
print("="*50)
mode_pls = detect_mode(pls_manual_n_components, pls_manual_k_neighbors, pls_custom_metric)

if mode_pls == 'optimal':
    print("\nOptimizing PLS and kNN simultaneously...")
    try:
        best_params_pls, pls_opt_results, grid_pls = optimize_pls_knn(X_train_proc, y_train, metrics=distance_metrics_list, k_range=k_range_list)
        pls_manual_n_components, pls_manual_k_neighbors, pls_custom_metric = best_params_pls['pls__n_components'], best_params_pls['knn__n_neighbors'], best_params_pls['knn__metric']
        print(f"==> OPTIMAL PLS PARAMETERS: n={pls_manual_n_components}, k={pls_manual_k_neighbors}, Metric={pls_custom_metric}")
    except Exception as e:
        print(f"Optimization error: {e}")
        pls_manual_n_components, pls_manual_k_neighbors, pls_custom_metric = 3, 5, 'euclidean'
        pls_opt_results = None
elif mode_pls == 'fixed_n':
    print("\nOptimizing k and metric with fixed n_components...")
    pipe = Pipeline([('pls', PLSTransformer(n_components=pls_manual_n_components)), ('knn', KNeighborsClassifier())])
    grid_pls = GridSearchCV(pipe, {'knn__n_neighbors': list(k_range_list), 'knn__metric': distance_metrics_list}, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
    grid_pls.fit(X_train_proc, y_train)
    pls_manual_k_neighbors, pls_custom_metric = grid_pls.best_params_['knn__n_neighbors'], grid_pls.best_params_['knn__metric']
    pls_opt_results = [{'n_components': pls_manual_n_components, 'k': p['knn__n_neighbors'], 'metric': p['knn__metric'], 'accuracy': grid_pls.cv_results_['mean_test_score'][i]} for i, p in enumerate(grid_pls.cv_results_['params'])]
    print(f"==> OPTIMAL PLS PARAMETERS (n={pls_manual_n_components}): k={pls_manual_k_neighbors}, Metric={pls_custom_metric}")
else:
    print(f"\nUsing manual parameters: n={pls_manual_n_components}, k={pls_manual_k_neighbors}, Metric={pls_custom_metric}")
    pls_opt_results = None

pls_transformer = PLSTransformer(n_components=pls_manual_n_components)
pls_transformer.fit(X_train_proc, y_train)
X_train_pls = pls_transformer.transform(X_train_proc)

pls_knn_model = KNeighborsClassifier(n_neighbors=pls_manual_k_neighbors, metric=pls_custom_metric)
pls_knn_model.fit(X_train_pls, y_train)

plot_params_pls = {
    'method': 'PLS', 'best_n_components': pls_manual_n_components,
    'optimization_results': pls_opt_results, 'manual_n_components': pls_manual_n_components,
    'custom_metric': pls_custom_metric, 'distance_metrics': distance_metrics_list,
    'k_range': k_range_list, 'k_values': list(k_range_list), 'mode': mode_pls
}

print("\n[PLOTTING TRAINING SET GRAPHS - PLS]")
plot_dr_3d(X_train_pls, y_train, train_raw_labels, encoder, plot_params_pls, title_suffix="Training Set (PLS)")
if pls_opt_results is not None:
    plot_accuracy_at_optimal_n(pls_opt_results, pls_manual_n_components, k_range_list, plot_params_pls)
plot_class_accuracy_per_metric(X_train_proc, y_train, encoder, pls_manual_n_components, [pls_custom_metric], k_range_list, plot_params_pls)
print("PLS Training Complete.")

# ==============================================================================
# 6. BATCH TESTING EXECUTION 
# ==============================================================================

test_paths = [
    # Insert test data paths here
]
    
print("\n" + "="*50)
print("[LAYER 1] RUNNING TEST BATCH: PCA + kNN")
print("="*50)

print("Loading Test data...")
test_df, test_raw_labels, test_group_labels = load_and_preprocess_data(test_paths, mir_ranges, manual_label_map)

# Use master_wavenumbers from RAM
test_df = test_df.reindex(master_wavenumbers).fillna(0)
X_test = test_df.T.to_numpy()
try:
    y_test = encoder.transform(test_group_labels)
except ValueError:
    y_test = np.zeros(len(test_group_labels))
X_test_proc = preprocess_spectra(X_test, apply_snv=True, apply_deriv=True)
print("Test data loading complete.")

X_test_pca = pca_transformer.transform(X_test_proc)
centroids_pca, thresholds_pca, _ = compute_distance_thresholds(X_train_pca, y_train, encoder, [pca_custom_metric], multiplier=threshold_multiplier)
y_pred_pca_unk, dists_pca, thresh_pca = predict_with_unknown(pca_knn_model, X_test_pca, encoder, centroids_pca, thresholds_pca, pca_custom_metric)

results_pca = pd.DataFrame({
    'File': test_raw_labels, 
    'Actual Label': test_group_labels, 
    'Predicted Label': y_pred_pca_unk,
    'Distance': dists_pca, 
    f'Threshold {int(threshold_multiplier*100)}%': thresh_pca,
    'Result': ['Correct' if p == t else ('Incorrect' if p != 'unknown' else 'Unknown') for p, t in zip(y_pred_pca_unk, test_group_labels)]
})

print("\n[PLOTTING TEST SET GRAPHS]")
plot_dr_3d(X_test_pca, y_test, test_raw_labels, encoder, plot_params_pca, title_suffix="Test Set (PCA)")
distances_dict_pca = calculate_class_distances(X_train_pca, y_train, X_test_pca, test_group_labels, encoder, distance_metrics_list)
plot_distance_boxplot(distances_dict_pca, encoder.classes_)
plot_test_accuracy_at_optimal_n_with_threshold(
    X_train=X_train_proc, 
    y_train=y_train, 
    X_test=X_test_proc, 
    y_test=y_test, 
    test_group_labels=test_group_labels, 
    encoder=encoder, 
    n_components=pca_manual_n_components, 
    metrics=distance_metrics_list, 
    k_range=k_range_list, 
    plot_params=plot_params_pca, 
    threshold_multiplier=threshold_multiplier
)

print("\n[EXPORT REPORT]")
generate_report(results_pca, "PCA", pca_manual_k_neighbors, pca_custom_metric)


# ==============================================================================
# 7. PREDICTION (SINGLE FILE TEST)
# ==============================================================================

def test_single_file(file_path, mir_ranges, master_wavenumbers):
    try: 
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    mask = np.zeros(len(df), dtype=bool)
    for start, end in mir_ranges: 
        mask |= ((df[0] >= start) & (df[0] <= end))
    df_filtered = df[mask].copy().set_index(0)[1]
    df_synced = df_filtered.reindex(master_wavenumbers).fillna(0)
    X_raw = df_synced.to_numpy().reshape(1, -1)
    return preprocess_spectra(X_raw, apply_snv=True, apply_deriv=True)

print("\n--- STEP 1: UPLOAD NEW SAMPLE FILE ---")
uploaded = files.upload()

if uploaded:
    uploaded_file_name = list(uploaded.keys())[0]
    print(f"\nSuccessfully uploaded file: {uploaded_file_name}")
    
    print("\n--- STEP 2: INPUT ACTUAL LABEL ---")
    true_label_input = input("Please enter the actual label for comparison: ").strip().lower()

    print(f"\n--- STEP 3: PREDICTION RESULTS ---")
    X_new_proc = test_single_file(uploaded_file_name, mir_ranges, master_wavenumbers)

    if X_new_proc is not None:
        # Predict using PCA
        X_new_pca = pca_transformer.transform(X_new_proc)
        y_pca, d_pca, t_pca = predict_with_unknown(pca_knn_model, X_new_pca, encoder, centroids_pca, thresholds_pca, pca_custom_metric)
        status_pca = "CORRECT" if y_pca[0] == true_label_input else ("INCORRECT" if y_pca[0] != 'unknown' else "UNKNOWN")

        # Predict using PLS
        X_new_pls = pls_transformer.transform(X_new_proc)
        
        # Ensure PLS thresholds and centroids are computed
        centroids_pls, thresholds_pls, _ = compute_distance_thresholds(X_train_pls, y_train, encoder, [pls_custom_metric], multiplier=threshold_multiplier)
        
        y_pls, d_pls, t_pls = predict_with_unknown(pls_knn_model, X_new_pls, encoder, centroids_pls, thresholds_pls, pls_custom_metric)
        status_pls = "CORRECT" if y_pls[0] == true_label_input else ("INCORRECT" if y_pls[0] != 'unknown' else "UNKNOWN")

        quick_test_result = pd.DataFrame({
            'Algorithm': ['PCA Layer 1', 'PLS Layer 2'],
            'Actual Label': [true_label_input, true_label_input],
            'Prediction': [y_pca[0], y_pls[0]],
            'Distance': [f"{d_pca[0]:.4f}", f"{d_pls[0]:.4f}"],
            'Threshold': [f"{t_pca[0]:.4f}", f"{t_pls[0]:.4f}"],
            'Result': [status_pca, status_pls]
        })
        display(quick_test_result)
else:
    print("No file was uploaded.")
