
import graphviz
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import math
from sklearn.calibration import cross_val_predict






from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.estimators import BicScore, ExhaustiveSearch
from pgmpy.models import BayesianModel
from pyvis.network import Network



import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.estimators import K2Score
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination




from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from matplotlib import pyplot as plt
from sklearn.metrics import completeness_score, homogeneity_score, silhouette_score, v_measure_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tqdm




dataset = pd.read_excel("DataSet.xlsx")

### --- ORGANIZZAZIONE DEL DATASET --- ###

print("queste sono le colonne nel nostro dataset")
print(dataset.columns)


### --- ANALISI DEI DATI E GRAFICO DI DISTRIBUZIONE CANCRI MALIGNI E BENIGNI --- ###

print (dataset.describe())

num_rows, num_colomns = dataset.shape
print(f"numero righe: {num_rows} numero colonne: {num_colomns}")

print("distinzione della natura delle colonne: ")

types = dataset.dtypes
print(types)

print("questi sono i valori nulli di ogni colonna: ")
valori_nulli = dataset.isnull().sum()
print(valori_nulli)

dataset['AFP'] = dataset['AFP'].replace('>', '', regex=True)
dataset['AFP'] = dataset['AFP'].astype(float)

dataset['CA125'] = dataset['CA125'].replace('>', '', regex=True)
dataset['CA125'] = dataset['CA125'].astype(float)

dataset['CA19-9'] = dataset['CA19-9'].replace('>', '', regex=True)
dataset['CA19-9'] = dataset['CA19-9'].astype(float)

y = dataset['tipo_tumore']
def gestisci_stringa(valore):
    if isinstance(valore, float):
        return valore  
    elif '>' in str(valore):
        return float(str(valore).replace('>', ''))
    else:
        return float(valore)

dataset['AFP'] = dataset['AFP'].apply(gestisci_stringa)
dataset['CA125'] = dataset['CA125'].apply(gestisci_stringa)
dataset['CA19-9'] = dataset['CA19-9'].apply(gestisci_stringa)



#abbiamo trovato dei valori nulli, utiliziamo la tecnica del KNNImputer
knn_imputer = KNNImputer(n_neighbors=5) 
dataset_imputed = pd.DataFrame(knn_imputer.fit_transform(dataset), columns=dataset.columns)
valori_nulli_imputati = dataset_imputed.isnull().sum()
print(valori_nulli_imputati)

#grafico a torta tumori benigni e maligni
conteggio_valori = dataset['tipo_tumore'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(conteggio_valori, labels = conteggio_valori.index, autopct = '%1.1f', startangle = 90, colors = ['red', 'green'])
plt.title("grafico delle distinzioni cancro maligno e benigno")
plt.show()


### --- OSSERVAZIONI GRAFICHE --- ###

correlazione = dataset.corr()

heatmap = sns.heatmap(correlazione, annot = True, cmap = 'coolwarm', fmt = '.2f', annot_kws={'size' : 6.5})
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=8)

plt.show()


plt.figure(figsize=(10, 8))
boxplot = sns.boxplot(x='CA72-4', hue='tipo_tumore', data=dataset)
plt.xticks(ticks=[0, 5, 10, 20, 30, 40, 50, 60, 100, 150, 160])
plt.title('valori del rapporto degli eosinofili')
plt.show()


scaler_standardizzazione = StandardScaler()
ds_stand = scaler_standardizzazione.fit_transform(dataset)

scaler_normalizzazione = MinMaxScaler()
ds_norm = scaler_normalizzazione.fit_transform(dataset)

#APPRENDIMENTO NON SUPERVISIONATO

#CLUSTERING


imputer = SimpleImputer(strategy='mean') 

#metodo del gomito dataset originale
ds_clustering_origin = dataset_imputed
ssd_origin = []
k_values_origin = range(2, 11)
warnings.simplefilter(action='ignore', category=FutureWarning)

for num_clusters_origin in k_values_origin:
    kmeans_origin = KMeans(n_clusters=num_clusters_origin, max_iter=100, init='k-means++', random_state=1)
    kmeans_origin.fit(ds_clustering_origin)
    ssd_origin.append(kmeans_origin.inertia_)

    media_silhouette_origin = silhouette_score(ds_clustering_origin, kmeans_origin.labels_)
    print('Con n_clusters={0}, il valore di silhouette {1}'.format(num_clusters_origin, media_silhouette_origin))

print('\n\n')
plt.title('Curva a gomito - Dataset Originale:')
plt.plot(k_values_origin, ssd_origin, marker='o')
plt.xlabel('Numero di Clusters')
plt.ylabel('Somma delle distanze al quadrato')
plt.grid(True)
plt.show()

optimal_clusters_origin = 3  

kmeans_origin = KMeans(n_clusters=optimal_clusters_origin, random_state=42, n_init=10)
kmeans_origin.fit(dataset_imputed)
labels_cluster_origin = kmeans_origin.labels_


#metodo del gomito dataset normalizzato

ds_norm_imputed = imputer.fit_transform(ds_norm)
ds_clustering_norm = ds_norm_imputed
ssd_norm = []
k_values_norm = range(2, 11)
warnings.simplefilter(action='ignore', category=FutureWarning)

for num_clusters_norm in k_values_norm:
    kmeans_norm = KMeans(n_clusters=num_clusters_norm, max_iter=100, init='k-means++', random_state=1)
    kmeans_norm.fit(ds_clustering_norm)
    ssd_norm.append(kmeans_norm.inertia_)

    media_silhouette_norm = silhouette_score(ds_clustering_norm, kmeans_norm.labels_)
    print('Con n_clusters={0}, il valore di silhouette {1}'.format(num_clusters_norm, media_silhouette_norm))

print('\n\n')
plt.title('Curva a gomito - Dataset Normalizzato:')
plt.plot(k_values_norm, ssd_norm, marker='o')
plt.xlabel('Numero di Clusters')
plt.ylabel('Somma delle distanze al quadrato')
plt.grid(True)
plt.show()

optimal_clusters_norm = 3 

kmeans_norm = KMeans(n_clusters=optimal_clusters_norm, random_state=42, n_init=10)
kmeans_norm.fit(ds_norm_imputed)
labels_cluster_norm = kmeans_norm.labels_

ds_stand_imputed = imputer.fit_transform(ds_stand)
ds_clustering_stand = ds_stand_imputed
ssd_stand = []
k_values_stand = range(2, 11)
warnings.simplefilter(action='ignore', category=FutureWarning)

for num_clusters_stand in k_values_stand:
    kmeans_stand = KMeans(n_clusters=num_clusters_stand, max_iter=100, init='k-means++', random_state=1)
    kmeans_stand.fit(ds_clustering_stand)
    ssd_stand.append(kmeans_stand.inertia_)

    media_silhouette_stand = silhouette_score(ds_clustering_stand, kmeans_stand.labels_)
    print('Con n_clusters={0}, il valore di silhouette {1}'.format(num_clusters_stand, media_silhouette_stand))

print('\n\n')
plt.title('Curva a gomito - Dataset Standardizzato:')
plt.plot(k_values_stand, ssd_stand, marker='o')
plt.xlabel('Numero di Clusters')
plt.ylabel('Somma delle distanze al quadrato')
plt.grid(True)
plt.show()

optimal_clusters_stand = 3 

kmeans_stand = KMeans(n_clusters=optimal_clusters_stand, random_state=42, n_init=10)
kmeans_stand.fit(ds_stand_imputed)
labels_cluster_stand = kmeans_stand.labels_



warnings.simplefilter(action='ignore', category=FutureWarning)

k_values = range(2, 6)  # da 2 a 5 cluster

for dataset_name, dataset_used in zip(['standardizzato', 'normalizzato', 'dataset'], [ds_stand_imputed, ds_norm_imputed, dataset_imputed]):
    print(f'\n\nValutazione {dataset_name}:\n')

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(dataset_used)
        labels_cluster = kmeans.labels_

        print(f'Con n_clusters={k}:')
        print('Omogeneità  : ', homogeneity_score(y, labels_cluster))
        print('Completezza : ', completeness_score(y, labels_cluster))
        print('V_measure   : ', v_measure_score(y, labels_cluster))
      
        
optimal_clusters_origin = 3  

kmeans_origin = KMeans(n_clusters=optimal_clusters_origin, random_state=42, n_init=10)
kmeans_origin.fit(dataset_imputed)
labels_cluster_origin = kmeans_origin.labels_

optimal_clusters_norm = 3 

kmeans_norm = KMeans(n_clusters=optimal_clusters_norm, random_state=42, n_init=10)
kmeans_norm.fit(ds_norm_imputed)
labels_cluster_norm = kmeans_norm.labels_

optimal_clusters_stand = 3 # Inserisci il numero ottimale di cluster identificato

kmeans_stand = KMeans(n_clusters=optimal_clusters_stand, random_state=42, n_init=10)
kmeans_stand.fit(ds_stand_imputed)
labels_cluster_stand = kmeans_stand.labels_


#APPRENDIMENTO SUPERVISIONATO

X = dataset.drop(['tipo_tumore'], axis=1)
y = dataset['tipo_tumore']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

missing_train = X_train.isnull().sum()
print("Dati mancanti nel set di addestramento:")
print(missing_train)

missing_test = X_test.isnull().sum()
print("\nDati mancanti nel set di test:")
print(missing_test)

imputer = SimpleImputer(strategy='mean') 

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = MinMaxScaler()

X_train_imputed_normalized = scaler.fit_transform(X_train_imputed)
X_test_imputed_normalized = scaler.transform(X_test_imputed)

k_values = range(1, 48)  

cross_val_scores = []

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, X_train_imputed_normalized, y_train, cv=5, scoring='accuracy')
    cross_val_scores.append(scores.mean())

plt.plot(k_values, cross_val_scores, marker='o')
plt.title('Cross-Validation')
plt.xlabel('Numero di vicini (k)')
plt.ylabel('Accuratezza Media')
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=10)
knn_classifier.fit(X_train_imputed_normalized, y_train)
y_pred = knn_classifier.predict(X_test_imputed_normalized)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print('\n\n')
print(f'Accuratezza: {accuracy:.4f}\n')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.title('Matrice di Confusione')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()
report = classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno'])

print("Classification Report per K-Nearest Neighbors (k=10):\n")
print(report)

y_train_pred = knn_classifier.predict(X_train_imputed_normalized)
accuracy_train = accuracy_score(y_train, y_train_pred)

accuracy_test = accuracy_score(y_test, y_pred)

labels = ['Training Set', 'Test Set']
accuracies = [accuracy_train, accuracy_test]

plt.bar(labels, accuracies, color=['red', 'green'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set')
plt.ylim(0, 1) 
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=10)

train_sizes, train_scores, test_scores = learning_curve(knn_classifier, X_train_imputed_normalized, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
print("Dimensione del set di addestramento: ", train_sizes)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, marker='o', label='Training Score', color='red')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='red')

plt.plot(train_sizes, test_mean, marker='o', label='Test Score', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.title('Curva di Overfitting per K-Nearest Neighbors (k=10)')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()


print("Dimensioni del set di addestramento:", X_train.shape)

print("Dimensioni del set di test:", X_test.shape)

### - SVM - ###
svm_classifier = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42)
svm_classifier.fit(X_train_imputed_normalized, y_train)

y_pred = svm_classifier.predict(X_test_imputed_normalized)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.title('Matrice di Confusione SVM')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuratezza: {accuracy:.4f}\n')
print('\nReport di Classificazione:')
print(class_report)





# -- SVM -- #

svm_classifier = SVC(kernel='rbf', C=100, gamma=0.1, random_state=42)

train_sizes_svm, train_scores_svm, test_scores_svm = learning_curve(svm_classifier, X_train_imputed_normalized, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

train_mean_svm = np.mean(train_scores_svm, axis=1)
train_std_svm = np.std(train_scores_svm, axis=1)
test_mean_svm = np.mean(test_scores_svm, axis=1)
test_std_svm = np.std(test_scores_svm, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_svm, train_mean_svm, marker='o', label='Training Score', color='red')
plt.fill_between(train_sizes_svm, train_mean_svm - train_std_svm, train_mean_svm + train_std_svm, alpha=0.15, color='red')

plt.plot(train_sizes_svm, test_mean_svm, marker='o', label='Test Score', color='green')
plt.fill_between(train_sizes_svm, test_mean_svm - test_std_svm, test_mean_svm + test_std_svm, alpha=0.15, color='green')

plt.title('Curva di Overfitting per SVM')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

plt.bar(labels, accuracy, color=['red', 'green'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set per SVM')
plt.ylim(0, 1) 
plt.show()

svm_classifier = SVC()

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
              'kernel': ['rbf']} 

normalizer = MinMaxScaler()
X_train_normalized = normalizer.fit_transform(X_train_imputed_normalized)

grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_normalized, y_train)

print("Migliori parametri:", grid_search.best_params_)

X_test_normalized = normalizer.transform(X_test_imputed_normalized)
y_pred = grid_search.best_estimator_.predict(X_test_normalized)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza del modello ottimale: {accuracy:.4f}')

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
plt.title('Matrice di Confusione SVM - Modello Ottimale (Normalizzato)')
plt.xlabel('Classe Predetta')
plt.ylabel('Classe Reale')
plt.show()

class_report = classification_report(y_test, y_pred)
print('\nReport di Classificazione:')
print(class_report)

### ---logistc regression--- ###

logreg_classifier = LogisticRegression(random_state=42)

logreg_classifier.fit(X_train_imputed_normalized, y_train)

y_train_pred_logreg = logreg_classifier.predict(X_train_imputed_normalized)

y_test_pred_logreg = logreg_classifier.predict(X_test_imputed_normalized)

accuracy_train_logreg = accuracy_score(y_train, y_train_pred_logreg)

accuracy_test_logreg = accuracy_score(y_test, y_test_pred_logreg)

labels_logreg = ['Training Set', 'Test Set']
accuracies_logreg = [accuracy_train_logreg, accuracy_test_logreg]

plt.bar(labels_logreg, accuracies_logreg, color=['red', 'green'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set per Logistic Regression')
plt.ylim(0, 1)  
plt.show()

y_test_pred_logreg = logreg_classifier.predict(X_test_imputed_normalized)

class_report_logreg = classification_report(y_test, y_test_pred_logreg, target_names=['Benigno', 'Maligno'])
logreg_model = LogisticRegression(random_state=42)

logreg_model.fit(X_train_imputed_normalized, y_train)
y_pred_logreg = logreg_model.predict(X_test_imputed_normalized)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)


print(f'Accuratezza del modello di logistic regression: {accuracy_logreg:.4f}')

print("Classification Report per Logistic Regression:\n")
print(class_report_logreg)

train_sizes_logreg, train_scores_logreg, test_scores_logreg = learning_curve(
    logreg_classifier, X_train_imputed_normalized, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))


train_mean_logreg = np.mean(train_scores_logreg, axis=1)
train_std_logreg = np.std(train_scores_logreg, axis=1)
test_mean_logreg = np.mean(test_scores_logreg, axis=1)
test_std_logreg = np.std(test_scores_logreg, axis=1)


plt.figure(figsize=(10, 6))
plt.plot(train_sizes_logreg, train_mean_logreg, marker='o', label='Training Score', color='red')
plt.fill_between(train_sizes_logreg, train_mean_logreg - train_std_logreg, train_mean_logreg + train_std_logreg, alpha=0.15, color='red')

plt.plot(train_sizes_logreg, test_mean_logreg, marker='o', label='Test Score', color='green')
plt.fill_between(train_sizes_logreg, test_mean_logreg - test_std_logreg, test_mean_logreg + test_std_logreg, alpha=0.15, color='green')

plt.title('Curva di Overfitting per Logistic Regression')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

### -- GRADIENT BOOSTER CLASSIFIER -- ###

gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=3, random_state=42)
gb_classifier.fit(X_train_imputed_normalized, y_train)

y_pred = gb_classifier.predict(X_test_imputed_normalized)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuratezza: {accuracy:.4f}\n')
print('\nReport di Classificazione:')
print(class_report)

gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=3)

train_sizes_gb, train_scores_gb, test_scores_gb = learning_curve(
    gb_classifier, X_train_imputed_normalized, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))


train_mean_gb = np.mean(train_scores_gb, axis=1)
train_std_gb = np.std(train_scores_gb, axis=1)
test_mean_gb = np.mean(test_scores_gb, axis=1)
test_std_gb = np.std(test_scores_gb, axis=1)


plt.figure(figsize=(10, 6))
plt.plot(train_sizes_gb, train_mean_gb, marker='o', label='Training Score', color='red')
plt.fill_between(train_sizes_gb, train_mean_gb - train_std_gb, train_mean_gb + train_std_gb, alpha=0.15, color='red')

plt.plot(train_sizes_gb, test_mean_gb, marker='o', label='Test Score', color='green')
plt.fill_between(train_sizes_gb, test_mean_gb - test_std_gb, test_mean_gb + test_std_gb, alpha=0.15, color='green')

plt.title('Curva di Overfitting per Gradient Boosting Classifier')
plt.xlabel('Dimensione del Set di Addestramento')
plt.ylabel('Accuratezza')
plt.legend(loc='best')
plt.grid(True)
plt.show()

#overfitting a barre

gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

gb_classifier.fit(X_train_imputed_normalized, y_train)

y_train_pred_gb = gb_classifier.predict(X_train_imputed_normalized)

y_test_pred_gb = gb_classifier.predict(X_test_imputed_normalized)

accuracy_train_gb = accuracy_score(y_train, y_train_pred_gb)

accuracy_test_gb = accuracy_score(y_test, y_test_pred_gb)


labels_gb = ['Training Set', 'Test Set']
accuracies_gb = [accuracy_train_gb, accuracy_test_gb]

plt.bar(labels_gb, accuracies_gb, color=['red', 'green'])
plt.ylabel('Accuratezza')
plt.title('Accuracy su Training Set e Test Set per Gradient Boosting Classifier')
plt.ylim(0, 1)  
plt.show()

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_imputed_normalized, y_train)

print("Migliori parametri:", grid_search.best_params_)



best_k = 10  
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)


stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


y_pred_cv = cross_val_predict(knn_classifier, X_train_imputed_normalized, y_train, cv=stratified_kfold)


classification_rep = classification_report(y_train, y_pred_cv, target_names=['Benigno', 'Maligno'])
print("Classification Report durante la Stratified K-Fold Cross-Validation:")
print(classification_rep)


best_svm_classifier = grid_search.best_estimator_


stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


y_pred_cv_svm = cross_val_predict(best_svm_classifier, X_train_normalized, y_train, cv=stratified_kfold)

classification_rep_svm = classification_report(y_train, y_pred_cv_svm, target_names=['Benigno', 'Maligno'])
print("Classification Report durante la Stratified K-Fold Cross-Validation per SVM:")
print(classification_rep_svm)


logreg_classifier = LogisticRegression(random_state=42)

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


y_pred_cv_logreg = cross_val_predict(logreg_classifier, X_train_imputed_normalized, y_train, cv=stratified_kfold)


classification_rep_logreg = classification_report(y_train, y_pred_cv_logreg, target_names=['Benigno', 'Maligno'])
print("Classification Report during Stratified K-Fold Cross-Validation for Logistic Regression:")
print(classification_rep_logreg)


gb_classifier = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01, max_depth=3, random_state=42)

stratified_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

y_pred_cv_gb = cross_val_predict(gb_classifier, X_train_imputed_normalized, y_train, cv=stratified_kfold)


classification_rep_gb = classification_report(y_train, y_pred_cv_gb, target_names=['Benigno', 'Maligno'])
print("Classification Report during Stratified K-Fold Cross-Validation for Gradient Boosting Classifier:")
print(classification_rep_gb)

#### RETE BAYESIANA ####

dataset_rete = pd.read_excel("DataSet.xlsx")


imputer = SimpleImputer(strategy='mean')  
dataset_rete_imputed = pd.DataFrame(imputer.fit_transform(dataset_rete), columns=dataset_rete.columns)

for col in dataset_rete_imputed.columns:
    dataset_rete_imputed[col] = dataset_rete_imputed[col].astype(int)


df_RBayes = dataset_rete_imputed.copy()


print(df_RBayes.isnull().sum())
print(df_RBayes.dtypes)

scaler = MinMaxScaler(feature_range=(0, 100))
df_RBayes_normalized = pd.DataFrame(scaler.fit_transform(df_RBayes), columns=df_RBayes.columns)

df_RBayes_normalized = df_RBayes_normalized[['tipo_tumore'] + [col for col in df_RBayes_normalized.columns if col != 'tipo_tumore']]


max_parents = 2
hc_k2_simplified = HillClimbSearch(df_RBayes_normalized)
modello_k2_simplified = hc_k2_simplified.estimate(scoring_method=K2Score(df_RBayes_normalized), max_indegree=max_parents, max_iter=1000)

rete_bayesiana = BayesianNetwork(modello_k2_simplified.edges())
rete_bayesiana.fit(df_RBayes_normalized)



print("Nodi della rete bayesiana:", end=" ")
print(*rete_bayesiana.nodes(), sep=", ")


print("\nArchi nella rete bayesiana:", end=" ")
print(*rete_bayesiana.edges(), sep=", ")




nodi = ["tipo_tumore", "HE4", "CA125", "Menopause", "ALB", "num_linfociti", "magnesio", "ALP", "rapp_eosinofili", "ALT", 
        "num_piastrine", "BUN", "calcio", "cloro", "K", "proteine_totali", "num_globuli_rossi", "età", "CA19-9", "CA72-4", 
        "PDW", "CEA", "creatinina", "DBIL", "TBIL", "acido_urine", "perc_linfociti", "MCH", "MCV", "HGB", "RDW"]

archi = [('tipo_tumore', 'HE4'), ('tipo_tumore', 'CA125'), ('tipo_tumore', 'Menopause'), ('tipo_tumore', 'ALB'), 
         ('tipo_tumore', 'num_linfociti'), ('tipo_tumore', 'magnesio'), ('HE4', 'acido_urine'), ('CA125', 'età'), 
         ('CA125', 'num_piastrine'), ('CA125', 'HE4'), ('CA125', 'CA19-9'), ('CA125', 'ALB'), ('CA125', 'num_linfociti'), 
         ('Menopause', 'età'), ('num_linfociti', 'perc_linfociti'), ('ALP', 'rapp_eosinofili'), ('ALP', 'CA125'),
         ('rapp_eosinofili', 'tipo_tumore'), ('rapp_eosinofili', 'magnesio'), ('ALT', 'num_piastrine'), 
         ('num_piastrine', 'TBIL'), ('num_piastrine', 'acido_urine'), ('num_piastrine', 'MCV'), 
         ('num_piastrine', 'perc_linfociti'), ('num_piastrine', 'HGB'), ('num_piastrine', 'PDW'), ('num_piastrine', 'MCH'), 
         ('num_piastrine', 'cloro'), ('num_piastrine', 'K'), ('num_piastrine', 'num_globuli_rossi'), ('BUN', 'Menopause'), 
         ('calcio', 'cloro'), ('calcio', 'K'), ('calcio', 'proteine_totali'), ('calcio', 'num_globuli_rossi'), 
         ('proteine_totali', 'ALP'), ('num_globuli_rossi', 'HGB'), ('CA72-4', 'PDW'), ('CA72-4', 'rapp_eosinofili'),
           ('CA72-4', 'tipo_tumore'), ('CEA', 'CA19-9'), ('creatinina', 'ALP'), ('creatinina', 'proteine_totali'),
             ('DBIL', 'TBIL'), ('MCH', 'MCV'), ('RDW', 'MCH')]

grafo = nx.DiGraph()
grafo.add_nodes_from(nodi)
grafo.add_edges_from(archi)


subgrafo = nx.bfs_tree(grafo, source='tipo_tumore')



pos = nx.spring_layout(subgrafo, seed=42)
y_spread_dict = {
'HE4' : 0.3,
'CA125' : 0.2
}

def organize_successors(node, level, y_offset):
    children = list(subgrafo.successors(node))
    if children:
        num_children = len(children)
        y = y_offset
        for child in children:
            y_spread = y_spread_dict.get(child, 0.2)
            pos[child] = (pos[node][0] + (y - num_children // 2) * y_spread, pos[node][1] - level)
            organize_successors(child, level + 1, y_offset - 1)
            y += 1
        

organize_successors('tipo_tumore', 1, -1)



plt.figure(figsize=(12, 8))
edge_labels = {(u, v): f"{u}->{v}" for u, v in subgrafo.edges()}
nx.draw(subgrafo, pos, with_labels=True, labels={n: n for n in subgrafo.nodes()}, font_size=5, node_size=800, node_color='lightblue', font_color='black', font_weight='bold', arrowsize=20)

plt.title("Rete Bayesiana - Albero con radice in 'tipo_tumore'")
plt.show()

bayes_estimator = BayesianEstimator

for column in dataset_rete_imputed.columns:
        rete_bayesiana.add_node(column)


rete_bayesiana.fit(dataset_rete_imputed, estimator=bayes_estimator, prior_type='BDeu', equivalent_sample_size=10)


inferenza = VariableElimination(rete_bayesiana)
nomi_colonne = dataset_rete_imputed.columns.tolist()



for variable in rete_bayesiana.nodes:
    cpd = rete_bayesiana.get_cpds(variable)
    min_value = cpd.values.min()
    max_value = cpd.values.max()
    print(f"Valori limite per la variabile '{variable}':")
    print(f"Minimo: {min_value}")
    print(f"Massimo: {max_value}")
    print("\n")




 
if 'tipo_tumore' not in rete_bayesiana.nodes:
    print("'tipo_tumore' non è presente nel sottografo.")
else:
    benigno = inferenza.query(variables=['tipo_tumore'], evidence={
    'AFP': 3,
    'età': 47,
    'ALB': 45,
    'ALP': 56,
    'ALT': 11,
    'AST': 24,
    'BUN': 5,
    'calcio': 2,
    'CA125': 15,
    'CA19-9': 36,
    'CA72-4': 6,
    'CEA': 1,
    'cloro': 107,
    'CO2CP': 19,
    'creatinina': 103,
    'DBIL': 2,
    'num_eosinofili': 0,
    'perc_eosinofili': 1,
    'globulina': 28,
    'ematocrito': 0,
    'HE4': 219,
    'HGB': 89,
    'K': 5,
    'num_linfociti': 0,
    'perc_linfociti': 16,
    'MCH': 33,
    'MCV': 103,
    'Menopause': 0,
    'magnesio': 0,
    'rapp_eosinofili': 76,
    'PDW': 13,
    'num_piastrine': 74,
    'num_globuli_rossi': 2,
    'RDW': 13,
    'TBIL': 5,
    'proteine_totali': 73,
    'acido_urine': 396,
})
    print('\nProbabilità per una donna di avere un tumore benigno: ')
    print(benigno, '\n')

maligno = inferenza.query(variables=['tipo_tumore'], evidence={
    'AFP': 1,
    'età': 42,
    'ALB': 40,
    'ALP': 69,
    'ALT': 17,
    'AST': 14,
    'BUN': 4,
    'calcio': 2,
    'CA125': 71,
    'CA19-9': 9,
    'CA72-4': 9,
    'CEA': 1,
    'cloro': 102,
    'CO2CP': 23,
    'creatinina': 71,
    'DBIL': 3,
    'num_eosinofili': 0,
    'perc_eosinofili': 0,
    'globulina': 29,
    'ematocrito': 0,
    'HE4': 49,
    'HGB': 131,
    'K': 5,
    'num_linfociti': 1,
    'perc_linfociti': 27,
    'MCH': 27,
    'MCV': 84,
    'Menopause': 0,
    'magnesio': 0,
    'rapp_eosinofili': 67,
    'PDW': 16,
    'num_piastrine': 197,
    'num_globuli_rossi': 4,
    'RDW': 15,
    'TBIL': 12,
    'proteine_totali': 69,
    'acido_urine': 200,
})

print('\nProbabilità per una donna di avere un tumore maligno: ')
print(maligno, '\n')

X = dataset_rete_imputed.drop(columns=['tipo_tumore'])
y = dataset_rete_imputed['tipo_tumore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



for column in X.columns:
    rete_bayesiana.add_node(column)

 

for node in rete_bayesiana.nodes:
    if node not in X.columns:
        print(f"Variabile '{node}' nel modello non trovata nelle colonne del set di dati.")



rete_bayesiana.fit(X_train, estimator=bayes_estimator, prior_type='BDeu', equivalent_sample_size=10)


y_pred = rete_bayesiana.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
