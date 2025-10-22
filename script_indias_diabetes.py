#%%
# Description des variables :
# Pregancies : Nombre de grossesses passées par la patiente 
# Glucose : Taux de glucose dans le sang (mg/dL)
# BloodPressure : Tension artérielle diastolique (mm Hg)
# SkinThickness : Épaisseur du pli cutané (mm)
# Insulin : Niveau d'insuline sérique (mu U/mL)
# BMI : Indice de masse corporelle (poids en kg / (taille en m)^2)
# DiabetesPedigreeFunction : Fonction de pedigree du diabète (facteur génétique)
# Age : Âge de la patiente (années)

# Outcome : Résultat du test de dépistage du diabète (1 = diabétique, 0 = non diabétique)

# Variables démographique et familiale : age, pregnancies 

# Variables biologiques : glucose, bloodpressure, insulin -> indicateurs médicaux directs 

# Variables antropométriques : bmi, skinthickness -> indicateurs liés au poids et à la composition corporelle

# Variables génétiques : diabetespedigreefunction -> facteur de risque héréditaire

#%%

# Importation des bibliothèques nécessaires :

#######################################################
# Manipulation des données :
#######################################################

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
from pathlib import Path
from IPython.display import display # pour afficher les dataframes dans les notebooks

#######################################################
# Visualisation des données :
#######################################################

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pio.templates.default = "plotly_white"


#######################################################
# Analyse statistique 
#######################################################

from scipy import stats # tests d'hypothèses, corrélation, distributions de proba
from scipy.stats import normaltest, shapiro, anderson # tests de normalité d'une distribution
from scipy.stats import pearsonr, spearmanr, kendalltau # tests de corrélation (pearson -> correlation linéaire - données quantitatives continues, normale
# spearman  -> corrélation de rangs - données non-normales ou ordinales ; kendalltau -> corrélation de rangs - petits enchantillons
import statsmodels.api as sm # modélisation statistique (régression, ANOVA, tests, séries temporelles
# -> Sert à aller plus loin que scikit-learn en analyse statistique)
from statsmodels.stats.outliers_influence import variance_inflation_factor # pour détecter la multicolinéarité entre variables explicatives
# VIF Variance Inflation Factor ~= 1 -> pas de corrélation ; 1-5 modérée acceptable ; >10 forte corrélation


#######################################################
# Machine Learning 
#######################################################

## Pre-processing 


from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold, cross_validate )
# train_test_split : diviser les données en ensembles d'entraînement et de test
# Kfol : validation croisée simple
# stratifiedKFold : validation croisée en conservant la proportion des classes
# cross_val_score : évaluer un modèle avec validation croisée
# cross_validate : évaluer plusieurs métriques avec validation croisée 
# gridSearchCV : recherche exhaustive d'hyperparamètres -> teste toutes les combinaisons possibles
# randomizedSearchCV : recherche aléatoire d'hyperparamètres -> teste un nombre fixe

from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, LabelEncoder, OneHotEncoder)
# StandardScaler : centrage-réduction (moyenne=0, écart-type=1) -> Reg linéaire, SVM, KNN, PCA
# MinMaxScaler : mise à l'échelle entre 0 et 1 -> réseaux de neurones
# RobustScaler : mise à l'échelle robuste aux outliers -> données avec outliers
# QuantileTransformer : transformation non-linéaire pour rendre les données plus normales ou uniforme -> modèles linéaires
# LabelEncoder : encoder les labels catégoriels en entiers -> variables cibles
# OneHotEncoder : encoder les variables catégorielles en variables binaires -> variables explicatives
from sklearn.impute import SimpleImputer, KNNImputer
# SimpleImputer : imputation des valeurs manquantes avec moyenne, médiane, mode ou constante
# KNNImputer : imputation des valeurs manquantes avec les k plus proches voisins
from sklearn.feature_selection import (SelectKBest, f_regression, mutual_info_regression, RFE, RFECV, SelectFromModel)
# SelectKBest : sélection des k meilleures caractéristiques selon un test statistique, score
# f_regression : test F pour la régression linéaire - score pour selectKBest
# mutual_info_regression : information mutuelle pour la régression - score pour selectKBest
# RFE : élimination récursive des caractéristiques - sélection des caractéristiques en fonction de l'importance
# RFECV : RFE avec validation croisée - sélection des caractéristiques en fonction de l'importance avec validation croisée
# SelectFromModel : sélection des caractéristiques en fonction de l'importance d'un modèle (utiliser coef_ ou feature_importances_)


## Models :

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor,
                              AdaBoostRegressor,  BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


## Metrics

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    max_error, median_absolute_error
)


## Advanced ML Libraries 

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool


## Utility Functions 
import time # time() pour mesurer le temps d'exécution, sleep(s) pour pauses de s sec, perf_counter() pour chronométrage haute précision
import gc #libération automatique de la mémoire non utilisée par Python -> libérer de la ram
from tqdm import tqdm # barre de progression pour les boucles
tqdm.pandas()


## Set Random Seeds for Reproducibility 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



#%% 
# ============================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# ============================================================================

def print_divider(title="", width=80, style="="):
    """
    Create a formatted section divider for better readability
    """
    if title:
        side_width = (width - len(title) - 2) // 2
        print(f"\n{style * side_width} {title} {style * side_width}")
    else:
        print(f"\n{style * width}")

def describe_dataframe(df, name="DataFrame"):
    """
    Comprehensive DataFrame description with memory usage and data types
    """
    print_divider(f"{name} Overview") # Nom du data frame
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns") # Nombre de ligne et de colonnes
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB") # mémoire utilisée (en MO)
    print(f"\nData Types:") 
    print(df.dtypes.value_counts())# types des variables (int, float etc)
    print(f"\nMissing Values: {df.isnull().sum().sum()}") # nombre de valeurs manquantes
    print(f"Duplicate Rows: {df.duplicated().sum()}") # nombre de lignes dupliquées
    
    return df.info() # montre les colonnes, types et valeurs non nulles

def calculate_vif(df, features): 
    """
    Calculate Variance Inflation Factor for multicollinearity detection
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    vif_data["VIF"] = [variance_inflation_factor(df[features].values, i) 
                       for i in range(len(features))] # calcul le VIF pour chaque variable (features)
    return vif_data.sort_values('VIF', ascending=False)

def plot_distribution_analysis(df, column, target_column=None):
    """
    Crée une analyse visuelle complète de la distribution d’une colonne
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{column} Distribution', 
                       f'{column} Box Plot',
                       f'{column} Q-Q Plot',
                       f'{column} vs Target' if target_column else 'Violin Plot'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}],
               [{'type': 'scatter'}, {'type': 'violin' if not target_column else 'scatter'}]]
    ) # target_column : variable cible pour relation cible vs colonne
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], name=column, showlegend=False),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=df[column], name=column, showlegend=False),
        row=1, col=2
    )
    
    # Q-Q plot
    theoretical_quantiles = stats.probplot(df[column], dist="norm")[0][0]
    sample_quantiles = stats.probplot(df[column], dist="norm")[0][1]
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                  mode='markers', name='Q-Q', showlegend=False),
        row=2, col=1
    )
    
    # Target relationship or Violin plot
    if target_column:
        fig.add_trace(
            go.Scatter(x=df[column], y=df[target_column], 
                      mode='markers', name='vs Target', showlegend=False),
            row=2, col=2
        )
    else:
        fig.add_trace(
            go.Violin(y=df[column], name=column, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False,
                     title_text=f"Distribution Analysis: {column}")
    return fig


#%% 

df = pd.read_csv('diabetes.csv')
describe_dataframe(df, name="Diabetes Dataset")

display(df.describe(include='all').round(2))

#%% 
print_divider("DATA QUALITY REPORT", style="=")

# Create comprehensive quality report
quality_report = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': df.count().values,
    'Null Count': df.isnull().sum().values,
    'Null %': (df.isnull().sum().values / len(df) * 100).round(2),
    'Unique Values': df.nunique().values,
    'Unique %': (df.nunique().values / len(df) * 100).round(2)
})

# Add additional metrics for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    quality_report.loc[quality_report['Column'] == col, 'Min'] = df[col].min()
    quality_report.loc[quality_report['Column'] == col, 'Max'] = df[col].max()
    quality_report.loc[quality_report['Column'] == col, 'Mean'] = round(df[col].mean(), 2)
    quality_report.loc[quality_report['Column'] == col, 'Std'] = round(df[col].std(), 2)

display(quality_report)

# Check for data anomalies
print_divider("DATA ANOMALY CHECK")

anomalies = []

# Check for negative values in numeric columns
for col in numeric_cols:
    neg_count = (df[col] < 0).sum()
    if neg_count > 0:
        anomalies.append(f"⚠️ {col}: {neg_count} negative values found")
    else:
        print(f"✅ {col}: No negative values")

# Check for duplicates
dup_count = df.duplicated().sum()
if dup_count > 0:
    anomalies.append(f"⚠️ {dup_count} duplicate rows found")
    print(f"⚠️ Found {dup_count} duplicate rows")
    print("Duplicate rows details:")
    display(df[df.duplicated(keep=False)].sort_values(df.columns.tolist()))
else:
    print("✅ No duplicate rows found")

# Check for outliers using IQR method
print_divider("OUTLIER DETECTION (IQR Method)")

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        print(f"📊 {col}:")
        print(f"   - Outliers: {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
        print(f"   - Lower Bound: {lower_bound:.2f}")
        print(f"   - Upper Bound: {upper_bound:.2f}")
        print(f"   - Outlier Range: [{outliers[col].min():.2f}, {outliers[col].max():.2f}]")

if anomalies:
    print_divider("ANOMALY SUMMARY")
    for anomaly in anomalies:
        print(anomaly)
else:
    print("\n✅ No major data anomalies detected!")


# %%

print(f"Mean : ${df['Glucose'].mean():.2f}, Median : ${df['Glucose'].median():.2f}, Std : ${df['Glucose'].std():.2f}")
print(f"Mode : ${df['Glucose'].mode()[0]:.2f}")
print(f"Minimum: ${df['Glucose'].min():,.2f}")
print(f"Maximum: ${df['Glucose'].max():,.2f}")
print(f"Range: ${df['Glucose'].max() - df['Glucose'].min():,.2f}")
print(f"Coefficient of Variation: {(df['Glucose'].std()/df['Glucose'].mean())*100:.2f}%")

# Percentiles
print("\n📈 Percentile Distribution:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"  {p}th percentile: ${df['Glucose'].quantile(p/100):,.2f}")

print("\n📐 Distribution Shape:")
print(f"Skewness: {df['Glucose'].skew():.3f}") # mesure la symétrie de la distribution par rapport à sa moyenne.
print(f"Kurtosis: {df['Glucose'].kurtosis():.3f}") # mesure la “pointedness” ou aplatissement de la distribution comparée à une loi normale.

# Interpretation
if df['Glucose'].skew() > 1:
    print("→ Highly right-skewed distribution (long tail to the right)")
elif df['Glucose'].skew() > 0.5:
    print("→ Moderately right-skewed distribution")
elif df['Glucose'].skew() > -0.5:
    print("→ Approximately symmetric distribution")
else:
    print("→ Left-skewed distribution")
# %%
display(plot_distribution_analysis(df, 'Glucose'))
display(plot_distribution_analysis(df, 'BMI'))

display(plot_distribution_analysis(df, 'DiabetesPedigreeFunction', target_column='Outcome'))

# %%
