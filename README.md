#  Diabetes Prediction Project (Pima Indians Dataset)

##  Author
**Nom :** Arthur TENA  
**Contact :** arthurtena3@gmail.com
**GitHub :** [Arthur-tena](https://github.com/Arthur-tena)

---

##  Description du projet

Ce projet a pour objectif d’analyser et de modéliser le **risque de diabète** à partir du célèbre dataset **Pima Indians Diabetes Database**.  
L’analyse suit une démarche complète en **statistiques exploratoires** puis en **machine learning**, avec les étapes suivantes :

1. **Prétraitement des données (preprocessing)** : nettoyage, traitement des valeurs manquantes, normalisation, etc.  
2. **Analyse statistique exploratoire (EDA)** : visualisations, corrélations, distribution des variables.  
3. **Modélisation Machine Learning** : entraînement et évaluation de modèles prédictifs (logistic regression, random forest, etc.).  
4. **Interprétation** : étude de l’importance des variables et validation du modèle.

---

## À propos du jeu de données

**Source :** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
**Origine :** National Institute of Diabetes and Digestive and Kidney Diseases.  
**Type de données :** Médicales (mesures physiologiques et démographiques).  
**Taille :** 768 observations × 9 variables.  
**Population étudiée :** Femmes âgées d’au moins 21 ans, d’origine Pima (Indiens d’Amérique).

### 🧬 Description des variables

| Variable | Description | Type |
|-----------|-------------|------|
| `Pregnancies` | Nombre de grossesses | Quantitative |
| `Glucose` | Concentration de glucose dans le sang (mg/dL) | Quantitative |
| `BloodPressure` | Pression artérielle diastolique (mmHg) | Quantitative |
| `SkinThickness` | Épaisseur du pli cutané tricipital (mm) | Quantitative |
| `Insulin` | Concentration d’insuline sérique (µU/mL) | Quantitative |
| `BMI` | Indice de masse corporelle (poids/taille²) | Quantitative |
| `DiabetesPedigreeFunction` | Indice génétique lié à l’hérédité du diabète | Quantitative |
| `Age` | Âge de la patiente | Quantitative |
| `Outcome` | Présence de diabète (1) ou non (0) | Cible binaire |

**Référence :**  
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).  
*Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*  
In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261–265). IEEE Computer Society Press.

---

##  Installation et environnement

### 1. Cloner le projet
```bash
git clone https://github.com/tonprofil/diabetes.git
cd diabetes
pip install -r requirements.txt
```

