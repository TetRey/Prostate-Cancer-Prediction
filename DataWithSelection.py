import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt 
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay)
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Mengimport data 
data = pd.read_excel("Prostate_Cancer.xlsx")
df = pd.DataFrame(data)
print(df.describe())
# Data preview
print("Data Preview : \n", df)
# Mengecek missing value
print("Mengecek missing value : \n", df.isnull().sum())
# Menghilangkan kolom id dan mengganti diagnosis result menjadi boolean 
df = df.drop(['id'], axis=1)
df['diagnosis_result'].replace({'M':1,'B':0},inplace=True)
# Mendeteksi outliers
print("Deteksi outlier : \n")
outliers=[]
def detect_outlier(data):
    threshold=3
    mean = np.mean(data)
    std = np.std(data)
    
    for x in data:
        z_score = (x-mean)/std
        if np.abs(z_score)>threshold:
            outliers.append(x)
    return outliers
variabel = ['diagnosis_result','radius','area','smoothness','compactness','symmetry', 'fractal_dimension', 'texture', 'perimeter']
for var in variabel:
  outlier_datapoints = detect_outlier(df[var])
  print("Outlier ", var, " = ", outlier_datapoints)

df.isna().sum()

# Mengganti Outlier value dengan mean
for i in variabel:
  df[i] = df[i].fillna(df[i].mean)

print("Data telah di preprocessing : \n",df)

# Grouping data
x = df.iloc[:,1:9]
y = df.iloc[:, 0]
print("Data x : \n", x)
print("Data y : \n", y)

# Normalisasi data dengan Z-Score
Zscore = StandardScaler()
x = Zscore.fit_transform(x)

# Seleksi Fitur Dengan PCA
x = PCA(n_components=5).fit_transform(x)

# Data variable setelah Normalisasi dan PCA
print(x)

# Spliting data menjadi data training dan data testing
x_1, x_2, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

# Mengklasifikasikan data variable menjadi 3 bagian
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', random_state=0)
x_train = kbins.fit_transform(x_1)
x_test = kbins.transform(x_2)


print("Data training variable :", x_train)
print("Data training class :", *y_train)
print("Data testing variable :", x_test)
print("Data testing class :", *y_test)

# Naive Bayes
print("Naive Bayes")
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_predictionNB = gaussian.predict(x_test)
accuracy_nb = round(accuracy_score(y_test, y_predictionNB)* 100, 2)
acc_gaussianNB = round(gaussian.score(x_train, y_train)* 100, 2)
print("Prediksi Naive Bayes : ", y_predictionNB)

# Confusion Matrix Naive Bayes
CMNB = confusion_matrix(y_test, y_predictionNB)
accuracyNB = accuracy_score(y_test, y_predictionNB)
precisionNB = precision_score(y_test, y_predictionNB)
recallNB = recall_score(y_test, y_predictionNB)
f1NB = f1_score(y_test, y_predictionNB)

TNNB = CMNB[1][1] * 1.0
FNNB = CMNB[1][0] * 1.0
TPNB = CMNB[0][0] * 1.0
FPNB = CMNB[0][1] * 1.0
total = TNNB + TPNB + FPNB + FNNB
sensitivityNB = TNNB / (TNNB + FPNB)* 100
specificityNB = TPNB / (TPNB + FNNB)* 100

print("Akurasi Naive Bayes: ", accuracyNB * 100, "%")
print("Recall Naive Bayes: ", recallNB*100, "%")
print("Precision Naive Bayes: ", + precisionNB)

# Menampilkan Confusion Matrix Naive Bayes
cm_displayNB=ConfusionMatrixDisplay(confusion_matrix=CMNB)
print('Confusion matrix for Naive Bayes\n',CMNB)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_predictionNB), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Decision Tree
print("Decision Tree")
Decision = DecisionTreeClassifier(random_state=0)
Decision.fit(x_train, y_train)
y_predictionDT = Decision.predict(x_test)
accuracy_DT = round(accuracy_score(y_test, y_predictionDT)* 100, 2)
acc_DecisionDT = round(Decision.score(x_train, y_train)* 100, 2)
print("Prediksi Decision Tree : ", y_predictionDT)

# Confusion Matrix Decision Tree
CMDT = confusion_matrix(y_test, y_predictionDT)
accuracyDT = accuracy_score(y_test, y_predictionDT)
precisionDT = precision_score(y_test, y_predictionDT)
recallDT = recall_score(y_test, y_predictionDT)
f1DT = f1_score(y_test, y_predictionDT)

TNDT = CMDT[1][1] * 1.0
FNDT = CMDT[1][0] * 1.0
TPDT = CMDT[0][0] * 1.0
FPDT = CMDT[0][1] * 1.0
total = TNDT + TPDT + FPDT + FNDT
sensitivityDT = TNDT / (TNDT + FPDT)* 100
specificityDT = TPDT / (TPDT + FNDT)* 100

print("Akurasi Decision Tree: ", accuracyDT * 100, "%")
print("Recall Decision Tree: ", recallDT * 100, "%")
print("Precision Decision Tree: ", + precisionDT)

# Menampilkan Confusion Matrix Decision Tree
cm_displayDT=ConfusionMatrixDisplay(confusion_matrix=CMDT)
print('Confusion matrix for Decision Tree\n',CMDT)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_predictionDT), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Random Forest
print("Random Forest")
Forest = RandomForestClassifier(random_state=0)
Forest.fit(x_train, y_train)
y_predictionRF = Forest.predict(x_test)
accuracy_RF = round(accuracy_score(y_test, y_predictionRF)* 100, 2)
acc_DecisionRF = round(Forest.score(x_train, y_train)* 100, 2)
print("Prediksi Random Forest : ", y_predictionRF)

# Confusion Matrix Random Forest
CMRF = confusion_matrix(y_test, y_predictionRF)
accuracyRF = accuracy_score(y_test, y_predictionRF)
precisionRF = precision_score(y_test, y_predictionRF)
recallRF = recall_score(y_test, y_predictionRF)
f1RF = f1_score(y_test, y_predictionRF)

TNRF = CMRF[1][1] * 1.0
FNRF = CMRF[1][0] * 1.0
TPRF = CMRF[0][0] * 1.0
FPRF = CMRF[0][1] * 1.0
total = TNRF + TPRF + FPRF + FNRF
sensitivityRF = TNRF / (TNRF + FPRF)* 100
specificityRF = TPRF / (TPRF + FNRF)* 100

print("Akurasi Random Forest: ", accuracyRF * 100, "%")
print("Recall Random Forest: ", recallRF * 100, "%")
print("Precision Random Forest: ", + precisionRF)

# Menampilkan Confusion Matrix Random Forest
cm_displayRF=ConfusionMatrixDisplay(confusion_matrix=CMRF)
print('Confusion matrix for Random Forest\n',CMRF)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_predictionRF), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# SVM 
print("SVM")
SVM = SVC(random_state=0)
SVM.fit(x_train, y_train)
y_predictionSVM = SVM.predict(x_test)
accuracy_SVM = round(accuracy_score(y_test, y_predictionSVM)* 100, 2)
acc_DecisionSVM = round(SVM.score(x_train, y_train)* 100, 2)
print("Prediksi SVM : ", y_predictionSVM)

# Confusion Matrix Decision Tree
CMSVM = confusion_matrix(y_test, y_predictionSVM)
accuracySVM = accuracy_score(y_test, y_predictionSVM)
precisionSVM = precision_score(y_test, y_predictionSVM)
recallSVM = recall_score(y_test, y_predictionSVM)
f1SVM = f1_score(y_test, y_predictionSVM)

TNSVM = CMDT[1][1] * 1.0
FNSVM = CMDT[1][0] * 1.0
TPSVM = CMDT[0][0] * 1.0
FPSVM = CMDT[0][1] * 1.0
total = TNSVM + TPSVM + FPSVM + FNSVM
sensitivityDT = TNSVM / (TNSVM + FPSVM)* 100
specificityDT = TPSVM / (TPSVM + FNSVM)* 100

print("Akurasi SVM: ", accuracySVM * 100, "%")
print("Recall SVM: ", recallSVM*100, "%")
print("Precision SVM: ", + precisionSVM)

# Menampilkan Confusion Matrix SVM
cm_displaySVM=ConfusionMatrixDisplay(confusion_matrix=CMSVM)
print('Confusion matrix for SVM\n',CMSVM)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(y_test, y_predictionSVM), annot=True, fmt=".0f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Perbandingan Akurasi Dengan Transformasi dan Seleksi Fitur: \n Naive Bayes : ", accuracyNB* 100, "%", "\n Decision Tree : ", accuracyDT* 100, "%", "\n Random Forest : ", accuracyRF* 100, "%", "\n SVM : ", accuracySVM* 100, "%" )