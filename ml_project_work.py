import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/Pirate_Radio_Broadcasting new.csv')

# Preprocessing
df = df.fillna(0)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str) # Convert to string type before encoding
    df[col] = le.fit_transform(df[col])

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/Pirate_Radio_Broadcasting new.csv')

# Preprocessing
df = df.fillna(0)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str) # Convert to string type before encoding
    df[col] = le.fit_transform(df[col])

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('/content/Pirate_Radio_Broadcasting new.csv')

# Preprocessing
df = df.fillna(0)

le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str) # Convert to string type before encoding
    df[col] = le.fit_transform(df[col])

# Split data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (important for KNN)
specified_columns = X.columns # Store original column names
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=specified_columns)
X_test = pd.DataFrame(X_test, columns=specified_columns)

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt

# Accuracy values (replace with your output)
models = ['Logistic', 'Random Forest', 'KNN']
accuracies = [0.85, 0.92, 0.88]

plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Re-train RandomForestClassifier for feature importance
# (X_train, y_train, and X are assumed to be available from previous cells)
rf_model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
features = X.columns

# Sort feature importances for better visualization
sorted_idx = importances.argsort()
plt.barh(features[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")

plt.show()