import numpy as np
from joblib import dump, load

## Load data
X = np.load('./Data/X.npy')
Y = np.load('./Data/Y.npy')

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Flatten the images for PCA/UMAP
X_train_flattened = X_train.reshape((X_train.shape[0], -1))
X_test_flattened = X_test.reshape((X_test.shape[0], -1))


from sklearn.preprocessing import StandardScaler
# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)
# Save the fitted scaler
dump(scaler, 'fitted_scaler.joblib')

from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_scaled, Y_train)

# Save the model to disk
filename = 'rf_model.joblib'
dump(rf_model, filename)

