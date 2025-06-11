import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, ELU
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_excel('NaturalGasDemand_Input.xlsx', engine='openpyxl')

# Parse date and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

target_col = 'India total Consumption of Natural Gas (in BCM)'
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build deep neural network
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1]))
model.add(ELU())
for _ in range(47):
    model.add(Dense(128))
    model.add(ELU())
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# Early stopping
es = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=500, batch_size=16, callbacks=[es], verbose=2)

# Save model
model.save('natural_gas_demand_model.h5')

# Predict
y_pred = model.predict(X_test_scaled).flatten()

# Plot test vs prediction
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.legend()
plt.title('Test vs Prediction')
plt.xlabel('Month')
plt.ylabel(target_col)
plt.tight_layout()
plt.savefig('test_vs_prediction.png')
plt.close()

# Save results to Excel
results = pd.DataFrame({'Month': y_test.index, 'Actual': y_test.values, 'Predicted': y_pred})
results.to_excel('test_vs_prediction_results.xlsx', index=False)

print('Model training complete. Model and results saved.')
