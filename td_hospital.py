import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('TD_HOSPITAL_TRAIN.csv')
df.drop(['dose','pdeath'],axis=1)
for index in range(len(df['sex'])):
    val = df['sex'][index]
    if val == "male" or val == "Male" or val == "M" or val == 1:
        df['sex'][index] = 0
    else:
        df['sex'][index] = 1

# print(df['sex'])

def attempt_convert_to_float(column):
    try:
        return column.astype(float)
    except ValueError:  # If conversion fails, return the original column
        return column

# Apply the function to each column
df = df.apply(attempt_convert_to_float)

for col in df.columns:
    if df.dtypes[col] == "float64":
        df[col].fillna(df[col].mean(), inplace=True)

# df = pd.get_dummies(df, columns=['sex', 'race', 'primary'], drop_first=True)
df = pd.get_dummies(df, columns=['dnr','race', 'primary', 'disability', 'income' ,'extraprimary', 'cancer' ], drop_first=True)

df.to_csv('output.csv')


correlation = df.corr()['death'].sort_values(ascending=False)
# print(correlation)
# threshold = float(input("Enter the threshold value you would like to analyze:"))
# print(threshold)

threshold = 0.05

# Identify features that have a correlation magnitude above the threshold
significant_features = correlation[correlation.abs() > threshold].index.tolist()

# Exclude the target variable 'death' from the features list
significant_features.remove('death')

new_df = df[significant_features + ['death']]
print(new_df)

new_df.to_csv('sig.csv')




df = pd.read_csv('sig.csv')

# Split data into features and target variable
X = df.drop('death', axis=1)
y = df['death']

# Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X_train, y_train, 
                    epochs=20, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Sample prediction
sample_data = X_test.iloc[0]  # get the first row from the test set
predicted_prob = model.predict(np.array([sample_data]))[0][0]
print(f"The probability of death is: {predicted_prob * 100:.2f}%")
predicted_label = (predicted_prob > 0.5).astype(int)
print(f"Predicted Label: {predicted_label}")