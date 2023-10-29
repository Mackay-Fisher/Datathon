import pandas as pd
import numpy as np



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

threshold = 0.1

# Identify features that have a correlation magnitude above the threshold
significant_features = correlation[correlation.abs() > threshold].index.tolist()

# Exclude the target variable 'death' from the features list
significant_features.remove('death')

new_df = df[significant_features + ['death']]

new_df.to_csv('sig.csv')


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('sig.csv')

# Split data into features and target variable
X = df.drop('death', axis=1)
y = df['death']

# Standardize numerical features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regulizer = tf.keras.regularizers.l1(0.01)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=regulizer),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(opt, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X_train, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Sample prediction
sample_data = X_test.iloc[0]  # get the first row from the test set
predicted_prob = model.predict(np.array([sample_data]))[0][0]
# print(f"The probability of death is: {predicted_prob * 100:.2f}%")
# predicted_label = (predicted_prob > 0.5).astype(int)
# print(f"Predicted Label: {predicted_label}")

model.save('Mynewmodel.h5')

def preprocess_data(df):
    df = df.drop(['dose', 'pdeath'], axis=1)
    df['sex'] = df['sex'].map(lambda x: 0 if x in ["male", "Male", "M", 1] else 1)
    df = df.apply(attempt_convert_to_float)
    
    for col in df.columns:
        if df.dtypes[col] == "float64":
            df[col].fillna(df[col].mean(), inplace=True)

    df = pd.get_dummies(df, columns=['dnr','race', 'primary', 'disability', 'income' ,'extraprimary', 'cancer' ], drop_first=True)
    
    return df

class Solution:
    def __init__(self):
        self.model = tf.keras.models.load_model('Mynewmodel.h5')
        # Note: You might also want to load 'significant_features' and 'scaler' if they are saved externally

    def calculate_death_prob(self, data):
        df = preprocess_data(data)
        X_new = df[significant_features]
        scaler = StandardScaler()
        X_new_scaled = scaler.transform(X_new) # Assuming 'scaler' is a globally available object
        prediction = self.model.predict(X_new_scaled)
        return float(prediction[0][0])


# app = Flask(__name__)

# @app.route("/death_probability", methods=["POST"])
def q1():
    # data = request.json
    # df = pd.DataFrame(data)
    df = pd.read_csv('TD_HOSPITAL_TRAIN.csv')
    solution = Solution()
    prob = solution.calculate_death_prob(df)
    return jsonify({"probability": prob})

print(q1())

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5555)




Also this is the way that teh data s typically passed in to the fucntions
def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer)


Where each value is an array holding that column of the data