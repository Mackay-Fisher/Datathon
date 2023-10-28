# Sample participant submission for testing
from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import random

app = Flask(__name__)


class Solution:
    def __init__(self):
        #Initialize any global variables here
        self.model = tf.keras.models.load_model('example.h5')

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        
        """
        This function should return your final prediction!
        """
        labels = ['age', 'blood', 'reflex', 'bloodchem1', 'bloodchem2', 'psych1', 'glucose']
        values = [float(x) for x in [age, blood, reflex, bloodchem1, bloodchem2, psych1, glucose]]
        df = dict()
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)
        prediction = self.model.predict(df.to_numpy())
        return float(prediction[0][0])


# BOILERPLATE

