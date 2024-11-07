import unittest
import json
import sys as system
import io
import pandas as pd
import re
import numpy as np

# Execute code marked with #si-exercise
with open("Lesson.ipynb", "r") as file:
    f_str = file.read()

doc = json.loads(f_str)
code = [i for i in doc['cells'] if i['cell_type']=='code']
si = {}
for i in code:
    for j in i['source']:
        if "#si-exercise" in j:
            exec(compile("".join(i['source']), '<string>', 'exec'))

class TestCase(unittest.TestCase):

    def testLogitCorrectValues(self):
    
        data = pd.read_csv("tests/files/assignment8Data.csv")
        
    
        print(f"Data shape: {data.shape}")
        
    
        print("Data preview:")
        print(data[['sex', 'age', 'educ', 'white']].head())
        
    
        print("Data types:")
        print(data[['sex', 'age', 'educ', 'white']].dtypes)
        
    
        print("Missing values in each column:")
        print(data[['sex', 'age', 'educ', 'white']].isnull().sum())
        
        print(f"Unique values in 'white': {data['white'].unique()}")
        
        x = data.loc[:100, ['sex', 'age', 'educ']]
        y = data.loc[:100, 'white']
        reg = RegressionModel(x, y, create_intercept=True, regression_type='logit')
        reg.fit_model()
    
        print("Actual Regression Results:")
        for var, metrics in reg.results.items():
            print(f"{var}: {metrics}")
        
        sex_expected = {'coefficient': -1.1229156890097627,
                       'standard_error': 0.39798772782618025,
                       'z_stat': -2.821483202869492,
                       'p_value': 0.004780214077269219}
        age_expected = {'coefficient': -0.007012518056833769,
                       'standard_error': 0.010835821823286998,
                       'z_stat': -0.6471607019011091,
                       'p_value': 0.5175279421902776}
        educ_expected = {'coefficient': -0.046485475816343394,
                        'standard_error': 0.10100278092776117,
                        'z_stat': -0.46023956359766527,
                        'p_value': 0.6453442758780246}
        intercept_expected = {'coefficient': 5.735435005488546,
                             'standard_error': 1.1266207023561843,
                             'z_stat': 5.090830475148922,
                             'p_value': 3.56498650369634e-07}
    
        tol = 1e-2  
    
        def within_tolerance(expected, actual, tolerance):
            return abs(expected - actual) <= tolerance
    
        self.assertTrue(
            within_tolerance(sex_expected['coefficient'], reg.results['sex']['coefficient'], tol) and
            within_tolerance(sex_expected['standard_error'], reg.results['sex']['standard_error'], tol) and
            within_tolerance(sex_expected['z_stat'], reg.results['sex']['z_stat'], tol) and
            within_tolerance(sex_expected['p_value'], reg.results['sex']['p_value'], tol),
            "Sex coefficients are not correctly calculated."
        )
    
        self.assertTrue(
            within_tolerance(age_expected['coefficient'], reg.results['age']['coefficient'], tol) and
            within_tolerance(age_expected['standard_error'], reg.results['age']['standard_error'], tol) and
            within_tolerance(age_expected['z_stat'], reg.results['age']['z_stat'], tol) and
            within_tolerance(age_expected['p_value'], reg.results['age']['p_value'], tol),
            "Age coefficients are not correctly calculated."
        )
    
        self.assertTrue(
            within_tolerance(educ_expected['coefficient'], reg.results['educ']['coefficient'], tol) and
            within_tolerance(educ_expected['standard_error'], reg.results['educ']['standard_error'], tol) and
            within_tolerance(educ_expected['z_stat'], reg.results['educ']['z_stat'], tol) and
            within_tolerance(educ_expected['p_value'], reg.results['educ']['p_value'], tol),
            "Education coefficients are not correctly calculated."
        )
    
        self.assertTrue(
            within_tolerance(intercept_expected['coefficient'], reg.results['intercept']['coefficient'], tol) and
            within_tolerance(intercept_expected['standard_error'], reg.results['intercept']['standard_error'], tol) and
            within_tolerance(intercept_expected['z_stat'], reg.results['intercept']['z_stat'], tol) and
            within_tolerance(intercept_expected['p_value'], reg.results['intercept']['p_value'], tol),
            "Intercept coefficients are not correctly calculated."
        )
