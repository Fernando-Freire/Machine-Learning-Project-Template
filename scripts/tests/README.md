# How to Write tests for Machine Learning Models

There are few kinds of tests that can evaluate if 
an machine learning code is functioning as it shoud, 
due to the results being very dependant on the 
training data and the statistical nature of models. 
Some of the checks that can be made are:

1. Test integration to data sources: (Integration tests)
  The code that fetches the datasets used in the
  ml pipeline should be tested for any possible 
  discrepancies between what is saved elsewhere 
  and what was retrieved by the code.

2. Check if the pre-processing of data is correct: (Unit tests)
  the data pre-processing steps are usually just
  deterministic code being applied to the trained 
  and test datasets and should be tested just as 
  any other kind of code. Any pre-processing methods
  should be written as static methods, if possible, 
  in order to facilitate unit testing.

3. Check if ML models are consistent: (E2E tests)
  One of the main lessons in data science is to 
  never use your training dataset to test the model,
  for this is known as data leakage. However, 
  in the tradicional code testing world, this can 
  be used to test the machine learning code itself to 
  test if something went very wrong on the actual 
  model training step.


