# Machine Learning Project Template
This project is a template for creating a Machine Learning model
based on the MLFlow MLProject and using hexagonal architecture

## Dependencies

In order to run the notebooks, it is recommended to use docker 
and docker-compose. 
In order to get training and test data for the experiment, 
it is necessary to clone the following repo:

`git clone --branch setup https://github.com/Fernando-Freire/MLFlow_docker_compose_template.git`

and execute the commands specified in that repository's README.

After that, the docker-compose file in the notebooks directory should
be able to connect to the minio instance and fetch the necessary data.

## Organization

This project is designed to be incorporated in a MLFlow, so it is 
separated into two parts:
 - The notebooks directory, which creates a docker container with 
 a bind-mount in order to design and test new models. These notebooks
 should show what is being done to pre-process the data and 
 how the machine learning frameworks are being used to determine 
 the result of each categorization.
 - The scripts directory, which contains the finalized versions of 
 modeling functions created in the notebooks. The script versions
 for training models is necessary in order to use this repository 
 as a path for MLFlow to train new models with updated data. 

## Model Training and Saving

The scripts directory is written as an Application using
Hexagonal architecture for training models and saving them 
on MLFlow model versioning repository. 


#### To-do
 - Unit tests on Gensim[X]
 - Type verification on input data[]
 - (Not this repo)e2e tests for loaded models from mlflow[]
 - metrics of training time, model size, memory usage, codification time, vocab size[]
 - apply analogies comparision for word embedding models[] 
