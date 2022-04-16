# Hexagonal architecture for ML Projects


## Hexagonal architecture 

Hexagonal architecture is defined by separatinhg a project into three
main parts:
 - Adapters: An Adapter will initiate the interaction with the Application
  through a Port, using a specific technology.
 - Ports: a technology-agnostic entry point, it determines the interface 
 which will allow foreign actors to communicate with the Application, 
 regardless of who or what will implement said interface
 - Core: This is main part of the Application, it is where all the logic of
 processing is determined.
 - Tests: Where all the automated tests for the Application are defined.

One of the best practices in adopting hexagonal architecture is to not
use complex external libraries on wirting parts of the core or ports,
only in adapters.

## Hexagonal architecture for ML projects

In order to adapt hexagonal architecture for MLProjects, it is 
necessary to define where the ML models should be located and how 
to integrate them into the three main parts.
In this project, it was chosen to place the models on a separate directory,
and treat them as adapters are treated. 
A port is written for every kind of MLmodel, defining which methods 
that model Class should have and every model that could serve 
the same function uses the same port. 


## Additional parts


Beyond Adapters, Ports, Core, Tests and Models, the Application will need one 
or more entrypoints, from where the processing should be initiated.
If ther is only a single entrypoint, it is commom to have an additional file 
on the same level as other directories. If there are more than one entrypoint,
another directory should be placed on the same level as the others
where all entrypoints should be located. 
