# Battery-Aging-Model-using-Deep-Learning

![image description](images/Figure1.svg)

> **Abstract:**
> *In the context of the global energy transition, the need for sustainable and reliable energy storage solutions is more urgent than ever. Lithium-ion batteries and fuel cells, which are central to this paradigm shift, are increasingly being integrated into a wide range of applications, from industrial manufacturing to electric vehicles and renewable energy systems. However, a critical challenge that hinders their wider adoption and optimal use is the ageing and degradation phenomena inherent in these technologies. Understanding and accurately predicting the ageing behaviour of lithium-ion batteries and fuel cells is crucial to improving their longevity, safety and overall performance. Therefore, we present selected deep learning models to address these challenges and provide a novel approach to predict and model the ageing and degradation processes of these energy systems.*
<br />

### Project structure
This repsoitory consists of the following parts: 
- **data** folder: Here are all datasets and scripts to collect the datasets, preprocess them, performe feature engineering and create the final dataset used for the forecasting task.
- **utils** folder: Here helper classes for data handling, model generation, and model handling are stored
- **images** folder: Here are all figures and plots stored and the script to create them
- **models** folder: Here the model weights are stored
- **src** folder: Here the main scripts are stored for the forecasting baseline, local learning, federated learning and evaluation


### Install and Run the project 
To run the project you can fork this repository and following the instructions: 
First create your own virtual environment: 
```
python -m venv .venv
```
Activate your environment with
```
 .\.venv\Scripts\activate.ps1
```
Install all dependencies with
```
pip install -r requirements.txt
```
