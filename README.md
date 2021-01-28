# ml-opt-app

# Watch the tutorial video

Machine Learning Hyperparameter Optimizer App (Streamlit + Scikit-learn + Python)

<a href="https://youtu.be/HT2WHLgYpxY"><img src="http://img.youtube.com/vi/HT2WHLgYpxY/0.jpg" alt="How to Build a ML Hyperparameter Optimization App (Streamlit + Scikit-learn + Python)" title="How to Build a ML Hyperparameter Optimization App (Streamlit + Scikit-learn + Python)" width="400" /></a>

# Demo

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/dataprofessor/ml-opt-app/main/ml-opt-app.py)

# Reproducing this web app
To recreate this web app on your own computer, do the following.

### Create conda environment
Firstly, we will create a conda environment called *mlopt*
```
conda create -n mlopt python=3.7.9
```
Secondly, we will login to the *mlopt* environement
```
conda activate mlopt
```
### Install prerequisite libraries

Download requirements.txt file

```
wget https://raw.githubusercontent.com/dataprofessor/ml-opt-app/main/requirements.txt

```

Pip install libraries
```
pip install -r requirements.txt
```

###  Launch the app

```
streamlit run app.py
```
