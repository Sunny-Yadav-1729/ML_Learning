
import streamlit as st
import warnings
from Bagging.Visualize_app_Bagging import Bagging
from Bagging.Random_Forest.Visualize_random_forest import Random_Forest
from Boosting.Adaboost.VIsualize_app_Adaboost import Adaboost
from Boosting.Gradient_Boost.visualize_app_gradient_boosting import Gradient_Boosting
from Voting.visualize_app_voting import Voting
warnings.filterwarnings("ignore")

st.sidebar.markdown("# Ensemble Learing")
choose_algo = st.sidebar.selectbox(
    "Choose Ensemble Techinque ",
    ['Bagging','Voting','Random_Forest','Adaboost','Gradient_Boost']
    ,index=0
)

if choose_algo == 'Bagging':
    Bagging()
elif choose_algo == 'Random_Forest':
    Random_Forest()
elif choose_algo == 'Adaboost':
    Adaboost()
elif choose_algo == 'Voting':
    Voting()
elif choose_algo == 'Gradient_Boost':
    Gradient_Boosting()
       