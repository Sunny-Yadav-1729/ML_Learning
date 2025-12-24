import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_classification,make_regression,load_iris
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")

def Adaboost():
    st.header("AdaBoost Ensemble")
    def get_x_y_regression(path):
        df = pd.read_csv(path)
        x = np.array(df.iloc[:,:1])
        y = df.iloc[:,-1]
        return x,y
    def get_x_y_classification(path):
        df = pd.read_csv(path)
        x = np.array(df.iloc[:,:2])
        y = df.iloc[:,-1]
        return x,y

    def give_dataset_classifaction():
        lists = []

        for i in range(5):
            x,y = make_blobs(100,centers=2,n_features =2,cluster_std=1.7,random_state=42*i)
            lists.append((f"Dateset_{i+1}",x,y))
        x,y = np.array(sns.load_dataset('iris').iloc[:,:4]),sns.load_dataset('iris').iloc[:,-1]
        lists.append(('iris',x,y))
        folder_path = r"C:\Users\WELCOME\OneDrive\Desktop\Learning\classification__datasets"
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                list_full_path = os.path.join(folder_path,file)
                x,y = get_x_y_classification(list_full_path)
                lists.append((file[:-4],x,y))
        return lists
    def give_dataset_regression():
        lists = []

        for i in range(5):
            x,y = make_regression(100,n_features=1,noise=12,random_state=42*i)
            lists.append((f"Dateset_{i+1}",x,y))
        folder_path = r"C:\Users\WELCOME\OneDrive\Desktop\Learning\\Regression_Dataset"
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                list_full_path = os.path.join(folder_path,file)
                x,y = get_x_y_regression(list_full_path)
                lists.append((file[:-4],x,y))
        return lists
        # return lists

    def draw_meshgrid(X):
        a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array

    def load_initial_graph(dataset,typr,classifier,regress,ax):

        if typr == 'Regression':
            name_list = [(i[0]) for i in regress]
            idx = name_list.index(dataset)
            n,x,y  = regress[idx]
            ax.scatter(x=x,y=y,cmap = 'rainbow')
            return x,y
        else  :
            name_list = [(i[0]) for i in classifier]
            idx = name_list.index(dataset)
            n,x,y = classifier[idx]
            if dataset =='iris':
                
                y = LabelEncoder().fit_transform(y)
                # st.write(y)
                ax.scatter(x=x.T[0],y=x.T[1],c=y,cmap = 'rainbow')
                x = x[:,:2]
                return x,y
        
            # idx = int(dataset[-1]) -1
            n,x,y  = classifier[idx]
            ax.scatter(x=x.T[0],y=x.T[1],c=y,cmap = 'rainbow')
            return x,y
            
    def plot_decision_region(ax, model, X, y, title,typeof):
        # Make a mesh grid
        
        # Only for 2D classification
        # if X.shape[1] < 2:
        #     raise ValueError("Decision region requires 2D features")
        if X.shape[1] >= 2 and typeOf == 'Classification':
        # 2D classification decision region
            a = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.05)
            b = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.05)
            XX, YY = np.meshgrid(a, b)
            input_array = np.c_[XX.ravel(), YY.ravel()]

            model.fit(X, y)
            Z = model.predict(input_array)
            Z = Z.reshape(XX.shape)

            ax.contourf(XX, YY, Z, alpha=0.3, cmap='rainbow')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolors='k', s=25)
            ax.set_title(title)
        
        elif typeOf == 'Regression' and X.shape[1] == 1:
            # 1D regression line plot
            model.fit(X, y)
            x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_line)
            ax.scatter(X, y, color='blue', label='Data')
            ax.plot(x_line, y_pred, color='red', label='Prediction')
            ax.set_title(title)

        elif X.shape[1] >= 2 and typeOf == 'Regression':
            # Optional: 2D regression meshgrid (rare)
            a = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.05)
            b = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.05)
            XX, YY = np.meshgrid(a, b)
            input_array = np.c_[XX.ravel(), YY.ravel()]

            model.fit(X, y)
            Z = model.predict(input_array)
            Z = Z.reshape(XX.shape)

            ax.contourf(XX, YY, Z, alpha=0.3, cmap='rainbow')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolors='k', s=25)
            ax.set_title(title)

        else:
            # If dataset has <2 features for classification, raise warning
            ax.scatter(X[:, 0], y, color='blue', label='Data')
            ax.set_title(title + " (Not enough features for meshgrid)")





    classification_dataset = give_dataset_classifaction()
    regression_dataset = give_dataset_regression()
    st.sidebar.markdown("## AdaBoost ")
    problem_tpye = ['Regression','Classification']
    typeOf = st.sidebar.selectbox(
        label='Choose Problem Type',options=problem_tpye
    )

    name_classification = [i[0] for i in classification_dataset]
    name_regression = [i[0] for i in regression_dataset]

    dataset = st.sidebar.selectbox(
        label='Choose dataset type',
        options=name_regression if typeOf == problem_tpye[0] else name_classification
    )

     

     
    algo = st.sidebar.selectbox(
        label=f'Choose {"loss" if typeOf == "Regression"  else "algorithm" }',
        options=['linear', 'square', 'exponential'] if typeOf == 'Regression'  else ['SAMME', 'SAMME.R']
    )
    

    
    n_estimtor = st.sidebar.number_input(
        label='Choose number estimator', 
        value=50,
        min_value=5 ,max_value=200
    ) 

    learing_rate = st.sidebar.number_input(
        label='Choose learning_rate', 
        step=0.1, 
        min_value=0.1,max_value=10.0
    ) 
     


    fig, ax = plt.subplots()

    # Plot initial graph
    X,y = load_initial_graph(dataset,typeOf,classification_dataset,regression_dataset,ax)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    orig = st.pyplot(fig)

    if st.sidebar.button('Run Algorithm'):
        # orig.empty()
        
        # clf = LogisticRegression(penalty=penalty,C=c_input,solver=solver,max_iter=max_iter,multi_class=multi_class,l1_ratio=l1_ratio)
        # clf.fit(X_train,y_train)
         
           
            m = None
            bc = br = None
            if typeOf == problem_tpye[1]:
                bc = AdaBoostClassifier(None,n_estimators=n_estimtor,algorithm=algo,learning_rate=learing_rate)
                sc = cross_val_score(bc,X,y,cv=5,scoring='accuracy')
                st.sidebar.write(f" Adaboost  : {np.round(sc.mean(),2)}")
                m = bc
            else :
                br = AdaBoostRegressor(None,n_estimators=n_estimtor,loss=algo,learning_rate=learing_rate)
                sc = cross_val_score(br,X,y,cv=5,scoring='r2')
                st.sidebar.write(f" Adaboost : {np.round(sc.mean(),2)}")
                m = br
            # labels = clf.predict(input_array)
            fig, axs = plt.subplots(1, 1 + 1, figsize=(5*(1+1), 4))

        
            plot_decision_region(axs[0], m, X, y, title=m.__class__.__name__,typeof=typeOf)

        
            if typeOf == 'Classification':
            
                plot_decision_region(axs[-1], bc, X, y, title=f"Adaboost ({typeOf})",typeof=typeOf)
            else :
                
                score = cross_val_score(br, X, y, cv=5, scoring='r2')
                st.sidebar.subheader(f"🧩 Adaboost R²: {np.round(score.mean(), 3)}")
                if X.shape[1] == 1:  # 1D regression
                    br.fit(X, y)
                    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_pred = br.predict(x_line)
                    axs[-1].scatter(X, y, color='blue', label='Data')
                    axs[-1].plot(x_line, y_pred, color='red', label='Adaboost Prediction')
                    axs[-1].set_title("Adaboost Regressor")
                else:  # 2D regression
                    # use existing meshgrid plot if you want
                    plot_decision_region(axs[0], m, X, y, title=m.__class__.__name__,typeof=typeOf)
                    pass

            st.pyplot(fig)
# Adaboost()