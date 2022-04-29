# Carregando as bibliotecas
from math import prod
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import streamlit as st
from minio import Minio
import joblib
from pycaret.classification import load_model, predict_model


# Baixando os arquivos do Data Lake
client = Minio('localhost:9000',
               access_key='minio_admin',
               secret_key='minioadmin',
               secure=False
               )

# Modelo de Classificação, dataset e cluster
client.fget_object("curated", "model.pkl", "model.pkl")
client.fget_object("curated", "dataset.csv", "dataset.csv")
client.fget_object("curated", "cluster.joblib", "cluster.joblib")

var_model = "model"
var_dataset = "dataset.csv"
var_model_cluster = "cluster.joblib"

# Carregando o Modelo treinado
model = load_model(var_model)
model_cluster = joblib.load(var_model_cluster)

# Carregando o conjunto de dados
dataset = pd.read_csv(var_dataset)

# Título
st.title("Human Resource Analytics")

# Subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de Human Resources Analytics")

# Imprime o conjunto de Dados
st.dataframe(dataset.drop("turnover", axis=1).head())

# Grupos de Colaboradores
kmeans_colors = ['green' if  c == 0 else 'red' if c == 1 else 'blue' for c in model_cluster.labels_]

st.sidebar.subheader("Defina os atributos do Colaborador para predição do turnover")

# Mapeando dados do usuário para cada atributo
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# Inserindo um botão na tela
btn_predict = st.sidebar.button("Realizar Classificação")

# Verifica se o botão foi acionado
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] = [evaluation]
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]
    
# Imprime os dados de Teste
result = predict_model(model, data=data_teste)

# Realiza a predição
classe = result["Label"][0]
prob = result["Score"][0]*100

if classe==1:
    st.write("A predição do modelo para a amostra de teste é de evasão com o valor de probabilidade: {0:.2f}%".format(prob))
else:
    st.write("A predição do modelo para a amostra de teste é de permanência com o valor de probabilidade: {0:.2f}%".format(prob))

fig = plt.figure(figsize=(10,6))
plt.scatter(x="satisfaction"
            ,y="evaluation"
            ,data=dataset[dataset.turnover==1]
            ,alpha=0.25
            ,color=kmeans_colors)

plt.xlabel("Satisfaction")
plt.ylabel("Evaluation")

plt.scatter(x=model_cluster.cluster_centers_[:,0]
            ,y=model_cluster.cluster_centers_[:,1]
            ,color="black"
            ,marker="X",s=100)

plt.scatter(x=[satisfaction]
            ,y=[evaluation]
            ,color="yellow"
            ,marker="X",s=300)

plt.title("Grupos de Colaboradores - Satisfação vs Avaliação")
plt.show()
plt.pyplot(fig)