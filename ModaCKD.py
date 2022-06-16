#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cufflinks
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.io as pio 
import plotly.graph_objects as go
import warnings
import streamlit as st
warnings.filterwarnings("ignore")
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
pio.renderers.default = "notebook" # should change by looking into pio.renderers
from random import shuffle
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from colors import *
import plotly.express as px
from numpy import where
from numpy import unique
from sklearn.mixture import GaussianMixture
from scipy import stats
import scipy.integrate
import scipy.special
import scipy
from PIL import Image

data = pd.read_csv("testckd2.csv")
data.describe()

columns = ['Diabetes', 'Behavioral Health/Substance Abuse History',
       'Average_Adherance', 'Inpatient_CKD_Visits_by_Month',
       'Outpatient_Visits_by_Month', 'Comorbidity']

img = Image.open("AArete.png")

def get_colors():
    s='''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
    li=s.split(',')
    li=[l.replace('\n','') for l in li]
    li=[l.replace(' ','') for l in li]
    shuffle(li)
    return li

colors = get_colors()

mode = st.sidebar.radio("Mode", ["Clustering"])
st.markdown("<h1 style='text-align: left; color: #f68b28;'>CKD-Moda Health</h1>", unsafe_allow_html=True)
st.markdown("# Mode: {}".format(mode), unsafe_allow_html=True)
st.image(img, width=100, caption='AArete LLC; Humanizing Data for Purpusful Change')


if mode=="Clustering":    
    features = st.sidebar.multiselect("Select Features", columns, default=columns)

    # select a clustering algorithm
    calg = st.sidebar.selectbox("Select a clustering algorithm", ["K-Means","K-Medoids", "Spectral Clustering", "Gaussian Mixture Clustering", "Agglomerative Clustering"])

    # select number of clusters
    ks = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=2)

    # select a dataframe to apply cluster on


    if len(features)>=2:

        if calg == "K-Means":
            st.markdown("### K-Means Clustering")        
            use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
            if use_pca=="No":
                st.markdown("### Not Using PCA")
                inertias = []
                for c in range(2,ks+1):
                    X = data[features]                
                    # colors=['red','green','blue','magenta','black','yellow']
                    model = KMeans(n_clusters=c, n_init=10, random_state = 29)
                    model.fit(X)
                    y_kmeans = model.predict(X)
                    centers = model.cluster_centers_
                    clusters_df = pd.DataFrame(centers, columns=features)
                    cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                    dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                    fig = go.Figure(data=dataa)
                    # Change the bar mode
                    fig.update_layout(barmode='group')
                    st.plotly_chart(fig)

            if use_pca=="Yes":
                st.markdown("### Using PCA")
                comp = st.sidebar.number_input("Choose Components",1,10,3)

                X = data[features]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=int(comp))
                principalComponents = pca.fit_transform(X_scaled)
                feat = list(range(pca.n_components_))
                PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
                chosen_component = st.sidebar.multiselect("Choose Components",feat,default=[1,2])
                chosen_component=[int(i) for i in chosen_component]
                inertias = []
                if len(chosen_component)>1:
                    for c in range(2,ks+1):
                        X = PCA_components[chosen_component]

                        model = KMeans(n_clusters=c)
                        model.fit(X)

                        trace0 = go.Scatter(x=X[chosen_component[0]],y=X[chosen_component[1]],mode='markers',  
                                            marker=dict(
                                                    color=colors,
                                                    colorscale='Viridis',
                                                    showscale=True,
                                                    opacity = 0.9,
                                                    reversescale = True,
                                                    symbol = 'pentagon'
                                                    ))

                        trace1 = go.Scatter(x=model.cluster_centers_[:, 0], y=model.cluster_centers_[:, 1],
                                            mode='markers', 
                                            marker=dict(
                                                color=colors,
                                                size=20,
                                                symbol="circle",
                                                showscale=True,
                                                line = dict(
                                                    width=1,
                                                    color='rgba(102, 102, 102)'
                                                    )

                                                ),
                                            name="Cluster Mean")

                        data7 = go.Data([trace0, trace1])
                        fig = go.Figure(data=data7)

                        layout = go.Layout(
                                    height=600, width=800, title=f"KMeans Cluster Size {c}",
                                    xaxis=dict(
                                        title=f"Component {chosen_component[0]}",
                                    ),
                                    yaxis=dict(
                                        title=f"Component {chosen_component[1]}"
                                    ) ) 
                        fig.update_layout(layout)
                        st.plotly_chart(fig)                                 
                    
        elif calg == "K-Medoids":
            st.markdown("### K-Medoids Clustering")        
            use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
            if use_pca=="No":
                st.markdown("### Not Using PCA")
                inertias = []
                for c in range(2,ks+1):
                    X = data[features]                
                    Y = X.to_numpy()
                    # colors=['red','green','blue','magenta','black','yellow']
                    model = KMedoids(n_clusters=c, random_state = 0)
                    model.fit(Y)
                    y_kmedoids = model.predict(Y)
                    centers = model.cluster_centers_
                    clusters_df = pd.DataFrame(centers, columns=features)
                    cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                    dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                    fig = go.Figure(data=dataa)
                    # Change the bar mode
                    fig.update_layout(barmode='group')
                    st.plotly_chart(fig)

        elif calg == "Spectral Clustering":
            st.markdown("### Spectral Clustering")        
            use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
            if use_pca=="No":
                st.markdown("### Not Using PCA")
                inertias = []
                for c in range(2,ks+1):
                    X = data[features]                
                    # colors=['red','green','blue','magenta','black','yellow']
                    model = SpectralClustering(n_clusters=c, assign_labels='discretize', random_state=0)
                    model.fit(X)
                    centers = model.cluster_centers_
                    clusters_df = pd.DataFrame(centers, columns=features)
                    cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                    dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                    fig = go.Figure(data=dataa)
                    # Change the bar mode
                    fig.update_layout(barmode='group')
                    st.plotly_chart(fig)

        elif calg == "Gaussian Mixture Clustering":
            st.markdown("### GMM Clustering")        
            use_pca = st.sidebar.radio("Use PCA?",["Yes","No"])
            if use_pca=="No":
                st.markdown("### Not Using PCA")
                inertias = []
                for c in range(2,ks+1):
                    X = data[features]                
                    # colors=['red','green','blue','magenta','black','yellow']
                    Y = X.to_numpy()
                    gmm = GaussianMixture(n_components=c, covariance_type='full').fit(Y)
                    centers = np.empty(shape=(gmm.n_components, Y.shape[1]))
                    for i in range(gmm.n_components):
                        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y)
                        centers[i, :] = Y[np.argmax(density)]
                    clusters_df = pd.DataFrame(centers, columns=features)
                    cluster_names = [f'Cluster {k}' for k in range(1,c+1, 1)]  # clust 1, 2, 3
                    dataa = [go.Bar(name=f, x=cluster_names, y=clusters_df[f]) for f in clusters_df.columns]
                    fig = go.Figure(data=dataa)
                    # Change the bar mode
                    fig.update_layout(barmode='group')
                    st.plotly_chart(fig)

            if use_pca=="Yes":
                st.markdown("### Using PCA")
                comp = st.sidebar.number_input("Choose Components",1,10,3)

                X = data[features]
                Y = X.to_numpy()
                scaler = StandardScaler()
                Y_scaled = scaler.fit_transform(Y)

                pca = PCA(n_components=int(comp))
                principalComponents = pca.fit_transform(Y_scaled)
                feat = list(range(pca.n_components_))
                PCA_components = pd.DataFrame(principalComponents, columns=list(range(len(feat))))
                chosen_component = st.sidebar.multiselect("Choose Components",feat,default=[1,2])
                chosen_component=[int(i) for i in chosen_component]

                if len(chosen_component)>1:
                    for c in range(2,ks+1):
                        Y = PCA_components[chosen_component]


                        gmm = GaussianMixture(n_components=c, covariance_type='full').fit(Y)
                        centers = np.empty(shape=(gmm.n_components, Y.shape[1]))
                        for i in range(gmm.n_components):
                            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y)
                            centers[i, :] = Y[np.argmax(density)]
                        
                        trace0 = go.Scatter(x=Y[chosen_component[0]],y=Y[chosen_component[1]],mode='markers',  
                                            marker=dict(
                                                    color=colors,
                                                    colorscale='Viridis',
                                                    showscale=True,
                                                    opacity = 0.9,
                                                    reversescale = True,
                                                    symbol = 'pentagon'
                                                    ))

                        trace1 = go.Scatter(x=centers[:, 0], y=centers[:, 1],
                                            mode='markers', 
                                            marker=dict(
                                                color=colors,
                                                size=20,
                                                symbol="circle",
                                                showscale=True,
                                                line = dict(
                                                    width=1,
                                                    color='rgba(102, 102, 102)'
                                                    )

                                                ),
                                            name="Cluster Mean")

                        data7 = go.Data([trace0, trace1])
                        fig = go.Figure(data=data7)

                        layout = go.Layout(
                                    height=600, width=800, title=f"KMeans Cluster Size {c}",
                                    xaxis=dict(
                                        title=f"Component {chosen_component[0]}",
                                    ),
                                    yaxis=dict(
                                        title=f"Component {chosen_component[1]}"
                                    ) ) 
                        fig.update_layout(layout)
                        st.plotly_chart(fig)             
