# -*- coding: utf-8 -*-
import os
import cv2
import streamlit as st
from scripts import Extractor, Retrievor
from preprocessors import ImageToArrayPreprocessor
from preprocessors import AspectAwarePreprocessor


# ### FUNCTION ###
@st.cache
def load_extractor():
    return Extractor('VGG16')


# ### PAGES ###
# header
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
extractor_list = ['AKAZE', 'ORB', 'SURF', 'VGG16']
distance_list = ['cosinus', 'manhattan', 'euclidean']
test_list = list(os.listdir('./data/test'))
akaze, orb, surf = Extractor('AKAZE'), Extractor('ORB'), Extractor('SURF')
vgg = load_extractor()
akaze_features, orb_features, surf_features, vgg_features = [
        Retrievor('./features/AKAZE_features.pck'),
        Retrievor('./features/ORB_features.pck'),
        Retrievor('./features/SURF_features.pck'),
        Retrievor('./features/VGG16_features.pck')
]
# header
st.title("Recherche d'image basée sur le contenu")
# sidebar
pages = ['Accueil', 'Essayer']
choice = st.sidebar.selectbox('Menu', pages)

if choice == 'Accueil':
    st.write(
        '''
            Cette application implemente la recherche d'image basée sur le contenu.
            Elle fonctionne en deux étapes: l'extraction de caractèristiques et 
            recherche de l'image avec les caractéristiques les plus similaires 
            (et ceux avec une mesure de similarité).
        '''
    )
    st.write(
        '''
            Pour l'extraction de caractéristiques, vous avez le choix entre un 
            reseau de neurones pré-entraîné ([VGG16](https://neurohive.io/en/popular-networks/vgg16/) 
            avec imagenet) et des
            descrpteurs tels que [AKAZE](https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html), 
            [ORB](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf) et 
            [SURF](https://fr.wikipedia.org/wiki/Speeded_Up_Robust_Features).
        '''
    )
    st.write(
        '''
            Pour la mesure de similarité, vous avez le choix entre:
            [cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus), 
            [manhattan](https://fr.wikipedia.org/wiki/Distance_de_Manhattan) et 
            [euclidien](https://fr.wikipedia.org/wiki/Distance_(math%C3%A9matiques)).
        '''
    )
    st.write(
        "Le code est open source et disponible sur "
        "[GitLab](https://gitlab.com/schalappe/content-based-image-retrieval)."
    )
elif choice == 'Essayer':
    # side bar
    extractor = st.sidebar.selectbox("Choix de la méthode d'extraction", extractor_list)
    distance = st.sidebar.selectbox("Choix de la mesure de similarité", distance_list)
    depth = st.sidebar.slider("Combien d'éléments", 0, 4, 1)
    image = st.sidebar.selectbox("Choix d'une image", test_list)
    st.sidebar.image('./data/test/'+image, use_column_width=True)
    # recherche
    images = []
    if extractor == 'AKAZE':
        img = cv2.imread('./data/test/'+image)
        img = aap.preprocess(img)
        features = akaze.extract(img)
        images, _ = akaze_features.search(features, distance, depth)
    elif extractor == 'ORB':
        img = cv2.imread('./data/test/'+image)
        img = aap.preprocess(img)
        features = orb.extract(img)
        images, _ = orb_features.search(features, distance, depth)
    elif extractor == 'SURF':
        img = cv2.imread('./data/test/'+image)
        img = aap.preprocess(img)
        features = surf.extract(img)
        images, _ = surf_features.search(features, distance, depth)
    elif extractor == 'VGG16':
        img = cv2.imread('./data/test/'+image)
        img = aap.preprocess(img)
        img = iap.preprocess(img)
        features = vgg.extract(img)
        images, _ = vgg_features.search(features, distance, depth)
    for image in images:
        st.image(image, width=224)
