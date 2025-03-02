import streamlit as st
import plotly.graph_objects as go
import io
import trimesh
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'Model')))
from Functions import CapsNet3D


# Configuration de la barre latérale
st.sidebar.header("1. Importation de fichier")

#-------------------------------------------------------------- Conversion du maillage en voxels
def load_obj_as_voxel(filepath, grid_size):
    # Load the mesh from the .obj file
    mesh = trimesh.load(filepath)

    # Convert the mesh to a voxel grid
    voxel_grid = mesh.voxelized(pitch=mesh.extents.max()/grid_size)

    # Convert voxel grid to a numpy array
    voxel_data = voxel_grid.matrix.astype(np.float32)

    # Resize voxel data to grid_size x grid_size x grid_size if necessary
    current_shape = voxel_data.shape
    if current_shape != (grid_size, grid_size, grid_size):
        zoom_factors = [g/c for g, c in zip((grid_size, grid_size, grid_size), current_shape)]
        voxel_data = zoom(voxel_data, zoom_factors, order=0)  # Nearest-neighbor interpolation
    voxel_data = voxel_data.reshape(-1, grid_size, grid_size, grid_size, 1)
    return voxel_data

# Conversion du maillage en graphique Plotly
def obj_to_plotly(mesh):
    try:
        x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
        faces = mesh.faces
        colors = ['#fbefef', '#010169'] * (len(faces) // 2)  # Couleurs alternantes

        trace = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            facecolor=colors,
            opacity=0.5,
            lighting=dict(ambient=0.3, diffuse=0.5),
            flatshading=True
        )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title='',
                xaxis=dict(showgrid=False, showticklabels=False, showline=False),
                yaxis=dict(showgrid=False, showticklabels=False, showline=False),
                zaxis=dict(showgrid=False, showticklabels=False, showline=False)
            ),
            margin=dict(r=0, l=0, b=0, t=30)
        )
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique : {e}")
        return None

# Fonction pour afficher la répartition des classes
def plot_class_distribution(predictions, class_names):
    try:
        trace = go.Pie(
            labels=class_names,
            values=predictions,
            hole=0.3,
            marker=dict(colors=['#fbefef' if i == np.argmax(predictions) else '#010169' for i in range(len(class_names))]),
        )

        fig = go.Figure(data=[trace])
        fig.update_layout(
            margin=dict(r=0, l=0, b=0, t=30),
            annotations=[
                dict(
                    text='Répartition des Classes Prédictives',
                    x=0.5,
                    y=-0.1,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    font=dict(size=14, color='black')
                )
            ]
        )
        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de répartition : {e}")
        return None

# Définition du style HTML pour les titres
st.markdown("""
<style>
h2 {
    font-size: 24px !important;
    text-align: justify;
    font-family: Arial, Helvetica, sans-serif;
    color: #010169 !important;
    text-overflow: ellipsis;
}
.big-font {
    font-size: 24px !important;
    text-align: justify;
    font-family: Arial, Helvetica, sans-serif;
    color: #01213A;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)

# Titre et instructions
st.markdown("""
<div class="big-font">
<h2>Chargement d'un fichier .obj</h2>
<p>Assurez-vous d'importer un fichier .obj afin d'examiner les données 3D de manière appropriée.</p>
</div>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False
if 'mesh' not in st.session_state:
    st.session_state.mesh = None

# Chargement du fichier
uploaded_file = st.file_uploader("Choisissez un fichier...", type=["obj"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.obj'):
        st.success("Fichier importé avec succès!")
        try:
            file_content = uploaded_file.read()
            st.session_state.mesh = trimesh.load(io.BytesIO(file_content), file_type='obj')
            st.session_state.uploaded = True
        except Exception as e:
            st.error(f"Erreur lors du chargement du maillage : {e}")
            st.session_state.uploaded = False
    else:
        st.error("Erreur: Veuillez importer un fichier au format .obj.")

# Bouton de réinitialisation
if st.button("Restart"):
    st.session_state.uploaded = False
    st.session_state.mesh = None
    st.stop()

# Affichage du maillage si chargé
if st.session_state.uploaded and st.session_state.mesh is not None:
    st.markdown('<div class="big-font"><h2>Affichage de Modèle 3D</h2></div>', unsafe_allow_html=True)
    fig = obj_to_plotly(st.session_state.mesh)
    if fig:
        st.plotly_chart(fig)



# Transformation en voxel 
if st.session_state.uploaded and st.session_state.mesh is not None:
    voxel = load_obj_as_voxel(st.session_state.mesh, grid_size=30)
   
#----------------------------------------------------------------------------------
# Chargement du modèle et prédiction


train_model, eval_model = CapsNet3D(input_shape=(30, 30, 30, 1), n_class=2, routings=3)
model = eval_model

# Step 4: Load the weights file using the relative path
weights_path = os.path.join(os.path.dirname(__file__), '..','..', 'Model', 'weights-47.weights.h5')
#weights_path=r"C:\Users\SOUHAILA ELKADAOUI\Desktop\code (1)\code\APP_Web\weights-47.weights.h5"
model.load_weights(weights_path)

def compute_proba(maxim, minim, prediction):
    somme = maxim + minim
    if prediction == 0:
        return [(maxim*100)/somme, (minim*100)/somme]
    elif prediction == 1:
        return [(minim*100)/somme, (maxim*100)/somme]

if st.session_state.uploaded and st.session_state.mesh is not None:
    try:
        st.write("Analyse en cours...")
        
        if voxel is not None:
            predictions = model.predict(voxel)
            predictions_decoded = np.argmax(predictions, axis=1)
            class_names = ['Aneurysm', 'Vessel']
            prediction_proba = compute_proba(max(predictions[0]), min(predictions[0]), prediction = predictions_decoded)
            if predictions_decoded == 0:
                st.markdown('<div class="big-font"><h2>Résultat de dignostic de fichier</h2></div>', unsafe_allow_html=True)
                st.markdown(f"""
                <p class="big-font">Attention : Votre donnée indique l'existence d'un anévrisme avec une probabilité de {prediction_proba[0]:.2f}%. Veuillez consulter un professionnel de la santé .</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <p class="big-font">Résultat normal : Votre donnée est classée comme 'Vessel' avec une probabilité de {prediction_proba[1]:.2f}%. Votre état semble normal, mais il est toujours conseillé de consulter un professionnel de la santé pour confirmation.</p>
                """, unsafe_allow_html=True)
            st.sidebar.header("3. Visualisation de résultat")
            pie_fig = plot_class_distribution(prediction_proba, class_names)
            if pie_fig:
                st.plotly_chart(pie_fig)

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du maillage : {e}")
