from stmol import *
import streamlit as st
import base64

# Charger le fichier CSS
css_file_path = "style.css"  # Chemin relatif
with open(css_file_path, 'r') as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Charger et encoder les images en base64
image_path = "nerve.png"
image2_path = "steps.png"
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

with open(image2_path, "rb") as image_file:
    encoded_image2 = base64.b64encode(image_file.read()).decode()

# Titre
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="data:image/png;base64,{encoded_image}" width="50" style="margin-right: 20px;">
        <h1 style=" font-family: 'Times New Roman'; color:#ed1b24; ">Diagnostic d'Anévrisme</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    body {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
.big-font {
    font-size: 24px !important;
    text-align: justify;
    font-family: Arial, Helvetica, sans-serif;
    color: #01213A;
    text-overflow: ellipsis;
}
</style>
""", unsafe_allow_html=True)

texte_html = """
<div class="big-font">
Bienvenue sur notre application web, votre service de consultation médicale dédié au diagnostic des anévrismes. Nous sommes ravis de vous accompagner dans votre démarche de santé. Notre service vous aide à déterminer rapidement si vous présentez un risque d'anévrisme.<br>
Les étapes à suivre pour évaluer votre situation sont détaillées ci-dessous. </p> 
</div>"""
st.markdown(texte_html, unsafe_allow_html=True)

st.markdown(
    f"""
    <div style=" align-items: center; justify-content: center;">
        <img src="data:image/png;base64,{encoded_image2}">
    </div>
    """,
    unsafe_allow_html=True
)
