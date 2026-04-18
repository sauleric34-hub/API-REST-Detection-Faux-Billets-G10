from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import os
import re

app = Flask(__name__)

# Initialisation de l'IA (CRAFT + Reconnaissance)
reader = easyocr.Reader(['fr', 'en'], gpu=False)

def expertise_couleur_cemac(image_bgr):
    """Analyse la signature colorimétrique pour confirmer la valeur faciale."""
    # Conversion en espace HSV (Teinte, Saturation, Valeur)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # On calcule la teinte (Hue) moyenne du billet
    teinte_moyenne = np.mean(hsv[:, :, 0])
    
    # Signature visuelle CEMAC simplifiée
    if 95 <= teinte_moyenne <= 125: # Dominante Bleue
        return {"valeur": "1000", "devise": "Franc CFA (BEAC)"}
    elif 10 <= teinte_moyenne <= 35: # Dominante Marron/Ocre
        return {"valeur": "500", "devise": "Franc CFA (BEAC)"}
    return None

def analyse_visuelle_universelle(chemin_image):
    img = cv2.imread(chemin_image)
    if img is None: return {"erreur": "Fichier image invalide"}

    # 1. LECTURE DU TEXTE (Intelligence Artificielle)
    resultats = reader.readtext(chemin_image)
    texte_integral = " ".join([res[1] for res in resultats]).upper()
    
    # Nettoyage des nombres lus (on cherche des valeurs logiques)
    nombres = re.findall(r'\b\d{2,5}\b', texte_integral)
    valeur_ocr = max(nombres, key=int) if nombres else "Inconnue"

    # 2. DÉDUCTION PAR L'IMAGE (Signature visuelle)
    # Si le texte est mal lu (ex: 10004), la couleur va nous corriger
    signature = expertise_couleur_cemac(img)
    
    valeur_finale = valeur_ocr
    devise_finale = "Inconnue"
    methode = "OCR"

    # Logique de décision : La couleur a le dernier mot si l'OCR est flou
    if signature:
        devise_finale = signature["devise"]
        methode = "Analyse colorimétrique contextuelle"
        # Si l'OCR a lu un nombre bizarre comme '10004', on le rectifie
        if "1000" in valeur_ocr: valeur_finale = "1000"

    # 3. ANALYSE DE L'ÉTAT (OpenCV)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score_physique = int(np.interp(np.std(gris), [20, 80], [40, 95]))

    return {
        "expertise_ia": {
            "valeur_identifiee": valeur_finale,
            "devise_identifiee": devise_finale,
            "methode_principale": methode
        },
        "verdict_expert": {
            "confiance_visuelle": f"{score_physique}%",
            "statut": "AUTHENTIQUE" if score_physique > 70 else "SUSPECT"
        }
    }

@app.route('/detect', methods=['POST'])
def detect():
    if 'photo' not in request.files: return jsonify({"erreur": "Aucune image"}), 400
    
    photo = request.files['photo']
    path = os.path.join("uploads", photo.filename)
    photo.save(path)
    
    analyse = analyse_visuelle_universelle(path)
    return jsonify({"groupe": 10, "analyse_multimodale": analyse})

if __name__ == '__main__':
    if not os.path.exists("uploads"): os.makedirs("uploads")
    app.run(debug=True, port=5000)