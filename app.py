"""
Système de Détection de Faux Billets - Projet Étatique
Auteur: Système IA - Usage Police/Gouvernement uniquement
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import easyocr
import os
import re
import base64
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

app = Flask(__name__)

# EasyOCR : l'arabe est incompatible avec le français dans le même reader
# On utilise deux readers séparés et on fusionne les résultats
reader_latin  = easyocr.Reader(['fr', 'en'], gpu=False)          # Français + Anglais
reader_arabic = easyocr.Reader(['ar', 'en'], gpu=False)          # Arabe + Anglais (BEAC/BCEAO ont du texte arabe)

# ─────────────────────────────────────────────
# STRUCTURES DE DONNÉES
# ─────────────────────────────────────────────

@dataclass
class ResultatAnalyse:
    decision: str
    score_global: int
    niveau_confiance: str
    devise: str
    emetteur: str
    numero_serie: str
    anomalies: list
    scores_details: dict
    timestamp: str
    recommandation: str


# ─────────────────────────────────────────────
# MODULE 1 : ANALYSE TEXTE / OCR
# ─────────────────────────────────────────────

EMETTEURS_CONNUS = {
    "BEAC": {
        "noms": [
            "BANQUE DES ETATS DE L AFRIQUE CENTRALE",
            "BANQUE DES ÉTATS DE L'AFRIQUE CENTRALE",
            "BEAC", "B.E.A.C"
        ],
        "devises": ["FRANC", "FRANCS", "FCFA", "CFA"],
        "valeurs_valides": [500, 1000, 2000, 5000, 10000],
        "zones": ["C5", "E5", "D5", "F5"],
    },
    "BCEAO": {
        "noms": [
            "BANQUE CENTRALE DES ETATS DE L AFRIQUE DE L OUEST",
            "BANQUE CENTRALE DES ÉTATS DE L'AFRIQUE DE L'OUEST",
            "BCEAO", "B.C.E.A.O", "B C E A O",
            # OCR lit parfois chaque lettre séparément
            "BCEO", "BCE AO",
        ],
        "devises": ["FRANC", "FRANCS", "FCFA", "CFA", "DIX MILLE", "MILLE"],
        "valeurs_valides": [500, 1000, 2000, 5000, 10000],
        "zones": ["K", "A", "B", "C", "D", "H", "S", "T"],
    },
    "BANK OF CANADA": {
        "noms": ["BANK OF CANADA", "BANQUE DU CANADA"],
        "devises": ["DOLLAR", "DOLLARS"],
        "valeurs_valides": [5, 10, 20, 50, 100],
        "zones": [],
    },
}

# Mots-clés qui ne doivent PAS apparaître sur un vrai billet
MOTS_SUSPECTS = [
    "COPY", "SPECIMEN", "VOID", "FAKE", "PROP", "MOTION PICTURE",
    "CADAUA",  # faute sur billet canadien détecté
    "REPLICA", "REPRODUCTION",
]

# Regex numéros de série par zone
FORMATS_SERIE = {
    "BEAC":           r"^\d{7,9}[A-Z]\d$",       # ex: 00322179C5 (flexible)
    "BCEAO":          r"^\d{9,12}[A-Z]$",         # ex: 09693871680K (flexible)
    "BANK OF CANADA": r"^[A-Z]{2,3}\d{6,8}$",    # ex: EJX1234567 (flexible)
}

def detecter_image_composite(img: np.ndarray) -> bool:
    """
    Détecte si l'image contient plusieurs billets côte à côte (recto/verso).
    Cas typique : images de catalogue avec recto en haut et verso en bas.
    Si composite → on analyse chaque moitié séparément.
    """
    h, w = img.shape[:2]
    ratio = max(w, h) / min(w, h)
    # Image quasi-carrée avec grande résolution → probable composite recto/verso
    if ratio < 1.3 and min(w, h) > 400:
        # Vérifier discontinuité horizontale (ligne blanche entre les deux billets)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Chercher une ligne quasi-blanche au milieu (séparateur)
        milieu = h // 2
        zone_centrale = gris[milieu-20:milieu+20, :]
        moy_centrale = float(np.mean(zone_centrale))
        moy_globale  = float(np.mean(gris))
        return moy_centrale > moy_globale + 15  # ligne plus claire = séparateur
    return False

def evaluer_usure(img: np.ndarray) -> dict:
    """
    Évalue le niveau d'usure PHYSIQUE réelle du billet.
    IMPORTANT : on distingue usure réelle vs image de catalogue haute qualité.
    - Image catalogue (scan propre) → variance haute + saturation haute = NEUF
    - Billet usé → délavage + irrégularités locales de couleur
    Retourne coefficient 0 (neuf/catalogue) à 1 (très usé/délabré).
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Délavage : saturation basse ET faible contraste = délavé
    saturation = float(np.mean(hsv[:, :, 1]))
    # Un catalogue propre a une saturation normale (>50), un billet délavé < 40
    score_delavage = max(0.0, 1.0 - saturation / 70.0)

    # 2. Froissement RÉEL : variance locale disproportionnée vs variance globale
    # Un catalogue HD a variance locale ET globale élevées (cohérent → pas froissé)
    # Un billet froissé a des pics locaux incohérents avec le niveau global
    variance_locale = _variance_locale(gris, bloc=8)
    variance_globale = float(np.var(gris))
    # Ratio : si variance_locale >> variance_globale/10 → incohérence = froissement
    ratio_coherence = variance_locale / max(variance_globale / 10.0, 1.0)
    # Un billet neuf/catalogue : ratio proche de 1. Froissé : ratio > 3
    score_froissement = min(1.0, max(0.0, (ratio_coherence - 1.0) / 4.0))

    # 3. Taches réelles : zones sombres ISOLÉES (pas le fond du design)
    # On cherche des taches locales en comparant chaque bloc à son voisinage
    score_taches = _detecter_taches_locales(gris)

    # 4. Indicateur de qualité image : image floue → peut sembler usée
    # Laplacian : flou = faible, net = élevé
    laplacian_var = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    est_flou = laplacian_var < 100  # image floue → on réduit le score d'usure

    # Coefficient global d'usure
    usure_brute = float(np.mean([score_delavage, score_froissement, score_taches]))

    # Si image très nette (catalogue/scan), le froissement calculé est un artefact
    if not est_flou and saturation > 50:
        # Image de bonne qualité → on fait confiance aux vraies métriques
        usure_finale = usure_brute * 0.7  # légère correction vers le bas
    else:
        usure_finale = usure_brute

    usure_finale = round(min(1.0, usure_finale), 3)
    niveau = ("NEUF" if usure_finale < 0.2
              else "USAGÉ" if usure_finale < 0.45
              else "TRÈS USAGÉ" if usure_finale < 0.70
              else "DÉLABRÉ")

    return {
        "coefficient": usure_finale,
        "niveau": niveau,
        "details": {
            "delavage":     round(score_delavage, 3),
            "froissement":  round(score_froissement, 3),
            "taches":       round(score_taches, 3),
            "nettete_image":round(min(laplacian_var / 500, 1.0), 3),
        }
    }


def _detecter_taches_locales(gris: np.ndarray, bloc: int = 32) -> float:
    """Détecte des taches isolées (zones sombres anormales vs voisinage)."""
    h, w = gris.shape
    anomalies_count = 0
    total_blocs = 0
    for y in range(0, h - bloc, bloc):
        for x in range(0, w - bloc, bloc):
            patch = gris[y:y+bloc, x:x+bloc]
            moy_patch = float(np.mean(patch))
            # Comparer avec une zone plus large autour
            y1, y2 = max(0, y-bloc), min(h, y+2*bloc)
            x1, x2 = max(0, x-bloc), min(w, x+2*bloc)
            voisinage = gris[y1:y2, x1:x2]
            moy_voisin = float(np.mean(voisinage))
            # Tache = patch anormalement plus sombre que son voisinage
            if moy_voisin > 30 and (moy_voisin - moy_patch) > 40:
                anomalies_count += 1
            total_blocs += 1
    return min(1.0, anomalies_count / max(total_blocs * 0.15, 1))


def analyser_texte(chemin_image: str) -> dict:
    """
    Extrait et analyse tout le texte du billet.
    PRINCIPE : on part d'un score NEUTRE (50) et on ajuste.
    Un billet usé peut avoir un OCR faible → ce n'est PAS une preuve de faux.
    Seuls les MOTS INTERDITS (COPY, CADAUA...) sont rédhibitoires.
    """
    # Fusion des deux readers (latin + arabe)
    resultats_latin  = reader_latin.readtext(chemin_image, detail=1)
    resultats_arabic = reader_arabic.readtext(chemin_image, detail=1)
    resultats_ocr    = resultats_latin + resultats_arabic

    # Score de départ : présomption d'authenticité
    score = 50
    anomalies = []
    emetteur_detecte = "Inconnu"
    devise_detectee  = "Inconnue"
    numero_serie     = "Non détecté"
    valeur_numerique = None
    confiance_moy    = 0.0

    if not resultats_ocr:
        # Pas de texte lisible ≠ faux automatique (billet très usé, photo floue)
        return {
            "score": score,
            "anomalies": ["Texte non lisible (billet usé ou photo de faible qualité) - non rédhibitoire"],
            "texte_brut": "",
            "emetteur": emetteur_detecte,
            "devise": devise_detectee,
            "numero_serie": numero_serie,
            "valeur": valeur_numerique,
            "confiance_ocr": 0.0,
        }

    textes      = [r[1] for r in resultats_ocr]
    confiances  = [r[2] for r in resultats_ocr]
    texte_brut  = " ".join(textes).upper()

    # ── Normalisations multiples ──
    # 1. Supprimer points entre lettres : B.C.E.A.O → BCEAO
    texte_normalise = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '', texte_brut)
    # 2. Reconstruire acronymes de lettres isolées : "B C E A O" → "BCEAO"
    #    Pattern : 1 lettre, espace, 1 lettre, espace... (min 3 occurrences)
    texte_normalise = re.sub(r'\b([A-Z])\s([A-Z])\s([A-Z])\s([A-Z])\s([A-Z])\b',
                             r'\1\2\3\4\5', texte_normalise)
    texte_normalise = re.sub(r'\b([A-Z])\s([A-Z])\s([A-Z])\s([A-Z])\b',
                             r'\1\2\3\4', texte_normalise)
    # 3. Coller les tokens numériques + lettre finale séparés par espace
    #    "09693871680 K" → "09693871680K"
    texte_normalise = re.sub(r'(\d{7,12})\s+([A-Z])\b', r'\1\2', texte_normalise)
    confiance_moy = float(np.mean(confiances))

    # ── 1. Mots INTERDITS → seule vraie pénalité forte ──
    for mot in MOTS_SUSPECTS:
        if mot in texte_normalise:
            anomalies.append(f"🚨 MOT INTERDIT DÉTECTÉ: '{mot}' - preuve de contrefaçon")
            score -= 50

    # ── 2. Émetteur reconnu → bonus ──
    for nom_emetteur, infos in EMETTEURS_CONNUS.items():
        for nom_possible in infos["noms"]:
            if nom_possible in texte_normalise:
                emetteur_detecte = nom_emetteur
                score += 15
                break

    # ── 3. Devise reconnue → bonus ──
    for emetteur, infos in EMETTEURS_CONNUS.items():
        for devise in infos["devises"]:
            if devise in texte_normalise:
                devise_detectee = devise
                score += 8
                break

    # ── 4. Numéro de série : cherche token unique ET combinaisons adjacentes ──
    # a) Tokens simples
    tokens = re.split(r'\s+', texte_normalise)
    for token in tokens:
        token_clean = re.sub(r'[^A-Z0-9]', '', token)
        for zone, pattern in FORMATS_SERIE.items():
            if re.fullmatch(pattern, token_clean):
                numero_serie = token_clean
                score += 20
                if zone == emetteur_detecte:
                    score += 5
                break
        if numero_serie != "Non détecté":
            break

    # b) Si non trouvé, chercher dans le texte brut avec regex directement
    if numero_serie == "Non détecté":
        candidats = re.findall(r'\d{7,12}[A-Z]', texte_normalise)
        if candidats:
            numero_serie = candidats[0]
            score += 15  # bonus un peu réduit (moins de confiance)

    # ── 5. Valeur nominale cohérente → bonus ──
    nombres = re.findall(r'\b(500|1000|2000|5000|10000|20|50|100)\b', texte_normalise)
    if nombres:
        valeur_numerique = int(nombres[0])
        if emetteur_detecte in EMETTEURS_CONNUS:
            valides = EMETTEURS_CONNUS[emetteur_detecte]["valeurs_valides"]
            if valeur_numerique in valides:
                score += 10
            else:
                anomalies.append(f"Valeur {valeur_numerique} inhabituelle pour {emetteur_detecte}")
                score -= 15

    # ── 6. OCR faible : information contextuelle uniquement ──
    if confiance_moy < 0.35:
        anomalies.append(f"ℹ️ Lisibilité OCR faible ({confiance_moy*100:.0f}%) - peut indiquer usure ou photo floue")

    return {
        "score": max(0, min(score, 100)),
        "anomalies": anomalies,
        "texte_brut": texte_normalise[:600],
        "emetteur": emetteur_detecte,
        "devise": devise_detectee,
        "numero_serie": numero_serie,
        "valeur": valeur_numerique,
        "confiance_ocr": round(confiance_moy, 3),
    }


# ─────────────────────────────────────────────
# MODULE 2 : ANALYSE VISUELLE / PHYSIQUE
# ─────────────────────────────────────────────

def analyser_visuel(img: np.ndarray, usure: dict) -> dict:
    """
    Analyse les propriétés visuelles/physiques du billet.
    PRINCIPE : score de départ = 50 (présomption d'authenticité).
    Les seuils s'adaptent au niveau d'usure détecté.
    """
    anomalies = []
    score = 50  # présomption d'authenticité
    coeff_usure = usure["coefficient"]  # 0=neuf, 1=délabré

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # ── 1. Relief / Complexité ──
    # Un billet usé perd du relief → seuil abaissé proportionnellement
    relief = float(np.std(gris))
    seuil_relief_min = max(20, 40 - coeff_usure * 25)  # 40 si neuf, 15 si délabré
    if relief >= seuil_relief_min:
        score += 15
    elif relief >= seuil_relief_min * 0.6:
        score += 5
    else:
        anomalies.append(f"Relief très faible ({relief:.1f}) même pour un billet usé")
        score -= 10

    # ── 2. Densité de trame ──
    # Photo floue ou billet usé → trame moins visible → seuil adapté
    bords = cv2.Canny(gris, 60, 180)  # seuils Canny plus permissifs
    densite_trame = float(np.sum(bords > 0)) / bords.size
    seuil_trame_min = max(0.02, 0.06 - coeff_usure * 0.04)  # adaptatif
    if densite_trame >= seuil_trame_min:
        score += 15
    elif densite_trame >= seuil_trame_min * 0.5:
        score += 5
        anomalies.append(f"ℹ️ Trame faible ({densite_trame:.4f}) - cohérent avec usure niveau '{usure['niveau']}'")
    else:
        anomalies.append(f"Trame très insuffisante ({densite_trame:.4f}) - impression suspecte")
        score -= 10

    # ── 3. Couleurs (HSV) ──
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation_moy = float(np.mean(hsv[:, :, 1]))
    # Un billet usé/délavé a une saturation naturellement basse → normal
    if coeff_usure > 0.4 and saturation_moy < 50:
        score += 5  # cohérent avec l'usure déclarée
        anomalies.append(f"ℹ️ Couleurs délavées ({saturation_moy:.0f}) - cohérent avec usure")
    elif 35 < saturation_moy < 170:
        score += 12
    elif saturation_moy > 200:
        anomalies.append(f"Saturation artificielle ({saturation_moy:.0f}) - impression numérique suspectée")
        score -= 15
    else:
        score += 5  # saturation basse mais pas encore suspecte

    # ── 4. Zones lumineuses (hologramme / fil de sécurité) ──
    _, zones_lumineuses = cv2.threshold(gris, 215, 255, cv2.THRESH_BINARY)
    ratio_lumiere = float(np.sum(zones_lumineuses > 0)) / zones_lumineuses.size
    if 0.01 < ratio_lumiere < 0.30:
        score += 8
    elif ratio_lumiere > 0.6:
        anomalies.append("Image trop surexposée - zones de sécurité non lisibles")
        # Pas de pénalité forte : peut être dû à la prise de vue

    # ── 5. Ratio dimensions : INDÉPENDANT DE L'ORIENTATION ──
    # La photo peut être prise en portrait ou paysage → on normalise
    ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
    # Vrais billets CFA : ~156x66mm → ratio ~2.36
    # Vrais billets Canada : ~152x70mm → ratio ~2.17
    # On accepte 1.8 à 3.0 (photo légèrement de biais tolérée)
    if 1.7 < ratio < 3.2:
        score += 5
    else:
        anomalies.append(f"ℹ️ Ratio dimensions {ratio:.2f} - peut indiquer photo très inclinée")
        # Pas de pénalité : c'est souvent la photo, pas le billet

    # ── 6. Uniformité ── (faux imprimé = zones trop parfaites)
    variance_locale = _variance_locale(gris)
    # Un billet usé peut avoir variance élevée (froissements) → c'est normal
    if variance_locale > 150:
        score += 5  # textures présentes
    # On ne pénalise que si variance TRÈS basse ET billet supposé neuf
    elif variance_locale < 80 and coeff_usure < 0.2:
        anomalies.append(f"Uniformité suspecte ({variance_locale:.0f}) pour un billet neuf")
        score -= 8

    return {
        "score": max(0, min(score, 100)),
        "anomalies": anomalies,
        "mesures": {
            "relief": round(relief, 2),
            "densite_trame": round(densite_trame, 4),
            "saturation_moy": round(saturation_moy, 2),
            "ratio_lumiere": round(ratio_lumiere, 4),
            "variance_locale": round(variance_locale, 2),
            "ratio_dimensions": round(ratio, 3),
            "usure_coefficient": coeff_usure,
        }
    }

def _variance_locale(gris: np.ndarray, bloc=16) -> float:
    """Calcule la variance moyenne par bloc (détecte zones uniformes)."""
    variances = []
    h, w = gris.shape
    for y in range(0, h - bloc, bloc):
        for x in range(0, w - bloc, bloc):
            patch = gris[y:y+bloc, x:x+bloc]
            variances.append(float(np.var(patch)))
    return float(np.mean(variances)) if variances else 0.0


# ─────────────────────────────────────────────
# MODULE 3 : ANALYSE FRÉQUENTIELLE (FFT)
# ─────────────────────────────────────────────

def analyser_frequentiel(gris: np.ndarray, usure: dict) -> dict:
    """
    Analyse FFT : détecte la trame d'impression professionnelle.
    TOLÉRANCE : une photo compressée ou un billet usé perd des hautes fréquences.
    Score de départ : 50 (neutre).
    """
    anomalies = []
    score = 50
    coeff_usure = usure["coefficient"]

    fft = np.fft.fft2(gris)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shift))

    moy = float(np.mean(magnitude))
    max_val = float(np.max(magnitude))
    ratio_pic = max_val / moy if moy > 0 else 0

    # Seuil adapté à l'usure : billet usé/photo compressée → pics moins nets
    seuil_pic = max(2.5, 5.0 - coeff_usure * 3.0)  # 5.0 si neuf, 2.0 si délabré
    if ratio_pic > seuil_pic:
        score += 25  # trame d'impression détectée
    elif ratio_pic > seuil_pic * 0.6:
        score += 10
        anomalies.append(f"ℹ️ Trame partielle (ratio={ratio_pic:.2f}) - cohérent avec usure niveau '{usure['niveau']}'")
    else:
        # Seulement informatif, pas rédhibitoire pour un billet usé
        if coeff_usure < 0.3:
            anomalies.append(f"Spectre FFT faible (ratio={ratio_pic:.2f}) - trame d'impression non détectée")
            score -= 15
        else:
            anomalies.append(f"ℹ️ Spectre FFT faible (ratio={ratio_pic:.2f}) - normal pour billet très usé")
            score -= 5  # pénalité réduite

    # Énergie hautes fréquences
    h, w = magnitude.shape
    marge = min(h, w) // 6
    hautes_freq = magnitude.copy()
    hautes_freq[h//2-marge:h//2+marge, w//2-marge:w//2+marge] = 0
    energie_hf = float(np.mean(hautes_freq))

    if energie_hf > 4:
        score += 15
    elif energie_hf > 2:
        score += 8
    else:
        if coeff_usure < 0.4:
            score -= 5
            anomalies.append(f"Peu de micro-détails HF ({energie_hf:.2f})")

    return {
        "score": max(0, min(score, 100)),
        "anomalies": anomalies,
        "mesures": {
            "ratio_pic_fft": round(ratio_pic, 3),
            "energie_hautes_freq": round(energie_hf, 3),
            "seuil_adapte": round(seuil_pic, 2),
        }
    }


# ─────────────────────────────────────────────
# MODULE 4 : ANALYSE ANTHROPIC VISION (Claude)
# ─────────────────────────────────────────────

def analyser_avec_claude(chemin_image: str) -> dict:
    """
    Envoie le billet à Claude Vision via l'API Anthropic pour analyse experte.
    Nécessite : pip install anthropic + ANTHROPIC_API_KEY dans l'environnement.
    """
    try:
        import anthropic

        with open(chemin_image, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = chemin_image.rsplit(".", 1)[-1].lower()
        media_types = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                       "png": "image/png", "webp": "image/webp"}
        media_type = media_types.get(ext, "image/jpeg")

        client = anthropic.Anthropic()

        prompt = """Tu es un expert en détection de faux billets travaillant pour une autorité gouvernementale.
Analyse ce billet de banque et réponds UNIQUEMENT en JSON avec cette structure exacte:
{
  "verdict": "AUTHENTIQUE" ou "SUSPECT" ou "FAUX",
  "score_confiance": (0-100),
  "devise_identifiee": "...",
  "emetteur_identifie": "...",
  "valeur_nominale": ...,
  "anomalies_visuelles": ["liste des problèmes détectés"],
  "elements_securite_visibles": ["liste des éléments de sécurité observés"],
  "justification": "explication courte"
}
Cherche: fautes de frappe, couleurs incorrectes, éléments manquants, proportions suspectes, qualité d'impression."""

        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": image_data
                    }},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        texte_reponse = message.content[0].text
        # Nettoyer et parser JSON
        json_match = re.search(r'\{.*\}', texte_reponse, re.DOTALL)
        if json_match:
            import json
            resultat = json.loads(json_match.group())
            score = resultat.get("score_confiance", 50)
            if resultat.get("verdict") == "FAUX":
                score = max(0, score - 40)
            elif resultat.get("verdict") == "SUSPECT":
                score = max(0, score - 20)
            return {
                "disponible": True,
                "score": score,
                "verdict_claude": resultat.get("verdict", "INCONNU"),
                "anomalies": resultat.get("anomalies_visuelles", []),
                "elements_securite": resultat.get("elements_securite_visibles", []),
                "justification": resultat.get("justification", ""),
                "emetteur_claude": resultat.get("emetteur_identifie", ""),
                "devise_claude": resultat.get("devise_identifiee", ""),
            }

    except ImportError:
        pass
    except Exception as e:
        return {"disponible": False, "score": 50, "erreur": str(e), "anomalies": []}

    return {"disponible": False, "score": 50, "anomalies": []}


# ─────────────────────────────────────────────
# MOTEUR DE DÉCISION FINAL
# ─────────────────────────────────────────────

POIDS = {
    "texte":        0.35,   # OCR et cohérence textuelle
    "visuel":       0.30,   # Analyse physique/visuelle
    "frequentiel":  0.15,   # Analyse FFT
    "claude":       0.20,   # IA Vision (si disponible)
}

def prendre_decision(scores: dict, anomalies_totales: list) -> tuple[str, str, str]:
    """
    Calcule le score pondéré et rend le verdict final.
    Retourne (decision, niveau_confiance, recommandation)
    """
    # Score pondéré
    score_global = 0
    poids_total = 0

    for module, poids in POIDS.items():
        if module in scores and scores[module] is not None:
            score_global += scores[module] * poids
            poids_total += poids

    if poids_total > 0:
        score_global = score_global / poids_total

    score_global = int(max(0, min(score_global, 100)))

    # Règles de décision
    # Anomalie critique = faux automatique
    anomalies_critiques = [a for a in anomalies_totales if "SUSPECT" in a or "FAKE" in a or "COPY" in a or "CADAUA" in a]
    if anomalies_critiques:
        return "FAUX BILLET", "ÉLEVÉE", "Confisquer immédiatement. Anomalie critique détectée."

    if score_global >= 65:
        decision = "BILLET AUTHENTIQUE"
        confiance = "ÉLEVÉE" if score_global >= 80 else "MOYENNE"
        recommandation = "Billet probablement authentique. Vérification UV recommandée en cas de doute."
    elif score_global >= 45:
        decision = "BILLET SUSPECT"
        confiance = "FAIBLE"
        recommandation = "Soumettre à expertise physique (UV, loupe, détecteur magnétique)."
    else:
        decision = "FAUX BILLET"
        confiance = "ÉLEVÉE"
        recommandation = "Confisquer et transmettre au laboratoire forensique."

    return decision, confiance, recommandation


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def analyser_billet(chemin_image: str) -> dict:
    """Pipeline complet d'analyse d'un billet."""
    img = cv2.imread(chemin_image)
    if img is None:
        return {"erreur": "Image illisible ou format non supporté"}

    timestamp = datetime.now().isoformat()

    # ── 0. Détection image composite (recto/verso côte à côte) ──
    est_composite = detecter_image_composite(img)
    if est_composite:
        # Couper en deux et analyser la moitié supérieure (recto = plus informatif)
        h = img.shape[0]
        img_recto = img[:h//2, :]
        img_verso  = img[h//2:, :]
        # Sauvegarder temporairement pour l'OCR
        chemin_recto = chemin_image.replace(".", "_recto.")
        chemin_verso  = chemin_image.replace(".", "_verso.")
        cv2.imwrite(chemin_recto, img_recto)
        cv2.imwrite(chemin_verso, img_verso)
        # Analyser les deux faces et combiner
        note_composite = "Image composite détectée (recto+verso) - analyse des deux faces"
        img_principale = img_recto
        chemin_principal = chemin_recto
        chemin_secondaire = chemin_verso
    else:
        note_composite = None
        img_principale = img
        chemin_principal = chemin_image
        chemin_secondaire = None

    gris = cv2.cvtColor(img_principale, cv2.COLOR_BGR2GRAY)

    # ── 1. Évaluation de l'usure ──
    usure = evaluer_usure(img_principale)

    # ── 2. Lancer les 4 modules ──
    res_texte = analyser_texte(chemin_principal)

    # Si composite et émetteur non trouvé sur le recto → essayer le verso
    if est_composite and chemin_secondaire and res_texte.get("emetteur") == "Inconnu":
        res_texte_verso = analyser_texte(chemin_secondaire)
        # Fusionner : prendre le meilleur score et les infos trouvées
        if res_texte_verso.get("emetteur") != "Inconnu":
            res_texte["emetteur"] = res_texte_verso["emetteur"]
            res_texte["score"] = max(res_texte["score"], res_texte_verso["score"])
        if res_texte_verso.get("numero_serie") != "Non détecté":
            res_texte["numero_serie"] = res_texte_verso["numero_serie"]
        if res_texte_verso.get("devise") != "Inconnue":
            res_texte["devise"] = res_texte_verso["devise"]

    res_visuel = analyser_visuel(img_principale, usure)
    res_freq   = analyser_frequentiel(gris, usure)
    res_claude = analyser_avec_claude(chemin_image)  # toujours image complète pour Claude

    # ── Collecter tous les scores ──
    scores = {
        "texte":       res_texte["score"],
        "visuel":      res_visuel["score"],
        "frequentiel": res_freq["score"],
        "claude":      res_claude["score"] if res_claude.get("disponible") else None,
    }

    # ── Collecter toutes les anomalies ──
    anomalies = (
        res_texte.get("anomalies", []) +
        res_visuel.get("anomalies", []) +
        res_freq.get("anomalies", []) +
        res_claude.get("anomalies", [])
    )

    # ── Décision finale ──
    score_global = int(
        sum(v * POIDS[k] for k, v in scores.items() if v is not None) /
        sum(POIDS[k] for k, v in scores.items() if v is not None)
    )
    decision, confiance, recommandation = prendre_decision(scores, anomalies)

    # ── Choisir les meilleures métadonnées (Claude prioritaire si dispo) ──
    emetteur = res_claude.get("emetteur_claude") or res_texte.get("emetteur", "Inconnu")
    devise   = res_claude.get("devise_claude") or res_texte.get("devise", "Inconnue")

    return {
        "timestamp": timestamp,
        "decision_finale": decision,
        "score_global": score_global,
        "niveau_confiance": confiance,
        "recommandation_operationnelle": recommandation,

        "etat_billet": {
            "usure": usure["niveau"],
            "coefficient_usure": usure["coefficient"],
            "image_composite": est_composite,
            "note": note_composite or "Les seuils d'analyse sont adaptés au niveau d'usure détecté"
        },

        "identification": {
            "emetteur": emetteur,
            "devise": devise,
            "numero_serie": res_texte.get("numero_serie", "Non détecté"),
            "valeur_nominale": res_texte.get("valeur"),
        },

        "anomalies_detectees": anomalies if anomalies else ["Aucune anomalie détectée"],

        "scores_par_module": {
            "analyse_texte_ocr":    f"{res_texte['score']}%",
            "analyse_visuelle":     f"{res_visuel['score']}%",
            "analyse_frequentielle":f"{res_freq['score']}%",
            "analyse_ia_vision":    f"{res_claude['score']}%" if res_claude.get("disponible") else "Non disponible",
        },

        "details_techniques": {
            "texte":       {"confiance_ocr": res_texte.get("confiance_ocr"), **res_texte.get("mesures", {})},
            "visuel":      res_visuel.get("mesures", {}),
            "frequentiel": res_freq.get("mesures", {}),
            "claude":      {"verdict": res_claude.get("verdict_claude"), "justification": res_claude.get("justification", "")},
            "usure":       usure["details"],
        },

        "elements_securite_observes": res_claude.get("elements_securite", []),
    }


# ─────────────────────────────────────────────
# ENDPOINTS FLASK
# ─────────────────────────────────────────────

@app.route('/detect', methods=['POST'])
def detect():
    if 'photo' not in request.files:
        return jsonify({"erreur": "Fichier manquant (champ: 'photo')"}), 400

    f = request.files['photo']
    if not f.filename:
        return jsonify({"erreur": "Nom de fichier vide"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", f.filename)
    f.save(path)

    resultat = analyser_billet(path)
    return jsonify(resultat)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK", "version": "3.0-wear-tolerant", "modules": ["OCR", "Vision", "FFT", "Claude-Vision", "WearDetection"]})


@app.route('/batch', methods=['POST'])
def batch_detect():
    """Analyse plusieurs billets en une requête."""
    fichiers = request.files.getlist('photos')
    if not fichiers:
        return jsonify({"erreur": "Aucun fichier reçu"}), 400

    os.makedirs("uploads", exist_ok=True)
    resultats = []
    for f in fichiers:
        path = os.path.join("uploads", f.filename)
        f.save(path)
        res = analyser_billet(path)
        res["fichier"] = f.filename
        resultats.append(res)

    return jsonify({
        "total_analyses": len(resultats),
        "faux_detectes": sum(1 for r in resultats if "FAUX" in r.get("decision_finale", "")),
        "suspects": sum(1 for r in resultats if "SUSPECT" in r.get("decision_finale", "")),
        "resultats": resultats,
    })


if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    print("🚔 Système de Détection Faux Billets - Démarrage...")
    print("📡 Endpoints: /detect (POST), /batch (POST), /health (GET)")
    app.run(debug=False, port=5000, host='0.0.0.0')