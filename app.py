from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detecter_billet():
    # Récupération des données envoyées par Postman
    donnees = request.get_json()
    
    # 1. Extraction des informations (Diagonale et État)
    dimensions = donnees.get("dimensions", {})
    diag = dimensions.get("diagonale", 0)
    
    visuel = donnees.get("analyse_visuelle", {})
    est_dechire = visuel.get("est_dechire", False)

    # 2. Logique de décision (Multi-facteurs)
    if 171.0 <= diag <= 172.0:
        if est_dechire:
            verdict = "SUSPECT"
            message = "Dimensions OK, mais le billet est déchiré. À vérifier manuellement."
        else:
            verdict = "VRAI"
            message = "Billet conforme et en bon état."
    else:
        verdict = "FAUX"
        message = f"Alerte : La diagonale ({diag}mm) est hors norme."
    
    # 3. Envoi de la réponse au Groupe 10
    return jsonify({
        "groupe": 10,
        "resultat": verdict,
        "analyse": message,
        "details_techniques": {
            "diagonale_mesuree": diag,
            "est_abime": est_dechire
        }
    }), 200

if __name__ == '__main__':
    # Lancement du serveur sur le port 5000
    app.run(debug=True, port=5000)