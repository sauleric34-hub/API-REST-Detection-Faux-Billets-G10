from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detecter_billet():
    # On récupère les données envoyées par Postman
    donnees = request.get_json()
    
    # On simule une réponse de notre "IA"
    reponse = {
        "statut": "succès",
        "message": "Données du Groupe 10 bien reçues",
        "resultat_analyse": "Authentique (Simulation)",
        "donnees_recues": donnees
    }
    
    return jsonify(reponse), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)