# 💵 API de Diagnostic Prédictif : Détection de Contrefaçons
**Solution d'Intelligence Artificielle appliquée à la Cybersécurité Monétaire**

## 🧪 Vision du Projet
Ce projet démontre l'intégration d'un modèle de Machine Learning dans une architecture micro-services. L'objectif est de transformer des données géométriques brutes en un verdict de sécurité instantané via une interface de programmation REST.

---

## 🧠 Justifications Architecturales

### 1. Pourquoi l'absence de SDK est un choix volontaire ?
Pour ce projet précis, nous avons privilégié la **performance brute** et la **simplicité d'intégration**.
- **Micro-service Stateless :** La détection est une action "One-Shot". Créer un SDK pour une seule route serait une surcouche (Overhead) inutile.
- **Interopérabilité :** En gardant une API REST standard sans SDK propriétaire, n'importe quel système (un terminal de caisse en C++, une app Android en Java ou un site web en JS) peut consommer le service avec une simple requête standard.

### 2. Pourquoi utiliser POST pour une simple lecture de données ?
Bien que nous "récupérions" un diagnostic, le **POST** a été choisi pour plusieurs raisons techniques :
- **Complexité du Payload :** Nous envoyons une demi-douzaine de paramètres décimaux. Le format JSON en POST est beaucoup plus naturel et moins sujet aux erreurs d'encodage que les paramètres d'URL du GET.
- **Standard IA :** Dans l'industrie (ex: APIs d'OpenAI ou d'Azure ML), les prédictions passent toujours par du POST car l'entrée est considérée comme une "donnée à traiter" et non comme une "ressource à lister".

### 3. Analyse de Performance avec Postman
Postman a été crucial pour valider la **latence** du service. 
- **Tests de Charge :** Nous avons simulé des envois massifs pour vérifier que le modèle ML (chargé en mémoire via Pickle) répond de manière constante en moins de 50ms.
- **Validation du Verdict :** Utilisation de collections Postman pour automatiser les tests sur des jeux de données connus (vrais billets vs faux billets).

---

## 🧬 Pipeline de Traitement du Signal
1. **Collecte :** Réception des mesures (diagonale, hauteur gauche/droite, marges sup/inf, longueur).
2. **Normalisation :** Le serveur Flask prépare les données pour le modèle.
3. **Inférence :** Calcul du score par l'algorithme de Machine Learning.
4. **Verdict :** Renvoi d'un JSON clair contenant le résultat binaire et les métadonnées de confiance.

---

## ⚙️ Stack Technologique
- **Langage :** Python 3.13+
- **Moteur API :** Flask (Framework léger et rapide)
- **ML Engine :** Scikit-Learn
- **Validation :** Postman Suite
