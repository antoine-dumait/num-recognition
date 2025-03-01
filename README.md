# num-recognition
Reconnaissance de chiffres à partir de zéro.

Ce projet implémente un réseau de neurones pour la reconnaissance de chiffres manuscrits. L'entraînement est réalisé en Python utilisant seulement NumPy.<br>
Le réseau peut être exécutée en JavaScript grâce à Math.js pour la manipulation des matrices.

Un jeu de variables (poids et biais) est disponible dans les sources. Ces variables peuvent être recalculer en utilisant train.py.<br>
Précision sur les données de test: 94% (pas autant de précision en utilisant le canvas, aurait besoin de rogner le numéro d'abord).


##Architecture du Réseau de Neurones

Le réseau est constitué de trois couches :<br>
    **Couche d'entrée**: Une matrice 28x28, correspondant aux pixels de l'image d'entrée.<br>
    **Couche cachée**: 10 neurones avec un biais et activation ReLU (f(x) = max(0, x)).<br>
     **Couche de sortie**: 10 neurones avec une activation softmax pour obtenir une probabilité sur les 10 chiffres (0-9).<br>

TODO:
 - rogner et cadrer le dessin canvas pour correspondre au données d'entraînement.
