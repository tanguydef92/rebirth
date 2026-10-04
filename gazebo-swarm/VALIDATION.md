# Validation de la bêta — 4 octobre 2026

Environnement local : macOS 14.6 Apple Silicon, Gazebo Sim 8.10.0,
installé avec Pixi 0.81.0 / conda-forge. Plugin compilé avec AppleClang 16.
Les versions des paquets sont enregistrées dans `pixi.lock`.

- 7 tests Python : réussis (parcours, paramètres, limites, export, télémétrie).
- Test C++ / CTest : réussi (CSV, interpolation, retour dans le temps).
- JavaScript de l'aperçu : évalué avec JavaScriptCore, DOM simulé, phases de vol
  et commandes de lecture exercées. Ce contrôle ne remplace pas un test visuel.
- Gazebo sans interface : `bash run.sh --headless --iterations 10100`, terminé
  avec succès. Les dix modèles sont suivis pendant **100,91 s simulées**.
- `python3 verify_telemetry.py --minimum-duration 100` : réussi ; erreur maximale
  affichée **0,000000 m** entre poses enregistrées et trajectoires interpolées.
- Les dix dernières positions ont une altitude de **0,3 m**, point de départ
  et d'atterrissage défini dans le modèle.

Le parcours circulaire par défaut conserve au moins 6,180298 m entre centres et
atteint au maximum 1,472617 m/s. Ces bornes concernent les trajets prescrits ;
aucun modèle aérodynamique ni contrôle moteur n'est intégré.

La vue native a été lancée avec Ogre 2 / Metal après correction d'un crash Ogre 1.
Elle émet encore des messages de plugins et matériaux Ogre ainsi que des
avertissements Qt. Son rendu visuel n'a pas été confirmé par capture : utiliser
`output/preview.html` pour l'aperçu autonome si la vue native pose problème.
Le serveur et ses déplacements sont validés indépendamment de cette vue.

La tentative Homebrew a configuré et déclaré fiable le dépôt OSRF, puis a été
arrêtée avant installation de Gazebo. L'environnement opérationnel est celui du
projet, dans `.pixi/`. Aucun fichier de ForFiS n'a été modifié.
