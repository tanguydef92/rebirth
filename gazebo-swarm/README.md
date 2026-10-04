# Essaim Gazebo — bêta, 10 drones

Module indépendant de `forfis-prototype`. Gazebo **Harmonic / Sim 8**, Python ≥3.9
et C++17. Aucun paquet pip, ROS, PX4, Docker ou modèle téléchargé à l'exécution.

Pour le guide rapide en anglais et les fichiers à modifier pour les trajectoires
et destinations, voir le [README du dépôt](../README.md#change-trajectories-and-targets).

La bêta modélise les **trajectoires cinématiques** de dix quadricoptères colorés.
Le plugin prescrit leur pose à chaque pas du temps simulé : il ne calcule pas la
poussée des moteurs, le vent, la batterie ou un autopilote. Les modèles statiques
se déplacent sur commande ; les collisions ne modifient pas les trajectoires.
Les distances minimales sont validées pour les trajets prescrits, et ne sont pas
un algorithme d'évitement d'obstacles. Aucun drone réel n'est connecté.

## Aperçu immédiat, sans installation

Depuis la racine du dépôt :

```bash
python3 gazebo-swarm/swarm.py generate
open gazebo-swarm/output/preview.html
```

L'aperçu 3D autonome fonctionne sans réseau, avec lecture/pause, curseur de temps,
vitesse ×1/×2/×4, rotation et zoom. Il utilise les mêmes échantillons CSV que le
plugin Gazebo ; ce n'est pas une capture de Gazebo.

## Installer et lancer sur macOS Apple Silicon

Les outils de compilation Apple doivent être disponibles. L'installation locale
télécharge Pixi et les paquets précompilés conda-forge dans `.tools/`, `.pixi/`
et `.cache/`, sans modifier les paquets Python de ForFiS. Le fichier `pixi.lock`
fixe les versions résolues. Les scripts de lancement activent cet environnement
automatiquement. Le script s'arrête en cas d'erreur ; aucun sudo automatique.

```bash
bash gazebo-swarm/install-local.sh
bash gazebo-swarm/run.sh
```

Autre option : `bash gazebo-swarm/install-macos.sh` installe Harmonic via Homebrew
et déclare son dépôt officiel OSRF fiable. Sur certains macOS, cette option exige
beaucoup de compilations et met à jour les dépendances Homebrew existantes.

Le lanceur démarre séparément le serveur et l'interface, conformément aux
[instructions macOS de Gazebo](https://gazebosim.org/docs/harmonic/getstarted/#macos).
Il attend le service du monde avant d'ouvrir la vue et arrête son propre serveur
quand l'interface se ferme. L'interface macOS peut être instable selon le moteur
graphique ; le lanceur utilise Ogre 2 avec Metal sur macOS. Le mode sans interface
reste disponible.
La mission commence immédiatement avec `-r` ; après 100 s les drones restent posés.
Utiliser les commandes de pause/réinitialisation de Gazebo ou relancer le script.

```bash
bash gazebo-swarm/run.sh --pattern helix
bash gazebo-swarm/run.sh --pattern sweep
bash gazebo-swarm/run.sh --headless --iterations 1500
python3 gazebo-swarm/verify_telemetry.py
```

À 0,01 s par pas, 1500 itérations couvrent environ 15 s simulées. Pour toute la
mission : `--iterations 10100`. Sans limite, interrompre avec Ctrl+C.
Ne pas lancer deux missions en même temps dans le même dossier `output`.

## Ubuntu 22.04 / 24.04

Installer Harmonic suivant la
[documentation officielle](https://gazebosim.org/docs/harmonic/install_ubuntu/),
puis `cmake`, un compilateur C++ et les bibliothèques de développement :

```bash
sudo apt install cmake g++ libgz-sim8-dev libgz-plugin2-dev
bash gazebo-swarm/build.sh
bash gazebo-swarm/run.sh
```

## Configuration et résultats

Modifier `config.json` : nombre de drones (2–50), parcours, durée, altitude,
rayon, espacement de la grille, hauteur de l'hélice, seuil de séparation,
limite de vitesse et affichage des chemins. La configuration par défaut est :

- 10 drones ; cercle de rayon 10 m ; altitude 6 m.
- Décollage 10 s, mission 80 s, atterrissage 10 s.
- 20 échantillons/s, interpolation linéaire à chaque pas Gazebo (100 Hz).
- Écart minimal requis 2 m ; vitesse maximale autorisée 5 m/s.

Les parcours utilisent une progression quintique pour des départs et arrêts doux.
`circle` fait un tour ; `helix` ajoute une montée/descente collective ; `sweep`
déplace une grille sur un parcours sinusoïdal. Tous reviennent au point de départ.
Les positions sont exprimées en mètres, le temps en secondes, le lacet en radians.

```bash
python3 gazebo-swarm/swarm.py generate --pattern sweep
python3 gazebo-swarm/swarm.py generate --config gazebo-swarm/config.json --output /tmp/ma-mission
python3 gazebo-swarm/swarm.py doctor
```

Le générateur refuse les configurations qui dépassent la vitesse ou la distance
de sécurité. La séparation est calculée analytiquement sur **chaque segment
interpolé**, pas seulement aux points échantillonnés.

`output/` contient :

- `swarm.sdf` : monde, dix modèles locaux et chemins visibles.
- `trajectories/drone_01.csv` … `drone_10.csv` : temps, x, y, z, yaw.
- `mission.json`, `report.json` : mission complète et métriques validées.
- `preview.html` : visualisation autonome.
- `telemetry.csv` : positions réellement lues dans Gazebo à 10 Hz, après lancement.
- `server.log` : journal du serveur en mode graphique.

Le SDF contient des chemins absolus vers les CSV. Régénérer après déplacement du
projet. Le lanceur le fait automatiquement. La télémétrie est réécrite à chaque
lancement ; copier les résultats pour conserver une expérience. `output/` et
`build/` sont exclus de Git. ForFiS reste indépendant ; la connexion aux zones
d'incendie et un contrôle physique des rotors sont des extensions futures.

## Vérification

```bash
python3 -m unittest discover -s gazebo-swarm/tests -v
bash gazebo-swarm/build.sh
bash gazebo-swarm/run.sh --headless --iterations 1500
```

Les tests Python couvrent les trois parcours, décollage/atterrissage, vitesse,
séparation entre échantillons, configuration et concordance SDF/CSV/aperçu.
Le test C++ couvre le lecteur CSV, l'interpolation et le retour dans le temps.
Le lancement Gazebo vérifie séparément l'intégration du plugin et la télémétrie.
`verify_telemetry.py` compare les positions enregistrées aux trajectoires et
refuse les drones absents, les données incomplètes ou les écarts supérieurs à 5 cm.
Un aperçu navigateur réussi ne valide pas à lui seul l'exécution Gazebo.

Résultats de l'exécution locale et limites graphiques observées :
[VALIDATION.md](VALIDATION.md).
