# Validation de la bêta — 6 octobre 2026

Environnement local : macOS 14.6 Apple Silicon, Gazebo Sim 8.10.0,
installé avec Pixi 0.81.0 / conda-forge. Plugin compilé avec AppleClang 16.
Les versions des paquets sont enregistrées dans `pixi.lock`.

## PPO sur 100 drones individuels et déploiement live sur 1 000

- Séparation du code en `training/`, `simulation/`, `drones/` et `fire/` ;
  anciens modules de compatibilité retirés ; seuls les scripts exécutables restent.
  Les configurations rapides partagent exactement le modèle thermique,
  les réservoirs de 20 L, remplissages de 10 s et largages immédiats avec
  un maintien de 1 s. Seuls la flotte et le choix de l'acteur changent.
- **30 tests de simulation et cinq tests PPO réussis**, contrôlés dans leurs
  environnements respectifs.
  Nouveaux contrats : vraie cinématique individuelle dans Gymnasium,
  mise à jour PPO, observation 87 dimensions, compatibilité 100 → 1 000,
  refus d'un service incompatible, directions égales aux déplacements,
  maintien du largage, durée exacte du remplissage, clones thermiques
  indépendants, traversées entre échantillons et doses hors périmètre refusées.
- Entraînement **8 192 pas**, quatre processus de **100 drones**, sélection
  sur 301–302, tests réservés 9001–9003. Le modèle final sélectionné est
  `models/ppo-100-fast.json`, réseau 87 → 64 → 64 → 16. Les chemins NumPy
  et bibliothèque standard concordent avec le checkpoint PyTorch choisi
  sur douze observations : erreur maximale **4,39 × 10⁻⁸**.
- Résultats individuels à 100 drones, épisodes de 240 s après le décollage :

  | Politique | Retour moyen ↑ | Écart-type | Nouvelle surface brûlée moyenne ↓ |
  | --- | ---: | ---: | ---: |
  | PPO | −257,769 | 21,885 | 21 463 m² |
  | Uniforme | −257,936 | 18,502 | 21 551 m² |
  | Aléatoire concentrée | −255,508 | 18,069 | 21 199 m² |
  | Sans eau | −257,763 | 19,644 | 21 561 m² |

  L'aléatoire est meilleur dans ce court essai ; aucune supériorité de PPO
  n'est démontrée. Ces résultats remplacent le proxy de débit pour la
  nouvelle version. Ils ne sont pas directement comparables aux anciens
  scores du feu probabiliste.
- Déploiement avec le même acteur : **1 000 drones**, graine 42, mission
  complète de **690 s**, décisions courantes toutes les secondes, navigation
  à 2 Hz. **2 234 598 paires candidates** vérifiées analytiquement entre
  échantillons ; séparation certifiée **≥5 m**, avec tolérance numérique.
  Vitesse maximale **10,440307 m/s** ; les 1 000 drones atterrissent chez eux.
  Contrôle indépendant des trajets sur tout le volume potentiel du feu réussi.
- **1 430 largages de 20 L**, tous sur le périmètre observé au moment du
  largage. **1 077 remplissages terminés de 10 s chacun**.
  Bilan : **20 000 L initiaux + 21 776 L remplis = 28 600 L largués
  + 13 176 L restants**. **5 952 m cumulés** de frontière arrosée,
  soit 3,99 % des arêtes observées ; **5 743 cellules actives** à la fin.
  Nouvelle surface brûlée : **72 364 m²** ; le feu n'est pas annihilé.
  Bilan thermique relatif final : **1,46 × 10⁻¹⁶**.
- Premier run live complet : **114,69 s réelles**, validation comprise,
  soit **6,02× temps réel** pour une cible ×8. Le benchmark de 30 pas
  passe de 2,75 s à 1,57 s après vectorisation des observations.
  La résolution et les sous-pas thermiques de 0,25 s sont conservés.
- Vue réelle contrôlée dans un Chrome isolé : flux live de 1 000 drones,
  **967 directions horizontales non nulles à 153,5 s**, inspection du drone
  sélectionné, cible, allocation, couches feu/température/combustible,
  pause et redémarrage. Capture `output/live-1000-fast/viewer.png`
  inspectée. La vue est en plan et affiche l'altitude dans l'inspecteur.
  Les états ne sont pas des trajectoires préenregistrées.
- Sorties : acteur et évaluation dans `models/`, checkpoints complets dans
  `output/ppo-100-fast-training/`, traces et contrôles indépendants dans
  `output/live-1000-fast/` (`validated-run.json`, `trajectories.npz`,
  `refill-events.json`, `water-drops.json`, `viewer-check.json`). Le serveur
  peut rejouer la mission ; son PID et son URL figurent dans `endpoint.json`.

## Suppression du code obsolète

Les wrappers de compatibilité, ancien feu probabiliste, approximation de débit
pour l'apprentissage, navigation A*, parcours classiques, anciennes configurations
et anciens acteurs ont été retirés. Les outils de géométrie sont dans
`drones/geometry.py`. Gazebo utilise désormais `simulation/engine.py` pour
produire ses exports. Les tests portent sur les flux actuels ; les anciens
scénarios et leurs comptes de tests ne sont plus présentés comme actifs.

La suite actuelle comporte 30 contrôles de simulation et cinq de PPO dans
l'environnement isolé. La construction C++ et CTest, ainsi qu'un lancement
Gazebo sans interface et sa télémétrie, vérifient le lecteur conservé.
Le rejeu complet de 1 000 drones compare chaque position au run enregistré
avant suppression du code obsolète. Les résultats locaux sont dans
`output/legacy-cleanup-validation/`.

- Le rejeu de 690 s conserve **exactement** les positions enregistrées :
  **1 380 000 segments**, différence maximale **0 m**. Les bilans d'eau et
  la clairance du domaine complet passent ; les 1 000 drones atterrissent.
- Gazebo sans interface : huit drones, neuf tuiles de feu et huit flux d'eau
  contrôlés pendant **14,91 s simulées**, erreur maximale **0,000000 m**.
  Le plugin C++ est reconstruit et CTest passe.
