# 🏝️ CatanIA - Jeu en ligne inspiré de Catan 🎲

## 📜 Description du projet

CatanIA est une réimplémentation du jeu de société **Catan** en Python 🐍. L'objectif principal
n'est pas seulement de pouvoir y jouer : c'est d'exposer **l'intégralité de l'état de la partie**
sous une forme lisible par une machine, afin d'**entraîner une IA** à jouer. L'interface de jeu est
un *consommateur* du moteur, jamais une partie de celui-ci.

> 🚧 **Projet en cours.** Le moteur (`catan/`) gère une partie complète de bout en bout : mise en
> place, économie, construction, villes, banque, échange avec les ports, voleur et victoire à 10.
> Il manque encore les cartes de développement et les deux cartes de bonus (armée / route).
>
> - **[ROADMAP.md](ROADMAP.md)** — les phases, ce qui est fait et ce qui reste.
> - **[docs/engine.md](docs/engine.md)** — comment le moteur s'articule et comment piloter une
>   partie.
> - **[docs/](docs/README.md)** — l'audit initial, la référence de la géométrie, et les décisions
>   clés (avec leurs justifications et les écarts assumés par rapport aux règles officielles).

## ⭐ État actuel

**Fonctionne** (363 tests) :

- 🗺️ **Géométrie du plateau** : 19 tuiles, 54 emplacements de colonies, 72 emplacements de routes,
  toutes les relations d'adjacence — **générées** et vérifiées contre les schémas de `Images/`.
- 🎲 **Génération du plateau** : distribution standard des jetons et des ressources, avec une règle
  d'équilibrage (pas de nombres identiques adjacents, ni 6/8 ni 2/12).
- 🧭 **Modèle d'état complet** : propriétaire et type de chaque sommet, propriétaire de chaque
  route, mains, réserves de pièces, phase et tour. Copie (`clone`) économique pour la recherche.
- ⚖️ **Une seule autorité de légalité** : `legal_actions` et `apply` partagent les mêmes
  prédicats — un coup ne peut pas être proposé par l'un et refusé par l'autre.
- 🏠 **Construction** : routes, colonies et **villes**, avec le coût réellement débité, la règle de
  distance, la connectivité, et le blocage par une construction adverse.
- 🎲 **Production** selon les dés, doublée pour les villes.
- 🏦 **La banque** : 19 cartes par ressource, conservation des cartes (toujours 95 au total) et la
  règle officielle de pénurie.
- 🔄 **Échange avec la banque** à 4:1, et **les ports** à 3:1 et 2:1. Ports placés aléatoirement
  mais régulièrement espacés ; les deux extrémités d'un port en donnent le bénéfice, mais la règle
  de distance fait qu'un seul joueur peut en profiter.
- 🦹 **Le voleur** : gestion du 7, défausse de la moitié de la main au-delà de 7 cartes, blocage de
  la production de sa tuile, et vol d'une carte au hasard.
- 🏆 **Points de victoire et victoire à 10.**
- 🛤️ **Plus longue route** : chemin simple strict, interrompu par une construction adverse.
- 👥 **2 à 4 joueurs.**
- 🔁 **Déterminisme** : une partie est entièrement reproductible à partir d'une graine (`seed`).

**Pas encore implémenté :** les cartes de développement, l'armée la plus puissante, l'attribution
des 2 points de la plus longue route, ainsi que l'espace d'actions et d'observations destiné à
l'IA. Détails dans [ROADMAP.md](ROADMAP.md).

**Volontairement hors périmètre :** l'échange entre joueurs. La cible est le 1 contre 1, où céder
des ressources profite au seul adversaire capable de vous battre — et une offre libre est un
échange « ensemble contre ensemble » qui ne se ramène pas à un espace d'actions discret.
Voir [décision 0011](docs/decisions/0011-no-player-to-player-trading.md).

> 📈 **L'échange a rendu les parties jouables.** Avant lui, seules **4 parties sur 40** atteignaient
> 10 points : une colonie coûte quatre ressources différentes, or les colonies d'un joueur n'en
> atteignent souvent que trois, et sans conversion possible les joueurs restaient bloqués avec plus
> de 100 cartes inutilisables. Aujourd'hui : **40 parties sur 40** se terminent (393 tours en
> médiane). Détails : [docs/engine.md](docs/engine.md#trading-is-what-made-games-finishable).

## Plateau du jeu Catan

![Plateau de jeu](Images/Catan_board.png)

## Positions des routes

![Positions des routes](Images/Catan_road_positions.png)

## Positions des colonies

![Positions des colonies](Images/Catan_settlement_positions.png)

## 🛠️ Technologies utilisées

- 🐍 **Python** : Langage principal du backend

## 🚀 Installation et exécution

1. 📥 **Cloner le projet** :
   ```sh
   git clone https://github.com/votre-utilisateur/CatanIA.git
   cd CatanIA
   ```
2. 📦 **Installer les dépendances** :
   ```sh
   pip install -r requirements.txt
   ```
3. 🎮 **Piloter une partie** avec le moteur :
   ```python
   import random
   from catan.state import GameState, Phase
   from catan import rules

   state = GameState(num_players=3, seed=42)
   agent = random.Random(0)

   # La limite de tours reste prudente : une partie peut encore ne pas se terminer.
   while state.phase is not Phase.GAME_OVER and state.turn_number < 2000:
       if state.phase is Phase.ROLL:
           rules.roll_dice(state)
           continue
       actions = rules.legal_actions(state)
       if not actions:
           break
       rules.apply(state, agent.choice(actions))   # votre agent choisit ici

   print(state.winner, rules.scores(state))
   ```

   Voir [docs/engine.md](docs/engine.md). L'ancienne démo en terminal
   (`python Game_2_players.py`) fonctionne encore, mais elle est dépréciée.

## 📂 Structure du projet

```
CatanIA/
│-- 📦 catan/                 # Le moteur
│   │-- 📐 topology.py        #   Géométrie du plateau (générée, gelée à l'import)
│   │-- 🌾 resources.py       #   Les cinq ressources et les coûts
│   │-- 🗺️ board.py           #   Une disposition de plateau — IMMUABLE
│   │-- 🧭 state.py           #   GameState : tout ce qui change pendant la partie
│   │-- 🎯 actions.py         #   Action = (type, position, extra)
│   │-- ⚖️ rules.py           #   legal_actions / apply — l'unique autorité de légalité
│-- 🧪 tests/                 # Suite de tests (pytest)
│-- 📚 docs/                  # Audit, géométrie, moteur, décisions clés
│-- 🛣️ ROADMAP.md             # État du projet et phases
│-- 📄 README.md              # Documentation du projet
│
│-- 🗄️ Board.py Player.py Deck.py Dice.py Game_2_players.py
                              # Ancien moteur, DÉPRÉCIÉ. Conservé uniquement pour que
                              # `python Game_2_players.py` fonctionne encore ; supprimé
                              # en phase 4 avec interfaces/cli.py.
```

`catan/topology.py` ne contient **aucune** table écrite à la main. Toute la géométrie est
**générée** à partir d'une seule ligne :

```python
ROW_LENGTHS = (3, 4, 5, 4, 3)
```

Les centres des hexagones sont placés sur un réseau d'entiers, les sommets partagés sont fusionnés
par égalité exacte, puis les identifiants sont attribués par position : les sommets triés par
`(y, x)`, les routes par `(y minimal, x₁+x₂)`. Les identifiants obtenus sont **identiques** à ceux
dessinés dans `Images/` — les tests le vérifient, donc les schémas et le code ne peuvent pas
diverger.

Cela supprime environ 440 lignes de données maintenues à la main, rend impossible toute
désynchronisation entre les relations (deux entrées de l'ancienne table route→routes étaient
incorrectes, ce qui faussait silencieusement le calcul de la plus longue route), et accélère les
recherches d'un facteur 145. Détails et schémas : **[docs/board-geometry.md](docs/board-geometry.md)**.

## 🧪 Tests

```sh
python -m pytest tests -q
```

## 🎯 Fonctionnement du jeu

- 🏁 **Début de la partie** : Chaque joueur place ses colonies et routes initiales.
- 🔄 **Tours de jeu** : Chaque joueur lance les dés, reçoit des ressources et peut construire.

## Projet lié : Full Stack Catan

Pour plus de détails sur la génération visuelle du plateau avec une application Full Stack utilisant une API, vous pouvez consulter le dépôt suivant :

[Full Stack Catan - Génération du plateau visuellement avec l&#39;application Full Stack API](https://github.com/TheoLindqvist4/FullStackCatan)

## ✨ Auteurs

- [Théo Lindqvist](https://github.com/TheoLindqvist4) 🖊️👨‍💻
