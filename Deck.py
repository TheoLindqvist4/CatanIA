"""The bank: resource supply and the development-card deck.

Two changes from the previous version:

* the counts were *class* attributes, so every deck in the process shared them;
* the resource supply was 21 per type. Standard Catan is 19.

This is still only a container. Drawing, shuffling and play-timing rules are Phase 2
(see ROADMAP.md); nothing consumes it yet.
"""

#: Standard supply: 19 cards of each resource.
RESOURCE_SUPPLY = {
    'Ore': 19,
    'Weat': 19,
    'Sheep': 19,
    'Brick': 19,
    'Wood': 19,
}

#: Standard development-card deck, 25 cards.
DEV_CARD_SUPPLY = {
    'Knight': 14,
    'Victory Point': 5,
    'Road builder': 2,
    'Year of plenty': 2,
    'Monopoly': 2,
}


class Deck:
    def __init__(self):
        self.resources = dict(RESOURCE_SUPPLY)
        self.dev_cards = dict(DEV_CARD_SUPPLY)

    def __repr__(self):
        return (
            f"Deck(resources={sum(self.resources.values())} cards, "
            f"dev_cards={sum(self.dev_cards.values())} cards)"
        )
