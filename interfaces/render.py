"""Draw a :class:`~catan.state.GameState` as an image.

The engine already knows where everything is. :mod:`catan.topology` places tiles, vertices
and roads on an integer lattice, so rendering is one linear map from lattice units to
pixels — no separate layout logic, and nothing to keep in sync with the rules.

    from catan.env import CatanEnv
    from interfaces.render import render

    env = CatanEnv(); env.reset(seed=0)
    render(env.state).save("board.png")

The lattice
-----------
``x`` counts half hex-widths and ``y`` counts quarter hex-heights
(see ``docs/board-geometry.md``), so for a pointy-top hex of width ``w``::

    unit_x = w / 2
    unit_y = h / 4        where h = w * 2 / sqrt(3)

Adjacent tiles in a row are 2 lattice units apart, which is exactly one hex width, and rows
are 3 units apart, which is 3/4 of a height. Both fall out of the constants rather than
being tuned by eye.

.. note::
   The earlier FullStackCatan page stepped columns by ``tileWidth * 0.87``, which overlaps
   neighbouring tiles by 13%. Pointy-top hexes tile at exactly one full width; the assets
   confirm it, being 300x345 (a 0.869 ratio, and sqrt(3)/2 = 0.866). The lattice gets this
   right by construction.

Assets are the ones from FullStackCatan, cropped to their content.
"""

import math
import pathlib

from PIL import Image, ImageDraw

from catan.board import GENERIC_HARBOUR
from catan.resources import Resource
from catan.state import NO_OWNER, Piece
from catan.topology import (
    NUM_ROADS,
    NUM_TILES,
    NUM_VERTICES,
    ROAD_VERTICES,
    TILE_XY,
    VERTEX_XY,
)

IMAGES = pathlib.Path(__file__).parent / "static" / "images"

#: Pointy-top: width / height = sqrt(3) / 2.
WIDTH_OVER_HEIGHT = math.sqrt(3) / 2

#: Player colours, in player order. The asset set provides five.
PLAYER_COLOURS = ("red", "blue", "orange", "yellow", "black")

TILE_FILES = {
    Resource.WOOD: "wood.png",
    Resource.BRICK: "brick.png",
    Resource.SHEEP: "sheep.png",
    Resource.WHEAT: "weat.png",
    Resource.ORE: "stone.png",
    None: "desert.png",
}

HARBOUR_LABELS = {
    GENERIC_HARBOUR: "3:1",
    Resource.WOOD: "2:1 wood",
    Resource.BRICK: "2:1 brick",
    Resource.SHEEP: "2:1 sheep",
    Resource.WHEAT: "2:1 wheat",
    Resource.ORE: "2:1 ore",
}

#: Sizes as fractions of one hex width, so a different scale needs one number changed.
NUMBER_SCALE = 0.40
SETTLEMENT_SCALE = 0.40
CITY_SCALE = 0.50
SPOT_SCALE = 0.26
ROAD_LENGTH_SCALE = 0.80      # of one hex edge
ROBBER_SCALE = 0.30

#: How far below a tile's centre the number token belongs, as a fraction of one hex height.
#:
#: Not zero, and not guessed. Every resource tile has a blank panel punched clean out of the
#: art for the token to sit in — a genuine hole, alpha 0, not a light patch — and the artist
#: put it *below* the middle to leave room for the wheat sheaf or the tree above it. In the
#: 300x345 assets that hole spans y 163..274, so its centre is at 219/345 = 0.6348 of the
#: height, which is 0.135 of a height below the centre of the hex.
#:
#: Measured from the assets rather than taken by eye; the eye had it at 0.06, which put the
#: token high enough to clip the art above the panel.
#: ``tests/test_render.py::test_the_number_token_is_centred_in_the_blank_panel`` re-measures
#: them and fails if the art and this number ever disagree.
NUMBER_OFFSET = 0.135

BACKGROUND = (36, 78, 120)     # sea
_cache = {}


def _load(*parts):
    """Load an asset once and keep it. Rendering many frames should not re-read PNGs."""
    key = parts
    if key not in _cache:
        _cache[key] = Image.open(IMAGES.joinpath(*parts)).convert("RGBA")
    return _cache[key]


def _fitted(image, width):
    """``image`` scaled to ``width``, keeping its aspect ratio."""
    scale = width / image.width
    return image.resize((max(1, round(width)), max(1, round(image.height * scale))),
                        Image.LANCZOS)


def _stretched(image, width, height):
    """``image`` forced to exactly ``width`` x ``height``.

    Tiles must use this rather than :func:`_fitted`. The assets are 300x345, but a
    pointy-top hex of width 300 is 346.4 tall — keeping the asset's own ratio leaves a
    sub-pixel gap per tile, which accumulates into visible seams across the board. The
    0.4% distortion is invisible; the seams were not.
    """
    return image.resize((max(1, round(width)), max(1, round(height))), Image.LANCZOS)


def _font(size):
    """A legible font at ``size``, falling back to PIL's bitmap default.

    The default is fixed at about 11px and unreadable on a large board, so a TrueType
    face is worth looking for — but never worth failing over.
    """
    from PIL import ImageFont

    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Geometry:
    """Lattice units to pixels, and the canvas they need.

    The lattice spans x in -1..9 and y in -2..14 — corners stick out one unit beyond the
    outermost tile centres — so the origin is shifted to make everything positive.
    """

    def __init__(self, hex_width=120, margin=None):
        self.hex_width = hex_width
        self.hex_height = hex_width / WIDTH_OVER_HEIGHT
        self.unit_x = hex_width / 2
        self.unit_y = self.hex_height / 4
        self.edge = self.hex_height / 2          # one hexagon side
        # Enough room for harbour markers, which are pushed out past the coastline.
        self.margin = margin if margin is not None else hex_width * 0.75

        xs = [x for x, _ in VERTEX_XY[1:]]
        ys = [y for _, y in VERTEX_XY[1:]]
        self._min_x, self._min_y = min(xs), min(ys)
        self.width = round((max(xs) - self._min_x) * self.unit_x + 2 * self.margin)
        self.height = round((max(ys) - self._min_y) * self.unit_y + 2 * self.margin)

    @property
    def size(self):
        """``(width, height)``, matching PIL's convention."""
        return (self.width, self.height)

    def point(self, lattice):
        """A lattice ``(x, y)`` as pixel ``(x, y)``."""
        x, y = lattice
        return (
            (x - self._min_x) * self.unit_x + self.margin,
            (y - self._min_y) * self.unit_y + self.margin,
        )

    def tile(self, tile):
        return self.point(TILE_XY[tile])

    def vertex(self, vertex):
        return self.point(VERTEX_XY[vertex])

    def road(self, road):
        """``(centre, angle_degrees)`` for a road, the angle measured from vertical.

        The road assets are drawn vertically, so a road running down the screen needs no
        rotation.
        """
        first, second = (self.vertex(v) for v in ROAD_VERTICES[road])
        centre = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
        angle = math.degrees(math.atan2(second[1] - first[1], second[0] - first[0])) - 90
        return centre, angle


def _paste(canvas, sprite, centre, angle=None):
    """Paste ``sprite`` centred on ``centre``, optionally rotated."""
    if angle:
        sprite = sprite.rotate(-angle, expand=True, resample=Image.BICUBIC)
    canvas.alpha_composite(
        sprite,
        (round(centre[0] - sprite.width / 2), round(centre[1] - sprite.height / 2)),
    )


def render(state, hex_width=120, show_spots_for=None, show_labels=True):
    """Draw ``state`` and return an RGBA :class:`PIL.Image.Image`.

    Args:
        state: the game to draw.
        hex_width: pixels across one tile. Everything else scales from it.
        show_spots_for: mark where this player could legally settle. ``None`` for nobody.
        show_labels: draw harbour labels and the vertex/road id key.
    """
    geometry = Geometry(hex_width)
    canvas = Image.new("RGBA", (geometry.width, geometry.height), BACKGROUND + (255,))
    draw = ImageDraw.Draw(canvas)

    _draw_tiles(canvas, geometry, state)
    _draw_harbours(canvas, draw, geometry, state, show_labels)
    if show_spots_for is not None:
        _draw_available_spots(canvas, geometry, state, show_spots_for)
    _draw_roads(canvas, geometry, state)
    _draw_buildings(canvas, geometry, state)
    _draw_robber(draw, geometry, state)
    return canvas


def _colour_of(state, player):
    return PLAYER_COLOURS[state.player_order.index(player) % len(PLAYER_COLOURS)]


def _draw_tiles(canvas, geometry, state):
    for tile in range(1, NUM_TILES + 1):
        sprite = _stretched(_load("tiles", TILE_FILES[state.board.resource_at(tile)]),
                            geometry.hex_width, geometry.hex_height)
        _paste(canvas, sprite, geometry.tile(tile))

    for tile in range(1, NUM_TILES + 1):
        number = state.board.number_at(tile)
        if state.board.resource_at(tile) is None:
            continue          # the desert carries a 7 internally but shows no token
        token = _fitted(_load("numbers", f"{number}.png"),
                        geometry.hex_width * NUMBER_SCALE)
        centre = geometry.tile(tile)
        _paste(canvas, token,
               (centre[0], centre[1] + geometry.hex_height * NUMBER_OFFSET))


def _draw_harbours(canvas, draw, geometry, state, show_labels):
    """Harbour markers, pushed out into the sea.

    A harbour sits on a coastal road, so drawing it at the road's midpoint puts it on the
    coastline itself, half over the land. Offsetting it outward along the direction away
    from the board centre keeps the tiles readable — and is what a real board looks like,
    the harbour being on the sea frame.
    """
    board_centre = (geometry.width / 2, geometry.height / 2)
    radius = geometry.hex_width * 0.17
    font = _font(round(geometry.hex_width * 0.11))

    for road, harbour in state.board.harbours.items():
        centre, _ = geometry.road(road)
        away = (centre[0] - board_centre[0], centre[1] - board_centre[1])
        span = math.hypot(*away) or 1.0
        push = geometry.hex_width * 0.32
        centre = (centre[0] + away[0] / span * push, centre[1] + away[1] / span * push)

        draw.ellipse(
            [centre[0] - radius, centre[1] - radius,
             centre[0] + radius, centre[1] + radius],
            fill=(247, 240, 214, 240), outline=(70, 52, 28, 255), width=2,
        )
        if show_labels:
            draw.text(centre, HARBOUR_LABELS[harbour].replace(" ", "\n"),
                      fill=(45, 33, 12, 255), anchor="mm", align="center", font=font)


def _draw_available_spots(canvas, geometry, state, player):
    from catan import rules

    sprite = _fitted(_load("spots", "circle.png"), geometry.hex_width * SPOT_SCALE)
    for vertex in range(1, NUM_VERTICES + 1):
        if rules.respects_distance_rule(state, vertex) and (
            state.in_setup or rules.touches_own_road(state, player, vertex)
        ):
            _paste(canvas, sprite, geometry.vertex(vertex))


def _draw_roads(canvas, geometry, state):
    length = geometry.edge * ROAD_LENGTH_SCALE
    for road in range(1, NUM_ROADS + 1):
        owner = state.edge_owner[road]
        if owner == NO_OWNER:
            continue
        source = _load("roads", f"{_colour_of(state, owner)}_road.png")
        sprite = source.resize(
            (max(1, round(length * source.width / source.height)), max(1, round(length))),
            Image.LANCZOS,
        )
        centre, angle = geometry.road(road)
        _paste(canvas, sprite, centre, angle)


def _draw_buildings(canvas, geometry, state):
    for vertex in range(1, NUM_VERTICES + 1):
        owner = state.vertex_owner[vertex]
        if owner == NO_OWNER:
            continue
        colour = _colour_of(state, owner)
        if state.vertex_piece[vertex] is Piece.CITY:
            sprite = _fitted(_load("cities", f"{colour}_city.png"),
                             geometry.hex_width * CITY_SCALE)
        else:
            sprite = _fitted(_load("settlements", f"{colour}.png"),
                             geometry.hex_width * SETTLEMENT_SCALE)
        _paste(canvas, sprite, geometry.vertex(vertex))


def _draw_robber(draw, geometry, state):
    centre = geometry.tile(state.robber_tile)
    radius = geometry.hex_width * ROBBER_SCALE / 2
    draw.ellipse(
        [centre[0] - radius, centre[1] - radius * 1.35,
         centre[0] + radius, centre[1] + radius * 1.35],
        fill=(40, 40, 45, 230), outline=(15, 15, 18, 255), width=2,
    )


def save(state, path, **kwargs):
    """Render and write a PNG. Returns the path."""
    render(state, **kwargs).save(path)
    return path
