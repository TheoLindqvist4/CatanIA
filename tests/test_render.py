"""The board renderer.

Mostly geometry: the lattice-to-pixel map is the only new logic, and it is checkable
exactly. If tiles tessellate and pieces land on their vertices, the picture is right.
"""

import math

import pytest

import catan.topology as T
from catan.board import GENERIC_HARBOUR
from catan.resources import Resource
from catan.state import Piece
from helpers import fresh, play_random_game, put_building, put_road

render_module = pytest.importorskip("interfaces.render")
Geometry = render_module.Geometry
render = render_module.render


@pytest.fixture(scope="module")
def geometry():
    return Geometry(hex_width=120)


# =========================================================================== #
# GEOMETRY                                                                    #
# =========================================================================== #

def test_a_pointy_top_hex_has_the_expected_proportions(geometry):
    assert geometry.hex_width / geometry.hex_height == pytest.approx(math.sqrt(3) / 2)
    assert geometry.edge == pytest.approx(geometry.hex_height / 2)


def test_tiles_in_a_row_sit_exactly_one_width_apart(geometry):
    """The earlier web page stepped by width * 0.87, overlapping neighbours by 13%."""
    for row, length in enumerate(T.ROW_LENGTHS):
        centres = [geometry.tile(T.tile_index(row, col)) for col in range(length)]
        for left, right in zip(centres, centres[1:]):
            assert right[0] - left[0] == pytest.approx(geometry.hex_width)
            assert right[1] == pytest.approx(left[1]), "a row must be level"


def test_rows_are_three_quarters_of_a_height_apart(geometry):
    firsts = [geometry.tile(T.tile_index(row, 0))
              for row in range(len(T.ROW_LENGTHS))]
    for upper, lower in zip(firsts, firsts[1:]):
        assert lower[1] - upper[1] == pytest.approx(geometry.hex_height * 0.75)


def test_neighbouring_rows_interlock_by_half_a_width(geometry):
    top = geometry.tile(T.tile_index(0, 0))
    below = geometry.tile(T.tile_index(1, 0))
    assert top[0] - below[0] == pytest.approx(geometry.hex_width / 2)


def test_every_road_is_exactly_one_hex_edge_long(geometry):
    """A road spans two adjacent vertices, which are one side apart. If this holds for all
    72, the lattice-to-pixel map is linear and correct."""
    for road in range(1, T.NUM_ROADS + 1):
        first, second = (geometry.vertex(v) for v in T.ROAD_VERTICES[road])
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        assert length == pytest.approx(geometry.edge, rel=1e-9)


def test_a_tile_corner_is_one_circumradius_from_its_centre(geometry):
    for tile in range(1, T.NUM_TILES + 1):
        centre = geometry.tile(tile)
        for vertex in T.TILE_VERTICES[tile]:
            corner = geometry.vertex(vertex)
            distance = math.hypot(corner[0] - centre[0], corner[1] - centre[1])
            assert distance == pytest.approx(geometry.hex_height / 2, rel=1e-9)


def test_a_road_midpoint_lies_between_its_endpoints(geometry):
    for road in range(1, T.NUM_ROADS + 1):
        centre, _ = geometry.road(road)
        first, second = (geometry.vertex(v) for v in T.ROAD_VERTICES[road])
        assert centre[0] == pytest.approx((first[0] + second[0]) / 2)
        assert centre[1] == pytest.approx((second[1] + first[1]) / 2)


def test_a_vertical_road_needs_no_rotation(geometry):
    """The road assets are drawn vertically, so the angle is measured from vertical."""
    vertical = next(
        road for road in range(1, T.NUM_ROADS + 1)
        if T.VERTEX_XY[T.ROAD_VERTICES[road][0]][0]
        == T.VERTEX_XY[T.ROAD_VERTICES[road][1]][0]
    )
    _, angle = geometry.road(vertical)
    assert angle % 180 == pytest.approx(0)


def test_everything_lands_inside_the_canvas(geometry):
    for tile in range(1, T.NUM_TILES + 1):
        x, y = geometry.tile(tile)
        assert 0 < x < geometry.width and 0 < y < geometry.height
    for vertex in range(1, T.NUM_VERTICES + 1):
        x, y = geometry.vertex(vertex)
        assert 0 < x < geometry.width and 0 < y < geometry.height


def test_the_canvas_scales_with_the_hex_width():
    small, large = Geometry(hex_width=60), Geometry(hex_width=120)
    assert large.width > small.width
    assert large.height > small.height


# =========================================================================== #
# ASSETS                                                                      #
# =========================================================================== #

def test_every_asset_the_renderer_can_ask_for_exists():
    """A missing PNG would only surface when that resource happened to be drawn."""
    images = render_module.IMAGES
    assert images.is_dir(), f"asset directory missing: {images}"

    for filename in render_module.TILE_FILES.values():
        assert (images / "tiles" / filename).exists(), filename
    for number in list(range(2, 7)) + list(range(8, 13)):
        assert (images / "numbers" / f"{number}.png").exists(), number
    for colour in render_module.PLAYER_COLOURS:
        assert (images / "settlements" / f"{colour}.png").exists(), colour
        assert (images / "cities" / f"{colour}_city.png").exists(), colour
        assert (images / "roads" / f"{colour}_road.png").exists(), colour
    assert (images / "spots" / "circle.png").exists()


def test_there_is_a_colour_for_every_player():
    from catan.state import MAX_PLAYERS
    assert len(render_module.PLAYER_COLOURS) >= MAX_PLAYERS


def test_every_resource_and_the_desert_map_to_a_tile_image():
    assert set(render_module.TILE_FILES) == set(Resource) | {None}


def test_every_harbour_kind_has_a_label():
    assert set(render_module.HARBOUR_LABELS) == set(Resource) | {GENERIC_HARBOUR}


# --------------------------------------------------------------------------- #
# The blank panel the number token goes in                                    #
# --------------------------------------------------------------------------- #

def _longest_transparent_run(alpha):
    """``(start, stop)`` of the longest run of fully transparent pixels in ``alpha``.

    A run, not a bounding box: an anti-aliased hex corner contributes a pixel or two of
    transparency at the far ends of a scanline, which a bounding box would swallow whole.
    """
    best = longest = start = run = 0
    for index, value in enumerate(bytes(alpha) + b"\xff"):
        if value == 0:
            run = run + 1 if run else 1
            if run == 1:
                start = index
            if run > longest:
                longest, best = run, start
        else:
            run = 0
    return best, best + longest


def _blank_panel(image):
    """Where the tile art leaves a hole for the number token, as fractions of the asset.

    Returns ``(centre_x, centre_y, width, height)``, all relative to the image. The panel is
    a hole punched clean through the art — alpha 0 — so it can be found rather than assumed:
    scan the centre column for it, then the row through its middle. Both of those lines run
    right across the hexagon, so the only transparency on them is the panel itself.
    """
    width, height = image.size
    alpha = image.getchannel("A")
    column = alpha.crop((width // 2, 0, width // 2 + 1, height)).tobytes()
    top, bottom = _longest_transparent_run(column)
    middle = (top + bottom) // 2
    left, right = _longest_transparent_run(
        alpha.crop((0, middle, width, middle + 1)).tobytes())
    return ((left + right) / 2 / width, (top + bottom) / 2 / height,
            (right - left) / width, (bottom - top) / height)


def test_the_number_token_is_centred_in_the_blank_panel():
    """Every resource tile has a panel cut out of the art for its number, and it is *not*
    in the middle of the hex — the sheaf or the tree is drawn above it. The offset is
    measured from the assets here, so the art and the renderer cannot drift apart. Taken by
    eye it was 0.06, which put the token high enough to clip the art above the panel."""
    from PIL import Image

    tiles = [name for resource, name in render_module.TILE_FILES.items()
             if resource is not None]          # the desert shows no token, so has no panel
    assert tiles

    for name in tiles:
        image = Image.open(render_module.IMAGES / "tiles" / name).convert("RGBA")
        centre_x, centre_y, panel_w, panel_h = _blank_panel(image)
        assert panel_w > 0.2 and panel_h > 0.2, f"no panel found in {name}"
        # Horizontally the panel is simply centred, so only the vertical offset is needed.
        assert centre_x == pytest.approx(0.5, abs=0.01), name
        assert centre_y - 0.5 == pytest.approx(render_module.NUMBER_OFFSET, abs=0.005), name


def test_the_number_token_fits_inside_the_blank_panel():
    """Centring it is only worth anything if it also sits within the panel rather than
    spilling over the art around it."""
    from PIL import Image

    tile = Image.open(render_module.IMAGES / "tiles"
                      / render_module.TILE_FILES[Resource.WHEAT]).convert("RGBA")
    _, _, panel_w, panel_h = _blank_panel(tile)
    # In hex widths and heights, which is what both renderers scale everything by.
    panel = (panel_w, panel_h * tile.height / tile.width * render_module.WIDTH_OVER_HEIGHT)

    for number in list(range(2, 7)) + list(range(8, 13)):
        token = Image.open(render_module.IMAGES / "numbers" / f"{number}.png").convert("RGBA")
        box = token.split()[3].getbbox()         # the ink, not the transparent canvas
        drawn = ((box[2] - box[0]) / token.width * render_module.NUMBER_SCALE,
                 (box[3] - box[1]) / token.height * render_module.NUMBER_SCALE)
        assert drawn[0] <= panel[0], f"{number}.png is wider than the panel"
        assert drawn[1] <= panel[1], f"{number}.png is taller than the panel"


def _robber_sprite():
    from PIL import Image

    return Image.open(render_module.IMAGES / "robber" / "robber.png").convert("RGBA")


def test_the_robber_sprite_is_the_shape_the_client_is_told():
    """The browser is served this ratio because an SVG <image> needs a width *and* a height,
    where this renderer fits by width and takes the rest from the file. Two statements of one
    shape, so the served one is measured against the art rather than typed next to it."""
    sprite = _robber_sprite()
    assert render_module.ROBBER_ASPECT == pytest.approx(sprite.width / sprite.height)


def test_the_robber_has_a_transparent_background_and_no_margin():
    """It is pasted over a finished tile, so any background it carries is a white box on the
    board. It arrived as a 1920x1080 render on flat white; the mask is *not* enclosed by the
    silhouette — its tips reach the edge of the figure — so keying every white pixel would
    have taken the mask with it. What is left has to be transparent at the corners, opaque
    where the figure is, still white where the mask is, and cropped tight to its own ink."""
    sprite = _robber_sprite()
    alpha = sprite.getchannel("A")
    width, height = sprite.size

    assert alpha.getbbox() == (0, 0, width, height), "the sprite has a transparent margin"
    for corner in ((0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)):
        assert alpha.getpixel(corner) == 0, f"{corner} is not transparent"

    assert alpha.histogram()[0] > 0, "nothing was cut away"

    # The mask: light pixels that survived. Keying on colour alone would have removed them.
    pixels = sprite.load()
    mask = sum(1 for y in range(height) for x in range(width)
               if pixels[x, y][3] > 0 and pixels[x, y][0] > 200)
    assert mask > 1000, "the robber lost its mask to the background cut"


def test_the_robber_stands_in_the_clear_band_of_its_tile():
    """ROBBER_SCALE and ROBBER_OFFSET are one decision, and this is the constraint behind it.

    Below the robber is the number token, and which number is blocked is the one thing a
    player needs from a robbed tile — the grey ellipse this replaced sat centred and covered
    the token's top third. Above it is a settlement on the tile's top corner, which covered
    the hat and the mask when the robber was measured a size larger, leaving a black blob
    with nothing robber-like about it.

    So both ends are load-bearing, and both are measured from the art rather than trusted:
    the constraint is between four files and four constants, every one of them a number
    someone may reasonably want to change.

    A city is deliberately not in here. It is bigger than the band allows and clips the top
    of the hat; see ROBBER_OFFSET for why that is the accepted end of the trade.
    """
    from PIL import Image

    over_height = render_module.WIDTH_OVER_HEIGHT          # hex widths per hex height

    def sprite_height(image, scale):
        """``image`` drawn ``scale`` hex widths across, as a fraction of one hex height."""
        return scale * image.height / image.width * over_height

    # All in fractions of one hex height, measured from the tile's centre.
    robber = sprite_height(_robber_sprite(), render_module.ROBBER_SCALE)
    head = render_module.ROBBER_OFFSET - robber / 2
    foot = render_module.ROBBER_OFFSET + robber / 2

    assert head > -0.5, "the robber overflows the top of its own tile"

    for number in list(range(2, 7)) + list(range(8, 13)):
        token = Image.open(render_module.IMAGES / "numbers" / f"{number}.png").convert("RGBA")
        ink = token.getchannel("A").getbbox()
        ink_top = ((ink[1] - token.height / 2) / token.width * render_module.NUMBER_SCALE
                   * over_height + render_module.NUMBER_OFFSET)
        assert ink_top > foot, (
            f"the robber covers the {number} token by {foot - ink_top:.4f} of a hex height")

    for colour in render_module.PLAYER_COLOURS:
        settlement = Image.open(render_module.IMAGES / "settlements" / f"{colour}.png")
        # Centred on the corner, which is half a hex height above the tile's centre.
        hangs_to = -0.5 + sprite_height(settlement, render_module.SETTLEMENT_SCALE) / 2
        assert hangs_to < head, (
            f"a {colour} settlement covers the robber's head by {hangs_to - head:.4f}")


# =========================================================================== #
# RENDERING                                                                   #
# =========================================================================== #

def test_rendering_produces_an_image_of_the_declared_size():
    state = fresh(seed=1)
    image = render(state, hex_width=80)
    expected = Geometry(hex_width=80)
    assert image.size == (expected.width, expected.height)
    assert image.mode == "RGBA"


def test_rendering_is_deterministic():
    state = fresh(seed=1)
    assert render(state, hex_width=60).tobytes() == render(state, hex_width=60).tobytes()


def test_rendering_does_not_mutate_the_state():
    state = fresh(seed=1)
    before = state.clone()
    render(state, hex_width=60)
    assert state == before


def test_an_empty_board_and_a_played_one_look_different():
    empty = fresh(seed=1)
    played = play_random_game(seed=1, max_actions=400)
    assert render(empty, hex_width=60).tobytes() != render(played, hex_width=60).tobytes()


def test_pieces_change_the_picture():
    state = fresh(seed=1)
    before = render(state, hex_width=60).tobytes()

    put_building(state, 1, 20, Piece.SETTLEMENT)
    with_settlement = render(state, hex_width=60).tobytes()
    assert with_settlement != before

    state.vertex_piece[20] = Piece.CITY
    assert render(state, hex_width=60).tobytes() != with_settlement

    put_road(state, 1, T.VERTEX_ROADS[20][0])
    assert render(state, hex_width=60).tobytes() != with_settlement


def test_moving_the_robber_changes_the_picture():
    state = fresh(seed=1)
    before = render(state, hex_width=60).tobytes()
    state.robber_tile = next(t for t in range(1, T.NUM_TILES + 1)
                             if t != state.robber_tile)
    assert render(state, hex_width=60).tobytes() != before


def test_available_spots_can_be_shown_for_a_player():
    state = fresh(seed=1)
    plain = render(state, hex_width=60).tobytes()
    marked = render(state, hex_width=60, show_spots_for=1).tobytes()
    assert marked != plain, "setup should offer spots to mark"


def test_rendering_works_at_every_stage_of_a_game():
    sizes = set()

    def check(state):
        image = render(state, hex_width=50)
        sizes.add(image.size)

    play_random_game(seed=2, max_actions=120, on_step=check)
    assert len(sizes) == 1, "the canvas size must not wander"


@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_rendering_handles_every_player_count(num_players):
    state = play_random_game(seed=1, num_players=num_players, max_actions=300)
    assert render(state, hex_width=50).size == Geometry(hex_width=50).size


def test_save_writes_a_png(tmp_path):
    state = fresh(seed=1)
    target = tmp_path / "board.png"
    assert render_module.save(state, target, hex_width=50) == target
    assert target.exists() and target.stat().st_size > 0

    from PIL import Image
    assert Image.open(target).size == Geometry(hex_width=50).size
