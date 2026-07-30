/* CatanIA — the browser client.
 *
 * It renders and reports clicks. It holds no rules: no legality, no board generation, no
 * scoring. Every one of those lives in Python, where it is tested — the last time this
 * project had board logic in JavaScript it was a second implementation that could disagree
 * with the engine.
 *
 * So the whole client is: fetch the static geometry once, draw whatever state the server
 * sends, and POST an action index when the player clicks something the server marked legal.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

/* Tints, used only for accents — the pieces themselves are the painted assets, whose colour
 * names arrive from the server in `geometry.art.colours` so the board and the PNG renderer
 * cannot drift apart. */
const PLAYER_COLOURS = { 1: "#d64545", 2: "#3b6fd4", 3: "#e08b2a", 4: "#4aa564" };

const state = {
  geometry: null,
  view: null,
  gameId: null,
  mode: null,        // which board action type is armed
  busy: false,
};

/** Asset file name for a player's pieces, e.g. 1 -> "red". */
const colourOf = (player) => (state.geometry.art.colours[player] || "black");

/** Sprite sizes, taken from the PNG renderer so both draw the board the same. */
const scale = (name) => state.geometry.art.scales[name];

/** Asset file name for a resource. The art set calls wheat "weat" and ore "stone". */
const resourceImage = (name) => state.geometry.art.resources[name] || name;

/** A small picture of a resource, as an <img> for the side panels. */
function resourceIcon(name, size) {
  return `<img class="res-icon" src="/images/tiles/${resourceImage(name)}.png"` +
         ` alt="${name}" title="${name}" width="${size}" height="${size}">`;
}

/** A resource count as a picture with a number on it, rather than "wood 3". */
function resourceChip(name, count, muted) {
  return `<span class="chip${muted ? " chip-empty" : ""}" title="${name}: ${count}">` +
         `${resourceIcon(name, 22)}<b>${count}</b></span>`;
}

/* ---------------------------------------------------------------- server */

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `request failed: ${response.status}`);
  return body;
}

const getJSON = (path) => api(path);
const postJSON = (path, payload) =>
  api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });

/* ----------------------------------------------------------------- game */

async function newGame() {
  const seedField = document.getElementById("seed").value;
  state.view = await postJSON("/api/game", {
    opponent: document.getElementById("opponent").value,
    rules: document.getElementById("rules").value,
    seed: seedField === "" ? null : Number(seedField),
  });
  state.gameId = state.view.gameId;
  state.mode = null;
  render();
}

async function play(index) {
  if (state.busy) return;
  state.busy = true;
  try {
    state.view = await postJSON(`/api/game/${state.gameId}/action`, { index });
    state.mode = null;
    render();
  } catch (error) {
    // The server refuses anything illegal, so this means the two disagreed — say so
    // loudly rather than leaving a click that quietly does nothing.
    setHint(`⚠ ${error.message}`);
  } finally {
    state.busy = false;
  }
}

/* --------------------------------------------------------------- drawing */

function el(name, attrs, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function boardTargets(type) {
  return (state.view && state.view.actions.board[type]) || {};
}

/** Board action types offered right now, in the order they should appear. */
function availableModes() {
  const order = ["BUILD_SETTLEMENT", "BUILD_ROAD", "BUILD_CITY", "MOVE_ROBBER"];
  return order.filter((type) => Object.keys(boardTargets(type)).length > 0);
}

function render() {
  const view = state.view;
  if (!view) return;

  // With one board action available there is nothing to choose between, so arm it
  // automatically — during setup that means you just click a spot.
  const modes = availableModes();
  if (state.mode && !modes.includes(state.mode)) state.mode = null;
  if (!state.mode && modes.length === 1) state.mode = modes[0];

  drawBoard();
  drawStatus();
  drawModes(modes);
  drawPanel();
  drawLog();
  setHint(view.phaseHint);
}

function drawBoard() {
  const { geometry, view } = state;
  const svg = document.getElementById("board");
  svg.setAttribute("viewBox", `0 0 ${geometry.width} ${geometry.height}`);
  svg.innerHTML = "";

  const tileById = Object.fromEntries(view.board.tiles.map((t) => [t.id, t]));
  const buildingAt = Object.fromEntries(view.pieces.buildings.map((b) => [b.vertex, b]));
  const roadOn = Object.fromEntries(view.pieces.roads.map((r) => [r.road, r.player]));

  el("rect", { x: 0, y: 0, width: geometry.width, height: geometry.height, fill: "#244e78" }, svg);

  // tiles, then number tokens on top
  for (const spot of geometry.tiles) {
    const tile = tileById[spot.id];
    el("image", {
      href: `/images/tiles/${tile.resource}.png`,
      x: spot.x - geometry.hexWidth / 2,
      y: spot.y - geometry.hexHeight / 2,
      width: geometry.hexWidth,
      height: geometry.hexHeight,
      "data-tile": spot.id,
    }, svg);
  }
  for (const spot of geometry.tiles) {
    const tile = tileById[spot.id];
    if (tile.number === null) continue;
    const size = geometry.hexWidth * scale("number");
    el("image", {
      href: `/images/numbers/${tile.number}.png`,
      x: spot.x - size / 2,
      y: spot.y - size / 2 + geometry.hexHeight * 0.06,
      width: size,
      height: size,
    }, svg);
  }

  drawHarbours(svg);

  // robber
  const robber = geometry.tiles.find((t) => t.id === view.robber);
  if (robber) {
    const r = geometry.hexWidth * scale("robber") / 2;
    el("ellipse", {
      cx: robber.x, cy: robber.y, rx: r * 0.85, ry: r * 1.2,
      fill: "#26262b", stroke: "#0e0e12", "stroke-width": 2,
    }, svg);
  }

  drawRoads(svg, roadOn);
  drawVertices(svg, buildingAt);
}

function drawHarbours(svg) {
  const { geometry, view } = state;
  const byId = Object.fromEntries(geometry.roads.map((r) => [r.id, r]));
  const midX = geometry.width / 2;
  const midY = geometry.height / 2;

  for (const harbour of view.board.harbours) {
    const road = byId[harbour.road];
    if (!road) continue;
    // push the marker out to sea so it does not sit on top of the land
    const dx = road.cx - midX;
    const dy = road.cy - midY;
    const span = Math.hypot(dx, dy) || 1;
    const push = geometry.hexWidth * 0.3;
    const x = road.cx + (dx / span) * push;
    const y = road.cy + (dy / span) * push;

    el("circle", {
      cx: x, cy: y, r: geometry.hexWidth * 0.155,
      fill: "#f7f0d6", stroke: "#46341c", "stroke-width": 2,
    }, svg);
    const label = el("text", {
      x, y, "text-anchor": "middle", "dominant-baseline": "central",
      "font-size": geometry.hexWidth * 0.11, fill: "#2d2109",
    }, svg);
    label.textContent = harbour.kind.replace(" ", " ");
  }
}

function drawRoads(svg, roadOn) {
  const { geometry } = state;
  const targets = boardTargets("BUILD_ROAD");
  const armed = state.mode === "BUILD_ROAD";

  for (const road of geometry.roads) {
    const owner = roadOn[road.id];
    const index = targets[road.id];

    if (owner) {
      // The road assets are painted vertically and the geometry gives the angle from
      // vertical, which is the same convention interfaces/render.py uses for the PNG.
      const length = geometry.edge * scale("roadLength");
      const width = length * (56 / 225);          // the sprites are 56x225
      el("image", {
        href: `/images/roads/${colourOf(owner)}_road.png`,
        x: road.cx - width / 2, y: road.cy - length / 2,
        width, height: length,
        transform: `rotate(${road.angle} ${road.cx} ${road.cy})`,
      }, svg);
    } else if (armed && index !== undefined) {
      el("line", {
        x1: road.x1, y1: road.y1, x2: road.x2, y2: road.y2,
        stroke: "#ffe680", "stroke-width": geometry.hexWidth * 0.05,
        "stroke-linecap": "round", "stroke-dasharray": "6 5", class: "target",
      }, svg);
    }

    if (armed && index !== undefined) {
      // a fat invisible line, so the click target is bigger than the drawn road
      const hit = el("line", {
        x1: road.x1, y1: road.y1, x2: road.x2, y2: road.y2,
        stroke: "transparent", "stroke-width": geometry.hexWidth * 0.16,
        class: "hit",
      }, svg);
      hit.addEventListener("click", () => play(index));
    }
  }
}

function drawVertices(svg, buildingAt) {
  const { geometry } = state;
  const settlementTargets = boardTargets("BUILD_SETTLEMENT");
  const cityTargets = boardTargets("BUILD_CITY");
  const mode = state.mode;

  for (const spot of geometry.vertices) {
    const building = buildingAt[spot.id];
    if (building) {
      const city = building.kind === "city";
      const colour = colourOf(building.player);
      // sprites are 226x201 (cities) and 160x133 (settlements)
      const width = geometry.hexWidth * scale(city ? "city" : "settlement");
      const height = width * (city ? 201 / 226 : 133 / 160);
      el("image", {
        href: city ? `/images/cities/${colour}_city.png`
                   : `/images/settlements/${colour}.png`,
        x: spot.x - width / 2, y: spot.y - height / 2,
        width, height,
        class: "piece",
      }, svg);
    }

    const index =
      mode === "BUILD_SETTLEMENT" ? settlementTargets[spot.id]
      : mode === "BUILD_CITY" ? cityTargets[spot.id]
      : undefined;
    if (index === undefined) continue;

    const marker = geometry.hexWidth * scale("spot");
    el("image", {
      href: "/images/spots/circle.png",
      x: spot.x - marker / 2, y: spot.y - marker / 2,
      width: marker, height: marker,
      class: "target",
    }, svg);
    const hit = el("circle", {
      cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.16,
      fill: "transparent", class: "hit",
    }, svg);
    hit.addEventListener("click", () => play(index));
  }

  if (mode === "MOVE_ROBBER") {
    const targets = boardTargets("MOVE_ROBBER");
    for (const spot of geometry.tiles) {
      const index = targets[spot.id];
      if (index === undefined) continue;
      el("circle", {
        cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.3,
        fill: "rgba(255,230,128,0.28)", stroke: "#fff2ad",
        "stroke-width": 3, class: "target",
      }, svg);
      const hit = el("circle", {
        cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.34,
        fill: "transparent", class: "hit",
      }, svg);
      hit.addEventListener("click", () => play(index));
    }
  }
}

/* ---------------------------------------------------------------- panels */

function drawStatus() {
  const view = state.view;
  document.getElementById("dice").textContent =
    view.lastRoll === null ? "–" : view.lastRoll;
  document.getElementById("turn-label").textContent = `turn ${view.turn}`;
  document.getElementById("phase-label").textContent =
    `${view.phase.toLowerCase().replace(/_/g, " ")} · first to ${view.victoryTarget}`;

  const holder = document.getElementById("players");
  holder.innerHTML = "";
  for (const player of view.players) {
    const card = document.createElement("div");
    card.className = "player" + (player.you ? " you" : "");
    card.style.borderLeftColor = PLAYER_COLOURS[player.id];

    const badges = [];
    if (player.largestArmy) badges.push("largest army");
    if (player.longestRoadHolder) badges.push("longest road");

    const points = player.victoryPoints === undefined
      ? `${player.publicVictoryPoints} public vp`
      : `${player.victoryPoints} vp`;

    // Your own hand is shown card by card; an opponent's is hidden information, so all
    // that can be drawn is the right number of card backs.
    let cards;
    if (player.hand) {
      const chips = Object.entries(player.hand)
        .map(([name, n]) => resourceChip(name, n, n === 0))
        .join("");
      cards = `<div class="hand">${chips}</div>`;
    } else {
      const backs = Array.from({ length: Math.min(player.handCount, 14) },
        () => '<span class="card-back"></span>').join("");
      cards = `<div class="hand hidden-hand">${backs}` +
              `<span class="count">${player.handCount}</span></div>`;
    }

    card.innerHTML = `
      <div class="player-head">
        <strong>${player.you ? "You" : "Opponent"}</strong>
        <span>${points}</span>
      </div>
      ${cards}
      <div class="muted">${player.devCount} dev · ${player.knights} knights ·
        road ${player.longestRoad}</div>
      <div class="muted">left: ${player.settlementsLeft}s ${player.citiesLeft}c
        ${player.roadsLeft}r</div>
      ${badges.length ? `<div class="badges">${badges.join(" · ")}</div>` : ""}
    `;
    holder.appendChild(card);
  }
}

function drawModes(modes) {
  const holder = document.getElementById("modes");
  holder.innerHTML = "";
  const labels = {
    BUILD_SETTLEMENT: "Settlement",
    BUILD_ROAD: "Road",
    BUILD_CITY: "City",
    MOVE_ROBBER: "Move robber",
  };
  if (!modes.length) {
    holder.innerHTML = '<span class="muted">nothing to place</span>';
    return;
  }
  for (const mode of modes) {
    const button = document.createElement("button");
    button.textContent = `${labels[mode]} (${Object.keys(boardTargets(mode)).length})`;
    button.className = state.mode === mode ? "active" : "";
    button.addEventListener("click", () => {
      state.mode = state.mode === mode ? null : mode;
      render();
    });
    holder.appendChild(button);
  }
}

function drawPanel() {
  const holder = document.getElementById("panel");
  holder.innerHTML = "";
  const actions = state.view.actions.panel;
  if (!actions.length) {
    holder.innerHTML = '<span class="muted">nothing available</span>';
  }
  for (const action of actions) {
    const button = document.createElement("button");
    // Illustrate whichever resources the label names, so a row of trades can be read at a
    // glance instead of parsed. Word by word rather than by regular expression: a `\b` in
    // a template literal is a backspace character, not a word boundary, and the resulting
    // pattern matches nothing while looking entirely correct.
    button.innerHTML = action.label
      .split(" ")
      .map((word) => {
        const name = word.toLowerCase();
        return state.geometry.art.resources[name]
          ? `${resourceIcon(name, 16)}${word}`
          : word;
      })
      .join(" ");
    if (action.type === "END_TURN") button.className = "primary";
    button.addEventListener("click", () => play(action.index));
    holder.appendChild(button);
  }

  const view = state.view;
  document.getElementById("supply").innerHTML =
    `<div class="hand">` +
    Object.entries(view.bank).map(([name, n]) => resourceChip(name, n, n === 0)).join("") +
    `</div><div class="muted">dev deck ${view.devDeck}</div>`;
}

function drawLog() {
  const list = document.getElementById("log");
  list.innerHTML = "";
  for (const line of state.view.log.slice().reverse()) {
    const item = document.createElement("li");
    item.textContent = line;
    if (line.startsWith("You")) item.className = "mine";
    list.appendChild(item);
  }
}

function setHint(text) {
  document.getElementById("hint").textContent = text;
}

/* ------------------------------------------------------------------ boot */

async function start() {
  state.geometry = await getJSON("/api/geometry");
  document.getElementById("new-game").addEventListener("click", newGame);
  await newGame();
}

start().catch((error) => setHint(`⚠ ${error.message}`));
