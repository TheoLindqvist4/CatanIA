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
const PLAYER_COLOURS = { 1: "#d64545", 2: "#3b6fd4", 3: "#e08b2a", 4: "#4aa564" };

const state = {
  geometry: null,
  view: null,
  gameId: null,
  mode: null,        // which board action type is armed
  busy: false,
};

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
    const size = geometry.hexWidth * 0.4;
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
    el("ellipse", {
      cx: robber.x, cy: robber.y,
      rx: geometry.hexWidth * 0.13, ry: geometry.hexWidth * 0.18,
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
      el("line", {
        x1: road.x1, y1: road.y1, x2: road.x2, y2: road.y2,
        stroke: PLAYER_COLOURS[owner] || "#fff",
        "stroke-width": geometry.hexWidth * 0.09, "stroke-linecap": "round",
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
      const size = geometry.hexWidth * (building.kind === "city" ? 0.19 : 0.13);
      el(building.kind === "city" ? "rect" : "circle",
        building.kind === "city"
          ? { x: spot.x - size, y: spot.y - size, width: size * 2, height: size * 2,
              rx: size * 0.3, fill: PLAYER_COLOURS[building.player],
              stroke: "#1c1c22", "stroke-width": 2 }
          : { cx: spot.x, cy: spot.y, r: size,
              fill: PLAYER_COLOURS[building.player],
              stroke: "#1c1c22", "stroke-width": 2 },
        svg);
    }

    const index =
      mode === "BUILD_SETTLEMENT" ? settlementTargets[spot.id]
      : mode === "BUILD_CITY" ? cityTargets[spot.id]
      : undefined;
    if (index === undefined) continue;

    el("circle", {
      cx: spot.x, cy: spot.y, r: geometry.hexWidth * 0.11,
      fill: "rgba(255,230,128,0.55)", stroke: "#fff2ad", "stroke-width": 2,
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

    let cards;
    if (player.hand) {
      const parts = Object.entries(player.hand).filter(([, n]) => n > 0)
        .map(([name, n]) => `${name} ${n}`);
      cards = parts.length ? parts.join(", ") : "no cards";
    } else {
      cards = `${player.handCount} cards`;
    }

    card.innerHTML = `
      <div class="player-head">
        <strong>${player.you ? "You" : "Opponent"}</strong>
        <span>${points}</span>
      </div>
      <div class="muted">${cards}</div>
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
    button.textContent = action.label;
    if (action.type === "END_TURN") button.className = "primary";
    button.addEventListener("click", () => play(action.index));
    holder.appendChild(button);
  }

  const view = state.view;
  document.getElementById("supply").textContent =
    Object.entries(view.bank).map(([name, n]) => `${name} ${n}`).join(" · ")
    + ` · dev deck ${view.devDeck}`;
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
