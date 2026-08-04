"""A page showing what a training run did, and what its openings won.

    python -m training.alphazero.dashboard            # writes and opens the page
    python -m training.alphazero.dashboard --watch    # rewrite it every 30 s

Reads ``checkpoints/alphazero/metrics.jsonl`` and, if it exists,
``checkpoints/opening_study.json``, and writes one **self-contained HTML file** — inline SVG,
inline CSS, no JavaScript, no dependency, no server. Open it, or leave it open and refresh.

**It cannot slow training down, by construction.** It is a separate process that opens two
files read-only, spends a few milliseconds, and exits. There is no shared state, no lock and
no socket. `--watch` is a sleep loop around the same few milliseconds.

A static file rather than a live server for the same reason the trainer writes JSONL rather
than printing: the run outlives any process watching it, and a record you can open tomorrow is
worth more than a socket you had to be attached to. The file can also be sent to somebody.

**What it draws, and why each one is there**

pace
    positions and games per second, per iteration. Flat is healthy; a downward drift means
    something is competing for the machine, and a cliff means a worker died.
game length
    turns per game. It moves when the *policy* changes character — shorter games usually
    mean someone learned to finish.
losses
    policy cross-entropy, value MSE, and the policy's entropy. The value loss turning
    upward is the earliest warning that a run has passed its peak; that is how the run
    recorded in decision 0023 was caught.
win rate
    against the fixed heuristic, with its Wilson interval drawn — because two readings
    whose intervals overlap have not shown anything, and the bars make that impossible to
    forget. **This measures the raw policy, not the searching agent**; see D17.
openings
    what starting positions actually won, from ``training.alphazero.study``.
"""

import argparse
import html
import json
import pathlib
import time
import webbrowser

from training.alphazero import report, study

OUT = pathlib.Path("checkpoints/dashboard.html")

#: Ink. Kept to one accent plus greys: this is a page for reading numbers off, and a chart
#: that needs a legend to say which colour is which has already failed.
INK = "#1b2733"
MUTED = "#7b8794"
ACCENT = "#2f6fd0"
WARN = "#c1543a"
GOOD = "#2e8b57"
GRID = "#dde3ea"


def _series(entries, key):
    return [(e["iteration"], e[key]) for e in entries if e.get(key) is not None]


def _line_chart(points, title, subtitle="", width=560, height=170, colour=ACCENT,
                baseline=None):
    """One SVG line chart. Returns markup, or a placeholder when there is nothing to draw."""
    if len(points) < 2:
        return (f'<figure class="chart"><figcaption>{html.escape(title)}</figcaption>'
                f'<p class="empty">not enough data yet</p></figure>')

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    if baseline is not None:
        lo_y, hi_y = min(lo_y, baseline), max(hi_y, baseline)
    if hi_y == lo_y:
        hi_y = lo_y + 1e-9
    pad_l, pad_r, pad_t, pad_b = 52, 12, 16, 26
    span_x, span_y = hi_x - lo_x or 1, hi_y - lo_y

    def sx(x):
        return pad_l + (x - lo_x) / span_x * (width - pad_l - pad_r)

    def sy(y):
        return height - pad_b - (y - lo_y) / span_y * (height - pad_t - pad_b)

    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                    for i, (x, y) in enumerate(points))
    ticks = "".join(
        f'<line x1="{pad_l}" y1="{sy(v):.1f}" x2="{width - pad_r}" y2="{sy(v):.1f}" '
        f'stroke="{GRID}"/>'
        f'<text x="{pad_l - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" '
        f'class="tick">{v:.3g}</text>'
        for v in (lo_y, (lo_y + hi_y) / 2, hi_y)
    )
    rule = ""
    if baseline is not None:
        rule = (f'<line x1="{pad_l}" y1="{sy(baseline):.1f}" x2="{width - pad_r}" '
                f'y2="{sy(baseline):.1f}" stroke="{MUTED}" stroke-dasharray="4 3"/>')
    return f"""<figure class="chart">
  <figcaption>{html.escape(title)}<span>{html.escape(subtitle)}</span></figcaption>
  <svg viewBox="0 0 {width} {height}" role="img">
    {ticks}{rule}
    <path d="{path}" fill="none" stroke="{colour}" stroke-width="2"/>
    <text x="{pad_l}" y="{height - 6}" class="tick">iter {lo_x}</text>
    <text x="{width - pad_r}" y="{height - 6}" text-anchor="end" class="tick">iter {hi_x}</text>
  </svg>
</figure>"""


def _win_rate_chart(entries, width=560, height=190):
    """Win rate with its interval, because a point estimate here is misleading."""
    checks = report.evaluations(entries)
    if not checks:
        return ('<figure class="chart"><figcaption>win rate vs the heuristic</figcaption>'
                '<p class="empty">no evaluation has run yet</p></figure>')

    pad_l, pad_r, pad_t, pad_b = 52, 12, 16, 26
    lo_x = min(c[0] for c in checks)
    hi_x = max(c[0] for c in checks) or 1
    span = hi_x - lo_x or 1

    def sx(x):
        return pad_l + (x - lo_x) / span * (width - pad_l - pad_r)

    def sy(y):
        return height - pad_b - y * (height - pad_t - pad_b)

    bars = []
    for iteration, check in checks:
        low, high = check["ci"]
        x = sx(iteration)
        bars.append(
            f'<line x1="{x:.1f}" y1="{sy(low):.1f}" x2="{x:.1f}" y2="{sy(high):.1f}" '
            f'stroke="{MUTED}" stroke-width="6" stroke-linecap="round" opacity="0.35"/>'
            f'<circle cx="{x:.1f}" cy="{sy(check["win_rate"]):.1f}" r="3.5" '
            f'fill="{GOOD if check["win_rate"] >= 0.5 else WARN}"/>')
    half = (f'<line x1="{pad_l}" y1="{sy(0.5):.1f}" x2="{width - pad_r}" y2="{sy(0.5):.1f}" '
            f'stroke="{MUTED}" stroke-dasharray="4 3"/>')
    ticks = "".join(
        f'<text x="{pad_l - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" class="tick">'
        f'{100 * v:.0f}%</text>' for v in (0.0, 0.5, 1.0))
    return f"""<figure class="chart">
  <figcaption>win rate vs the heuristic<span>the raw policy, not the searching agent — bars are the 95% interval</span></figcaption>
  <svg viewBox="0 0 {width} {height}" role="img">{ticks}{half}{"".join(bars)}</svg>
</figure>"""


def _bar_table(rows, title, note=""):
    """Win rate by band, as a table with the rate drawn in the cell."""
    if not rows:
        return ""
    body = []
    for row in rows:
        rate = row["win_rate"]
        body.append(
            f'<tr><th>{html.escape(str(row["band"]))}</th>'
            f'<td class="n">{row["games"]}</td>'
            f'<td class="bar"><span style="width:{100 * rate:.1f}%;'
            f'background:{GOOD if rate >= 0.5 else WARN}"></span>'
            f'<em>{100 * rate:.0f}%</em></td></tr>')
    return (f'<figure class="chart"><figcaption>{html.escape(title)}'
            f'<span>{html.escape(note)}</span></figcaption>'
            f'<table><thead><tr><th>band</th><th class="n">games</th>'
            f'<th>win rate</th></tr></thead><tbody>{"".join(body)}</tbody></table></figure>')


def _openings(path):
    """The opening study, if one has been run."""
    path = pathlib.Path(path)
    if not path.is_file():
        return ('<section><h2>Openings</h2><p class="empty">No study yet — run '
                '<code>python -m training.alphazero.study --games 300</code>. '
                'It plays in its own process and does not touch training.</p></section>')
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    except (json.JSONDecodeError, KeyError, OSError):
        return '<section><h2>Openings</h2><p class="empty">the study file is unreadable</p></section>'
    if not summary.get("decided"):
        return '<section><h2>Openings</h2><p class="empty">no decided games</p></section>'

    def percent(rate):
        return "—" if rate is None else f"{100 * rate:.0f}%"

    resources = "".join(
        "<tr><th>{name}</th><td class='n'>{gw}</td><td class='n'>{rw}</td>"
        "<td class='n'>{go}</td><td class='n'>{ro}</td></tr>".format(
            name=html.escape(name),
            gw=row["with"]["games"], rw=percent(row["with"]["win_rate"]),
            go=row["without"]["games"], ro=percent(row["without"]["win_rate"]),
        )
        for name, row in summary["by_resource"].items())

    harbour = summary["harbour_in_opening"]
    return f"""<section>
  <h2>Openings</h2>
  <p class="lede">{summary['decided']} decided games. Overall
    <strong>{100 * summary['win_rate']:.1f}%</strong>. Mean opening production
    <strong>{summary['mean_pips']:.3f}</strong> cards a roll across both settlements, leaving
    <strong>{summary['mean_gap_to_best']:.3f}</strong> on the table. Harbours owned per game:
    <strong>{summary['harbours_owned']:.2f}</strong>.</p>
  <div class="grid">
    {_bar_table(summary['by_pips'], 'by opening production',
                'more pips is not the same as better')}
    {_bar_table(summary['by_strategy'], 'by ore-wheat-sheep share',
                'high share = the development-card opening')}
    {_bar_table(summary['by_diversity'], 'by distinct resources',
                'how actionable the opening hand is')}
    {_bar_table([
        {'band': 'opening touches a harbour', 'games': harbour['with']['games'],
         'win_rate': harbour['with']['win_rate'] or 0.0},
        {'band': 'it does not', 'games': harbour['without']['games'],
         'win_rate': harbour['without']['win_rate'] or 0.0},
    ], 'by harbour in the opening')}
  </div>
  <figure class="chart"><figcaption>win rate with and without each resource<span>"without" means the opening produces none of it at all</span></figcaption>
    <table><thead><tr><th>resource</th><th class="n">games with</th><th class="n">win</th>
      <th class="n">games without</th><th class="n">win</th></tr></thead>
      <tbody>{resources}</tbody></table></figure>
</section>"""


def build(run_directory="checkpoints/alphazero", study_path=study.STUDY):
    entries = report.load(run_directory)
    summary = report.summarise(entries)
    stamp = time.strftime("%Y-%m-%d %H:%M")

    if not entries:
        head = '<p class="empty">No metrics yet. Start a run with <code>python -u -m training.alphazero.train --hours 3</code>.</p>'
        charts = ""
    else:
        head = f"""<p class="lede">
          <strong>{summary['iterations']}</strong> iterations ·
          <strong>{summary['games']:,}</strong> games ·
          <strong>{summary['minutes']:.0f}</strong> minutes ·
          <strong>{summary['positions_per_second']:.0f}</strong> positions/sec ·
          <strong>{100 * summary['generation_share']:.0f}%</strong> of the clock generating ·
          buffer <strong>{summary['buffer']:,}</strong></p>"""
        charts = f"""<div class="grid">
          {_line_chart(_series(entries, 'positions_per_second'), 'pace', 'positions per second')}
          {_line_chart(_series(entries, 'turns'), 'game length', 'turns per game', colour=MUTED)}
          {_line_chart(_series(entries, 'value_loss'), 'value loss',
                       'turning upward is the first sign a run has peaked', colour=WARN)}
          {_line_chart(_series(entries, 'policy_loss'), 'policy loss', '', colour=ACCENT)}
          {_line_chart(_series(entries, 'entropy'), 'policy entropy',
                       'relaxing toward the search target is normal', colour=MUTED)}
          {_win_rate_chart(entries)}
        </div>"""

    return f"""<!doctype html>
<meta charset="utf-8">
<title>CatanIA — training</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
         color: {INK}; background: #f7f9fb; margin: 0; padding: 28px; }}
  h1 {{ font-size: 20px; margin: 0 0 2px; }}
  h2 {{ font-size: 16px; margin: 30px 0 10px; }}
  .stamp {{ color: {MUTED}; font-size: 13px; margin-bottom: 18px; }}
  .lede {{ margin: 0 0 16px; }}
  .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }}
  .chart {{ background: #fff; border: 1px solid {GRID}; border-radius: 8px;
            margin: 0; padding: 12px 14px 6px; overflow-x: auto; }}
  figcaption {{ font-weight: 600; font-size: 13px; margin-bottom: 6px; }}
  figcaption span {{ display: block; font-weight: 400; color: {MUTED}; font-size: 12px; }}
  svg {{ width: 100%; height: auto; }}
  .tick {{ font-size: 10px; fill: {MUTED}; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ text-align: left; padding: 3px 6px; border-bottom: 1px solid {GRID}; }}
  td.n, th.n {{ text-align: right; width: 4.5em; color: {MUTED}; }}
  td.bar {{ position: relative; min-width: 130px; }}
  td.bar span {{ display: inline-block; height: 11px; border-radius: 3px; opacity: .35; }}
  td.bar em {{ position: absolute; right: 6px; top: 3px; font-style: normal; font-size: 12px; }}
  .empty {{ color: {MUTED}; }}
  code {{ background: {GRID}; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #10151b; color: #e6edf3; }}
    .chart {{ background: #171d25; border-color: #26303b; }}
    code {{ background: #26303b; }}
    th, td {{ border-color: #26303b; }}
  }}
</style>
<h1>CatanIA — training</h1>
<div class="stamp">{run_directory} · generated {stamp} · this page is a static file and
  does not touch the run</div>
{head}
{charts}
{_openings(study_path)}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="A page showing what a training run did",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", default="checkpoints/alphazero")
    parser.add_argument("--study", default=str(study.STUDY))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--watch", type=int, nargs="?", const=30, default=None,
                        metavar="SECONDS", help="rewrite the page on a timer")
    parser.add_argument("--no-open", action="store_true")
    arguments = parser.parse_args(argv)

    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def write():
        out.write_text(build(arguments.run, arguments.study), encoding="utf-8")
        return out

    write()
    print(f"wrote {out.resolve()}")
    if not arguments.no_open:
        webbrowser.open(out.resolve().as_uri())
    if arguments.watch:
        print(f"rewriting every {arguments.watch}s — refresh the page. Ctrl-C to stop.")
        try:
            while True:
                time.sleep(arguments.watch)
                write()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
