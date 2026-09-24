"""
The daily brief and the page. Written from the numbers by fixed sentences,
so the words can never say more than the numbers do.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import List

SITE = Path(__file__).resolve().parent.parent / "site"


def brief(read: dict) -> str:
    idx = read.get("index") or {}
    br = read.get("breadth") or {}
    vol = read.get("volatility") or {}
    risk = read.get("risk") or {}
    secs = read.get("sectors") or []
    evs = read.get("events_today") or []
    lines = []
    lines.append(f"Market read for {read.get('as_of')}: {risk.get('word', '?').upper()} "
                 f"({risk.get('score')} of {risk.get('of')} checks).")
    if idx.get("status"):
        lines.append(f"SPY is in a{'n' if idx['status'][0] in 'aeiou' else ''} {idx['status']}, "
                     f"{idx.get('pct_from_200', 0):+.1f}% from its 200-day and "
                     f"{idx.get('pct_from_high', 0):+.1f}% from its 52-week high; "
                     f"{idx.get('ret_5d', 0):+.1f}% over 5 sessions, {idx.get('ret_21d', 0):+.1f}% over 21.")
    if br.get("counted"):
        lines.append(f"Breadth is {br['status']}: {br['above_200_pct']:.0f}% of {br['counted']} "
                     f"watchlist names are above their 200-day, {br['above_50_pct']:.0f}% above "
                     f"their 50-day; {br['new_20d_highs']} made a 20-day high today, "
                     f"{br['new_20d_lows']} a 20-day low.")
    if vol.get("vix") is not None:
        lines.append(f"VIX {vol['vix']:.1f} ({vol['vix_status']}, {vol['vix_change_5d_pct']:+.0f}% "
                     f"over 5 sessions); SPY's realized volatility is {vol.get('realized_20d_pct', 0):.0f}% "
                     f"annualized and {vol.get('realized_trend', 'steady')}.")
    if secs:
        top = ", ".join(f"{s['name']} ({s['rs_21d']:+.1f}%)" for s in secs[:3])
        line = f"Leading sectors over 21 sessions versus SPY: {top}."
        if len(secs) >= 6:
            bot = ", ".join(f"{s['name']} ({s['rs_21d']:+.1f}%)" for s in secs[-3:])
            line += f" Lagging: {bot}."
        lines.append(line)
    if evs:
        top_e = evs[:5]
        bits = ", ".join(f"{e['symbol']} {e['move_pct']:+.1f}% ({e['kind']})" for e in top_e)
        lines.append(f"{len(evs)} event(s) recorded today: {bits}"
                     + (" and more." if len(evs) > 5 else "."))
    else:
        lines.append("No events recorded today: no name moved 2.5 ATR and no headline burst.")
    rec = read.get("record") or {}
    n_ev = rec.get("events_total", 0)
    lines.append(f"The record holds {n_ev} event(s)"
                 + (f"; {rec.get('with_5d', 0)} have five-day follow-through." if n_ev else "."))
    return "\n".join(lines)


def page(read: dict) -> str:
    e = html.escape
    idx = read.get("index") or {}
    br = read.get("breadth") or {}
    vol = read.get("volatility") or {}
    risk = read.get("risk") or {}
    secs = read.get("sectors") or []
    evs = read.get("events_today") or []
    says = read.get("record", {}).get("what_it_says") or []
    tone = {"risk-on": "good", "risk-off": "bad", "mixed": "warn"}.get(risk.get("word"), "warn")

    def tile(label, value, sub=""):
        return (f'<div class="tile"><div class="l">{e(label)}</div>'
                f'<div class="v">{e(str(value))}</div><div class="s">{e(sub)}</div></div>')

    checks = "".join(f'<li class="{"ok" if c["ok"] else "no"}">{e(c["check"])}</li>'
                     for c in risk.get("checks", []))
    sec_rows = "".join(
        f"<tr><td>{s['rank']}</td><td>{e(s['name'])} <span class='sym'>{e(s['symbol'])}</span></td>"
        f"<td class='n {'pos' if s['rs_21d'] >= 0 else 'neg'}'>{s['rs_21d']:+.1f}%</td>"
        f"<td class='n {'pos' if s['rs_5d'] >= 0 else 'neg'}'>{s['rs_5d']:+.1f}%</td>"
        f"<td class='n'>{s['ret_21d']:+.1f}%</td></tr>" for s in secs)
    ev_rows = "".join(
        f"<tr><td><strong>{e(v['symbol'])}</strong></td><td>{e(v['kind'])}</td>"
        f"<td class='n {'pos' if v['move_pct'] >= 0 else 'neg'}'>{v['move_pct']:+.1f}%</td>"
        f"<td class='n'>{v['move_atr']:+.1f} ATR</td><td class='n'>{v['headlines_today']} "
        f"<span class='muted'>/ usual {v['headlines_usual']:.1f}</span></td>"
        f"<td class='hl'>{e('; '.join(v.get('top_headlines') or []))}</td></tr>" for v in evs)

    def stat(s):
        if s.get("n", 0) == 0:
            return "&mdash;"
        if "note" in s:
            return f"<span class='muted'>{e(s['note'])}</span>"
        real = " &check;" if s.get("real") else ""
        return f"{s['mean_pct']:+.2f}% avg, {s['up_share_pct']:.0f}% up (n={s['n']}){real}"
    say_rows = "".join(
        f"<tr><td>{e(r['group'])}</td><td class='n'>{r['events']}</td>"
        f"<td>{stat(r.get('after_1d', {}))}</td><td>{stat(r.get('after_5d', {}))}</td></tr>"
        for r in says)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market agent</title>
<style>
:root{{--bg:#f6f7fb;--card:#fff;--text:#111827;--muted:#6b7280;--line:#e5e7eb;--good:#0a7a3a;--bad:#c62828;--warn:#a86a00}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1115;--card:#171a21;--text:#e6e8ee;--muted:#9aa3b2;--line:#262a33;--good:#3ddc84;--bad:#ff6b6b;--warn:#fab219}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:20px 16px}}
h1{{font-size:22px;margin:0}} h2{{font-size:16px;margin:24px 0 8px}}
.sub{{color:var(--muted);margin:2px 0 14px}}
.verdict{{border:1px solid var(--line);border-left:4px solid var(--{tone});background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:14px}}
.verdict b{{font-size:18px;text-transform:uppercase}}
.verdict ul{{margin:8px 0 0;padding-left:18px}} .verdict li.ok::marker{{color:var(--good)}} .verdict li.no::marker{{color:var(--bad)}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}}
.tile .l{{color:var(--muted);font-size:12px}} .tile .v{{font-size:20px;font-weight:600}} .tile .s{{color:var(--muted);font-size:11.5px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:4px 12px;overflow-x:auto}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:8px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}} .pos{{color:var(--good)}} .neg{{color:var(--bad)}}
.muted{{color:var(--muted)}} .sym{{color:var(--muted);font-size:11px}} .hl{{font-size:12.5px;color:var(--muted)}}
pre{{white-space:pre-wrap;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;font-size:13px}}
footer{{color:var(--muted);font-size:12px;margin-top:24px}}
</style></head><body><div class="wrap">
<h1>Market agent</h1>
<p class="sub">Studies the same names the trading agent can buy. Trades nothing. Read for {e(str(read.get('as_of')))}, written {e(str(read.get('updated_at')))}.</p>
<div class="verdict"><b>{e(risk.get('word', '?'))}</b> &middot; {risk.get('score')} of {risk.get('of')} checks
<ul>{checks}</ul></div>
<div class="tiles">
{tile('SPY vs 200-day', f"{idx.get('pct_from_200', 0):+.1f}%", idx.get('status', ''))}
{tile('Breadth', f"{br.get('above_200_pct', 0):.0f}%", f"of {br.get('counted', 0)} names above their 200-day ({br.get('status', '')})")}
{tile('VIX', vol.get('vix', '—'), f"{vol.get('vix_status', '')}, {vol.get('vix_change_5d_pct', 0):+.0f}% in 5 sessions")}
{tile('SPY realized vol', f"{vol.get('realized_20d_pct', 0):.0f}%", f"20-day, {vol.get('realized_trend', '')}")}
{tile('Events today', len(evs), 'big moves or headline bursts')}
</div>
<h2>Today's brief</h2>
<pre>{e(read.get('brief', ''))}</pre>
<h2>Sectors versus SPY</h2>
<div class="card"><table><thead><tr><th>#</th><th>Sector</th><th class="n">21-day vs SPY</th><th class="n">5-day vs SPY</th><th class="n">21-day return</th></tr></thead><tbody>{sec_rows}</tbody></table></div>
<h2>Events recorded today</h2>
<div class="card"><table><thead><tr><th>Symbol</th><th>Kind</th><th class="n">Move</th><th class="n">In ATRs</th><th class="n">Headlines</th><th>What the headlines said</th></tr></thead><tbody>{ev_rows or '<tr><td colspan=6 class=muted>none</td></tr>'}</tbody></table></div>
<h2>What the record says so far</h2>
<p class="sub">How prices moved after each kind of event, with the count it comes from. Under {read.get('min_sample', 20)} events a line reports the count and nothing else. A check mark means the average is distinguishable from zero.</p>
<div class="card"><table><thead><tr><th>Event kind / direction</th><th class="n">Events</th><th>Next session</th><th>Five sessions later</th></tr></thead><tbody>{say_rows or '<tr><td colspan=4 class=muted>nothing recorded yet</td></tr>'}</tbody></table></div>
<footer>Every word above is produced by a fixed rule from free daily prices and headline counts. No model reads the news and nothing here is a forecast. Source: the agent's own record, state/events.jsonl.</footer>
</div></body></html>"""


def write_site(read: dict) -> Path:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "market.json").write_text(json.dumps(read, indent=2, default=str))
    (SITE / "index.html").write_text(page(read))
    return SITE / "index.html"
