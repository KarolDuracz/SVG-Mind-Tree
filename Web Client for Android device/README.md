# Flask Grid Demo — Development Build

This project is a touch-driven 3×3 grid test app for Android browsers and desktop browsers. Runs on https://pypi.org/project/Flask/
<br /><br />
<h3>To test what can be encoded from these patterns into actions on graphs, etc., a simple tool collects sequences of numbers or single digits, e.g., 1, 6, 2, 45, 989—various patterns from a 3x3 grid. Statistics are collected. You can assign unique graph actions to these patterns. To check if this makes sense.</h3>

## Routes

- `/` — client board
- `/admin` — live admin overview
- `/charts` — real-time charts
- `/patterns` — sequence analysis
- `/api/stats` — live stats JSON
- `/api/charts` — chart data JSON
- `/api/patterns` — pattern data JSON
- `/api/events` — server-sent events stream
- `/latest.png` — last PNG board snapshot

## What it does

- centers a 3×3 board on the client
- records press / hold / move / release events
- records tile coordinates and sequence strings
- streams live updates to the admin pages
- keeps the PNG upload path for low-latency visual snapshots
- aggregates chart data on the server side in a background thread
- shows:
  - tile / response / hold chart
  - movement direction chart
  - release direction chart
  - live pattern sequences

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

- `http://YOUR_SERVER_IP:5000/`
- `http://YOUR_SERVER_IP:5000/admin`
- `http://YOUR_SERVER_IP:5000/charts`
- `http://YOUR_SERVER_IP:5000/patterns`

<hr>

<h3>/admin main page and /patterns - SERVER SIDE</h3>

localhost:5000/admin - here are full statistics about reaction time, communication between client and server
<br /><br />
localhost:5000/patterns - I write down sequences of numbers at the bottom, which means I write down what interests me the most, PATTERNS from the client who clicks on the tiles or moves them

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Web%20Client%20for%20Android%20device/page%20admin%20and%20patterns.png?raw=true)

<h3>/charts - SERVER SIDE</h3>

Some statistics to collect data and see what works, what doesn't, etc.

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Web%20Client%20for%20Android%20device/charts%20stats.png?raw=true)

<h3>Client - CLIENT SIDE</h3>

This is what the user sees on a device, e.g. Android, in Chrome. Presses on the tiles or moves them and it goes to the server where it is processed and collected

![dump](https://github.com/KarolDuracz/SVG-Mind-Tree/blob/main/Web%20Client%20for%20Android%20device/Screenshot_com.android.chrome.png?raw=true)

