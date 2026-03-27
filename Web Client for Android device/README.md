# Flask Grid Demo — Development Build

This project is a touch-driven 3×3 grid test app for Android browsers and desktop browsers.

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
