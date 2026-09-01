# DoorKam

A self-hosted security camera system in Python. Watches one or two cameras with
YOLOv11 object detection, records clips when a person (or chosen object class)
appears, and emails you a snapshot and video. Everything runs locally — no cloud
service involved.

Runs on Linux (a spare PC or Raspberry Pi) with V4L2 USB cameras.

## Features

- **Detection**: YOLOv11 with segmentation, per-camera detection zones, motion
  gating so the model only runs when something changes
- **Recording**: rolling pre-record buffer, so clips include the seconds *before*
  the trigger; stored media browser with purge controls
- **Alerts**: email notifications with snapshot and video attached (SMTP app
  password, loaded from `.env` — never committed)
- **Web dashboard** (Flask): live MJPEG streams, login required
- **Access control**: bcrypt-hashed user accounts, session auth, and rate
  limiting on the login route to slow brute-force attempts
- **Config GUI**: tkinter control panel for cameras, zones, users, and email
  settings
- **Performance**: multiprocessing pipeline; tested against Python 3.14's
  free-threaded build (`PYTHON_GIL=0`) for true parallelism

## Run it

```sh
pip install ultralytics opencv-python flask flask-limiter flask-wtf bcrypt \
            python-dotenv imageio numpy
python branch14.py
```

First run opens the config GUI — add cameras, users, and (optionally) email
credentials. The dashboard then serves on the configured port; log in with the
account you created.

`branch14.py` is the current version; `branch13.py` is the previous iteration
kept for reference.
