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
pip install -r requirements.txt
python doorkam.py
```

First run opens the config GUI — add cameras, users, and (optionally) email
credentials. The dashboard then serves on the configured port; log in with the
account you created.

## Configuration

- **SECRET_KEY**: set the `SECRET_KEY` env var for Flask sessions. If unset, a
  key is generated on first run and saved to `.secret_key` (chmod 600, kept out
  of version control).
- **User accounts**: there are no default logins. Set `USER1_EMAIL` /
  `USER1_PASSWORD` (and optionally `USER2_*`) or add users via the config GUI on
  first run — until then the web dashboard has no users and login is impossible.
- The app is deliberately a single file (`doorkam.py`): one device, one file to
  deploy.
