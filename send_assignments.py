#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send CaminoLabs task emails to today's four interviewers (A–D),
driven by a Google Sheet ("Daily Assignments" + "Email Addresses")
and a local starts.csv (long format with columns: date, interviewer, task, address).

Default behavior (no flags):
  - Uses today's local date.
  - Uses the default Google Sheet from the prompt.
  - Uses the default base path for starts.csv.
  - Sends from dschonho@usc.edu (after OAuth).
  - Actually sends emails (not a dry run).

Usage examples:
  python send_assignments.py               # run with defaults for today
  python send_assignments.py --dry-run     # print emails instead of sending
  python send_assignments.py --date 2025-10-30
  python send_assignments.py --sender dschonho@usc.edu
  python send_assignments.py --sheet "https://docs.google.com/spreadsheets/d/...../edit"
  python send_assignments.py --base "/Users/dschonho/Dropbox/Research/SanDiego311/code/fieldprep/outputs/incoming/daily"

OAuth:
  - On first run, a browser window will ask you to sign in as dschonho@usc.edu
    and approve the scopes. The token is cached in token.json by default.
"""

import argparse
import base64
import os
import re
import sys
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from urllib.parse import quote_plus

import pandas as pd
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ---- Scopes kept minimal ----
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

# ---- Defaults ----
DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/15w-0gLUGNW-vbP0N7ZUBqjLgd1-U8_2yAif5kYhJ0C4/edit?usp=sharing"
DEFAULT_BASE_DIR = "/Users/dschonho/Dropbox/Research/SanDiego311/code/fieldprep/outputs/incoming/daily"
DEFAULT_SENDER   = "dschonho@usc.edu"

ROLES = ["A", "B", "C", "D"]
SUPERVISOR_NAMES = {"vickie", "carlie", "yifei"}  # case-insensitive

# ---- Helpers ----
def extract_spreadsheet_id(url_or_id: str) -> str:
    """Accepts a full Google Sheets URL or a bare spreadsheet ID."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url_or_id)
    return m.group(1) if m else url_or_id.strip()

def excel_serial_to_date(n: float) -> date:
    """Google Sheets serial date to Python date (base 1899-12-30)."""
    base = date(1899, 12, 30)
    return base + timedelta(days=int(float(n)))

def parse_any_date(value) -> date | None:
    """Best-effort date parsing for the 'Date' column coming from Sheets."""
    if value is None:
        return None
    s = str(value).strip()
    # Try pandas first (handles many formats)
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.notna(dt):
        return dt.date()
    # Try Sheets/Excel serial numbers
    try:
        return excel_serial_to_date(float(s))
    except Exception:
        return None

def maps_link_from_address(addr: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(addr)}"

def normalize_name(x: str) -> str:
    return (x or "").strip().lower()

def load_credentials(credentials_path: str, token_path: str = "token.json"):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return creds

def read_sheet_as_dataframe(sheets_service, spreadsheet_id: str, range_a1: str) -> pd.DataFrame:
    resp = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=range_a1
    ).execute()
    values = resp.get("values", [])
    if not values:
        return pd.DataFrame()
    header = values[0]
    rows = values[1:]
    # Pad rows so len(row) == len(header)
    padded = [row + [""] * (len(header) - len(row)) for row in rows]
    df = pd.DataFrame(padded, columns=header)
    return df

def pick_today_row(assign_df: pd.DataFrame, target: date) -> pd.Series | None:
    if "Date" not in assign_df.columns:
        return None
    parsed = assign_df["Date"].apply(parse_any_date)
    assign_df = assign_df.copy()
    assign_df["__parsed_date"] = parsed
    matches = assign_df[assign_df["__parsed_date"] == target]
    if matches.empty:
        return None
    return matches.iloc[0]

def build_message(sender: str, to: str, subject: str, body_html: str, cc_addrs=None):
    from email.mime.text import MIMEText
    import base64
    msg = MIMEText(body_html, "html", _charset="utf-8")
    msg["to"] = to
    msg["from"] = sender  # or f"David from CaminoLabs <{sender}>"
    if cc_addrs:
        msg["cc"] = ", ".join(cc_addrs)
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    return {"raw": raw}

def get_weather_summary():
    """Fetch a one-day San Diego weather summary in Fahrenheit."""
    lat, lon, tz = 32.7157, -117.1611, "America/Los_Angeles"
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&temperature_unit=fahrenheit&precipitation_unit=inch"
        f"&timezone={tz}"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        tmax = data["daily"]["temperature_2m_max"][0]
        tmin = data["daily"]["temperature_2m_min"][0]
        rain = data["daily"]["precipitation_sum"][0]
        if rain == 0:
            rain_txt = "no rain expected"
        elif rain < 0.1:
            rain_txt = "light rain possible"
        else:
            rain_txt = "heavy rain expected"
        return f"Forecast: {tmin:.0f}–{tmax:.0f} °F, {rain_txt}."
    except Exception as e:
        return ""  # silently skip if API fails


def send_email(gmail_service, sender: str, to: str, subject: str, body_html: str, cc_addrs=None, dry_run: bool = False):
    if dry_run:
        print(f"\n[DRY RUN] To: {to}")
        if cc_addrs:
            print(f"[DRY RUN] Cc: {', '.join(cc_addrs)}")
        print(f"Subject: {subject}\n\n{body_html}\n")
        return
    gmail_service.users().messages().send(
        userId="me",
        body=build_message(sender, to, subject, body_html, cc_addrs=cc_addrs)
    ).execute()

def main():
    parser = argparse.ArgumentParser(description="Send CaminoLabs task emails to interviewers A–D for a given date.")
    parser.add_argument("--sheet", default=DEFAULT_SHEET_URL, help="Google Sheet URL or spreadsheet ID.")
    parser.add_argument("--date", default=None, help="Date to use (YYYY-MM-DD). Defaults to today (local).")
    parser.add_argument("--sender", default=DEFAULT_SENDER, help="Sender Gmail account (default: dschonho@usc.edu).")
    parser.add_argument("--base", default=DEFAULT_BASE_DIR, help="Base folder containing daily/<date>/starts.csv")
    parser.add_argument("--credentials", default="client_secret.json", help="OAuth client_secret file path.")
    parser.add_argument("--token", default="token.json", help="OAuth token cache file path.")
    parser.add_argument("--dry-run", action="store_true", help="Print emails instead of sending.")
    args = parser.parse_args()

    # Determine working date and paths
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print("ERROR: --date must be YYYY-MM-DD", file=sys.stderr)
            sys.exit(2)
    else:
        target_date = date.today()

    date_str = target_date.isoformat()
    starts_csv = os.path.join(args.base, date_str, "starts.csv")

    # Auth
    creds = load_credentials(args.credentials, args.token)
    sheets_service = build("sheets", "v4", credentials=creds)
    gmail_service = build("gmail", "v1", credentials=creds)

    # Read sheets
    ssid = extract_spreadsheet_id(args.sheet)
    daily_df = read_sheet_as_dataframe(sheets_service, ssid, "Daily Assignments!A:Z")
    emails_df = read_sheet_as_dataframe(sheets_service, ssid, "Email Addresses!A:B")

    if daily_df.empty:
        print("ERROR: 'Daily Assignments' is empty or not found.")
        sys.exit(1)
    if emails_df.empty or not set(["Name", "Email"]).issubset(set(emails_df.columns)):
        print("ERROR: 'Email Addresses' is empty or missing Name/Email columns.")
        sys.exit(1)

    row = pick_today_row(daily_df, target_date)
    if row is None:
        print(f"ERROR: No row found in 'Daily Assignments' for date {date_str}.")
        sys.exit(1)

    # Names in A-D
    name_by_role = {r: (row[r].strip() if r in row and str(row[r]).strip() else "") for r in ROLES}
    names_today_lower = { (n or "").strip().lower() for n in name_by_role.values() if n }

    # Map to emails (existing)
    emails_df["__k"] = emails_df["Name"].astype(str).str.strip().str.lower()
    name_to_email = dict(zip(emails_df["__k"], emails_df["Email"].astype(str)))

    # Supervisors to CC always (if present in Email Addresses)
    supervisor_cc = []
    for sup in SUPERVISOR_NAMES:
        e = name_to_email.get(sup)
        if e:
            supervisor_cc.append(e)


    # Load starts.csv (long format with columns: date, interviewer, task, address)
    if not os.path.exists(starts_csv):
        print(f"WARNING: starts.csv not found at {starts_csv}. Emails will show 'TBD' links.")
        starts_df = pd.DataFrame(columns=["date", "interviewer", "task", "address"])
    else:
        starts_df = pd.read_csv(starts_csv)

    # Normalize columns
    for col in ["date", "interviewer", "task", "address"]:
        if col not in starts_df.columns:
            starts_df[col] = ""
    starts_df["date"] = starts_df["date"].astype(str).str.strip()
    starts_df["interviewer"] = starts_df["interviewer"].astype(str).str.strip().str.upper()
    starts_df["task"] = starts_df["task"].astype(str).str.strip().str.upper()
    starts_df["address"] = starts_df["address"].astype(str).str.strip()

    # Email each role
    for r in ROLES:
        name = name_by_role.get(r, "")
        if not name:
            print(f"INFO: Role {r} is blank for {date_str}; skipping.")
            continue
        email = name_to_email.get(normalize_name(name))
        if not email:
            print(f"WARNING: No email found for '{name}' in 'Email Addresses'; skipping {r}.")
            continue

        # Filter starts for this date and role
        s_slice = starts_df[(starts_df["date"] == date_str) & (starts_df["interviewer"] == r)]
        dh_row = s_slice[s_slice["task"] == "DH"].head(1)
        d2_row = s_slice[s_slice["task"].isin(["D2DS", "D2D"])].head(1)
        dh_link = maps_link_from_address(dh_row["address"].iloc[0]) if not dh_row.empty and dh_row["address"].iloc[0] else "TBD"
        d2ds_link = maps_link_from_address(d2_row["address"].iloc[0]) if not d2_row.empty and d2_row["address"].iloc[0] else "TBD"

        # Build HTML body with hyperlinked addresses
        dh_text = dh_row["address"].iloc[0] if not dh_row.empty and dh_row["address"].iloc[0] else "TBD"
        d2ds_text = d2_row["address"].iloc[0] if not d2_row.empty and d2_row["address"].iloc[0] else "TBD"

        subject = f"CaminoLabs Tasks for {date_str}"
        map_url = f"https://maps-deployment.pages.dev/{r}"

        weather_line = get_weather_summary()
        body_html = f"""
        <html>
          <body style="font-family:Arial,sans-serif;">
            <p>Hello {name},</p>
            <p>The starting points of your tasks today are:</p>
            <ol>
              <li>Door hangers (DH, 9-10am): <a href="{dh_link}">{dh_text}</a></li>
              <li>Door-to-door survey (D2DS, 10am-4pm): <a href="{d2ds_link}">{d2ds_text}</a></li>
            </ol>
            <p>As a reminder, you are assigned to be Interviewer {r}, so your map URL is
                <a href="{map_url}">{map_url}</a>.</p>
            <p>{weather_line}</p>
            <p>Good luck on the road today!<br>
               David from CaminoLabs</p>
          </body>
        </html>
        """

        from email.mime.text import MIMEText
        msg = MIMEText(body_html, "html", _charset="utf-8")


        try:
            cc_list = [c for c in supervisor_cc if c.lower() != email.lower()]

            send_email(
                gmail_service,
                args.sender,
                email,
                subject,
                body_html,
                cc_addrs=cc_list,
                dry_run=args.dry_run
            )
            print(f"Sent to {name} ({email}) for role {r}.")
        except HttpError as e:
            print(f"ERROR sending to {name} ({email}) for role {r}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
