#!/usr/bin/env python3
"""
Download historical plan CSV files from Google Drive.

This ensures GitHub Actions workflows can track previously used bundles
for without-replacement sampling.
"""
import argparse
import json
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io


def download_plans_from_drive(folder_id: str, output_dir: Path, credentials_json: str):
    """
    Download all plan CSV files from Google Drive folder.

    Parameters
    ----------
    folder_id : str
        Google Drive folder ID containing plan CSVs
    output_dir : Path
        Local directory to save downloaded files
    credentials_json : str
        OAuth credentials JSON string
    """
    # Parse credentials
    creds_dict = json.loads(credentials_json)
    creds = Credentials.from_authorized_user_info(creds_dict)

    # Build Drive API client
    service = build('drive', 'v3', credentials=creds)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # List all files in the folder
    print(f"Listing files in Google Drive folder {folder_id}...")

    query = f"'{folder_id}' in parents and mimeType='text/csv' and name contains 'bundles_plan_' and trashed=false"
    results = service.files().list(
        q=query,
        fields="files(id, name, modifiedTime)",
        orderBy="name"
    ).execute()

    files = results.get('files', [])

    if not files:
        print("No plan CSV files found in Google Drive folder")
        print("This might be the first workflow run")
        return

    print(f"Found {len(files)} plan CSV files")

    # Download each file
    for file_info in files:
        file_id = file_info['id']
        file_name = file_info['name']

        print(f"Downloading {file_name}...")

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        # Write to local file
        output_path = output_dir / file_name
        with open(output_path, 'wb') as f:
            f.write(fh.getvalue())

        print(f"  → Saved to {output_path}")

    print(f"\n✓ Downloaded {len(files)} plan files")


def main():
    parser = argparse.ArgumentParser(description="Download historical plan CSVs from Google Drive")
    parser.add_argument("folder_id", help="Google Drive folder ID")
    parser.add_argument("output_dir", help="Local directory to save files")
    parser.add_argument("--oauth-credentials", required=True, help="OAuth credentials JSON")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        download_plans_from_drive(
            folder_id=args.folder_id,
            output_dir=output_dir,
            credentials_json=args.oauth_credentials
        )
    except Exception as e:
        print(f"Error downloading plans: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
