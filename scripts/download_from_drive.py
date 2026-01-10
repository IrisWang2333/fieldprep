#!/usr/bin/env python
"""
Download files from Google Drive

Downloads a specific file or folder from Google Drive to local directory.
"""
import os
import sys
import io
from pathlib import Path
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def download_folder_from_drive(
    folder_id,
    date_folder_name,
    output_dir,
    credentials_path=None,
    credentials_json=None,
    oauth_credentials_json=None
):
    """
    Download all files from a date-specific subfolder in Google Drive.

    Parameters
    ----------
    folder_id : str
        Google Drive parent folder ID
    date_folder_name : str
        Name of the date subfolder (e.g., "2026-01-10")
    output_dir : str or Path
        Local directory to save downloaded files
    credentials_path : str, optional
        Path to service account JSON file
    credentials_json : str, optional
        Service account JSON as string
    oauth_credentials_json : str, optional
        OAuth credentials JSON as string (recommended for personal Drive)

    Returns
    -------
    list
        List of downloaded file paths
    """
    # Authenticate (use same scope as upload for consistency)
    SCOPES = ['https://www.googleapis.com/auth/drive']

    if oauth_credentials_json:
        # Use OAuth credentials (works with personal Google Drive)
        import json
        creds_dict = json.loads(oauth_credentials_json)
        credentials = Credentials(
            token=creds_dict['token'],
            refresh_token=creds_dict['refresh_token'],
            token_uri=creds_dict['token_uri'],
            client_id=creds_dict['client_id'],
            client_secret=creds_dict['client_secret'],
            scopes=creds_dict['scopes']
        )
    elif credentials_json:
        # Use service account JSON string
        import json
        creds_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
    elif credentials_path:
        # Use service account JSON file
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
    else:
        raise ValueError("Must provide either credentials_path, credentials_json, or oauth_credentials_json")

    # Build Drive service
    service = build('drive', 'v3', credentials=credentials)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Downloading from Google Drive")
    print(f"{'='*70}")
    print(f"Parent folder ID: {folder_id}")
    print(f"Looking for subfolder: {date_folder_name}")
    print(f"Output directory: {output_dir}")

    # Find the date-specific subfolder
    try:
        query = f"name='{date_folder_name}' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        subfolders = results.get('files', [])

        if not subfolders:
            raise FileNotFoundError(f"Subfolder '{date_folder_name}' not found in Drive folder {folder_id}")

        subfolder = subfolders[0]
        subfolder_id = subfolder.get('id')
        print(f"✓ Found subfolder: {subfolder.get('name')}")
        print(f"  Folder ID: {subfolder_id}")

    except Exception as e:
        print(f"✗ Error finding subfolder: {e}")
        raise

    # List all files in the subfolder
    try:
        query = f"'{subfolder_id}' in parents and trashed=false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType, size)'
        ).execute()

        files = results.get('files', [])

        if not files:
            print(f"Warning: No files found in subfolder '{date_folder_name}'")
            return []

        print(f"\nFiles to download: {len(files)}")

    except Exception as e:
        print(f"✗ Error listing files: {e}")
        raise

    # Download each file
    downloaded_paths = []
    failed_files = []

    for file_info in files:
        file_id = file_info['id']
        file_name = file_info['name']
        file_size = file_info.get('size', 0)

        print(f"\nDownloading: {file_name} ({file_size} bytes)...")

        output_path = output_dir / file_name

        try:
            # Download file content
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"  Progress: {progress}%", end='\r')

            # Save to file
            with open(output_path, 'wb') as f:
                f.write(fh.getvalue())

            downloaded_paths.append(output_path)
            print(f"  ✓ Downloaded: {output_path}")

        except Exception as e:
            print(f"  ✗ Error downloading {file_name}: {e}")
            failed_files.append((file_name, str(e)))

    print(f"\n{'='*70}")
    print(f"Download complete: {len(downloaded_paths)}/{len(files)} files")
    print(f"{'='*70}")

    # Raise error if any files failed
    if failed_files:
        print(f"\n❌ Failed to download {len(failed_files)} file(s):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
        if len(downloaded_paths) == 0:
            raise RuntimeError(f"All {len(files)} files failed to download")
        else:
            raise RuntimeError(f"{len(failed_files)} out of {len(files)} files failed to download")

    return downloaded_paths


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Download folder from Google Drive')
    parser.add_argument('folder_id', help='Google Drive parent folder ID')
    parser.add_argument('date', help='Date folder name (YYYY-MM-DD)')
    parser.add_argument('output_dir', help='Local directory to save files')
    parser.add_argument(
        '--credentials',
        help='Path to service account JSON file'
    )
    parser.add_argument(
        '--credentials-json',
        help='Service account JSON as string (from env var)',
        default=os.getenv('GOOGLE_DRIVE_CREDENTIALS')
    )
    parser.add_argument(
        '--oauth-credentials',
        help='OAuth credentials JSON as string (recommended for personal Drive)',
        default=os.getenv('GOOGLE_DRIVE_OAUTH_CREDENTIALS')
    )

    args = parser.parse_args()

    try:
        downloaded_paths = download_folder_from_drive(
            folder_id=args.folder_id,
            date_folder_name=args.date,
            output_dir=args.output_dir,
            credentials_path=args.credentials,
            credentials_json=args.credentials_json,
            oauth_credentials_json=args.oauth_credentials
        )

        print(f"\n✅ Successfully downloaded {len(downloaded_paths)} files from Google Drive!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
