#!/usr/bin/env python
"""
Upload files to Google Drive

Uploads a directory to a specific Google Drive folder.
"""
import os
import sys
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload_directory_to_drive(
    local_dir,
    folder_id,
    credentials_path=None,
    credentials_json=None
):
    """
    Upload all files in a directory to a Google Drive folder.

    Parameters
    ----------
    local_dir : str or Path
        Local directory to upload
    folder_id : str
        Google Drive folder ID to upload to
    credentials_path : str, optional
        Path to service account JSON file
    credentials_json : str, optional
        Service account JSON as string

    Returns
    -------
    list
        List of uploaded file IDs
    """
    # Authenticate
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    if credentials_json:
        # Use JSON string (from GitHub secret)
        import json
        import tempfile
        creds_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
    elif credentials_path:
        # Use JSON file
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES
        )
    else:
        raise ValueError("Must provide either credentials_path or credentials_json")

    # Build Drive service
    service = build('drive', 'v3', credentials=credentials)

    # Get all files in directory
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    files_to_upload = list(local_dir.glob('*'))
    if not files_to_upload:
        print(f"Warning: No files found in {local_dir}")
        return []

    print(f"\n{'='*70}")
    print(f"Uploading to Google Drive")
    print(f"{'='*70}")
    print(f"Local directory: {local_dir}")
    print(f"Google Drive folder ID: {folder_id}")
    print(f"Files to upload: {len(files_to_upload)}")

    uploaded_ids = []

    for file_path in files_to_upload:
        if file_path.is_file():
            print(f"\nUploading: {file_path.name}...")

            file_metadata = {
                'name': file_path.name,
                'parents': [folder_id]
            }

            media = MediaFileUpload(
                str(file_path),
                resumable=True
            )

            try:
                file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, name, webViewLink'
                ).execute()

                uploaded_ids.append(file.get('id'))
                print(f"  ✓ Uploaded: {file.get('name')}")
                print(f"    File ID: {file.get('id')}")
                print(f"    Link: {file.get('webViewLink')}")

            except Exception as e:
                print(f"  ✗ Error uploading {file_path.name}: {e}")

    print(f"\n{'='*70}")
    print(f"Upload complete: {len(uploaded_ids)}/{len(files_to_upload)} files")
    print(f"{'='*70}")

    return uploaded_ids


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Upload directory to Google Drive')
    parser.add_argument('local_dir', help='Local directory to upload')
    parser.add_argument('folder_id', help='Google Drive folder ID')
    parser.add_argument(
        '--credentials',
        help='Path to service account JSON file'
    )
    parser.add_argument(
        '--credentials-json',
        help='Service account JSON as string (from env var)',
        default=os.getenv('GOOGLE_DRIVE_CREDENTIALS')
    )

    args = parser.parse_args()

    try:
        uploaded_ids = upload_directory_to_drive(
            local_dir=args.local_dir,
            folder_id=args.folder_id,
            credentials_path=args.credentials,
            credentials_json=args.credentials_json
        )

        print(f"\n✅ Successfully uploaded {len(uploaded_ids)} files to Google Drive!")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
