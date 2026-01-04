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
    credentials_json=None,
    oauth_credentials_json=None
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
    oauth_credentials_json : str, optional
        OAuth credentials JSON as string (recommended for personal Drive)

    Returns
    -------
    list
        List of uploaded file IDs
    """
    # Authenticate
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    if oauth_credentials_json:
        # Use OAuth credentials (works with personal Google Drive)
        import json
        from google.oauth2.credentials import Credentials

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
        # Use service account JSON string (from GitHub secret)
        # Note: Service accounts cannot upload to personal Drive folders
        import json
        import tempfile
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

    # Get all files in directory
    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(f"Directory not found: {local_dir}")

    files_to_upload = list(local_dir.glob('*'))
    if not files_to_upload:
        print(f"Warning: No files found in {local_dir}")
        return []

    # Extract date folder name from local directory path (e.g., "2026-01-03")
    date_folder_name = local_dir.name

    print(f"\n{'='*70}")
    print(f"Uploading to Google Drive")
    print(f"{'='*70}")
    print(f"Local directory: {local_dir}")
    print(f"Parent folder ID: {folder_id}")
    print(f"Checking for subfolder: {date_folder_name}")

    # Check if subfolder already exists
    try:
        query = f"name='{date_folder_name}' and '{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, webViewLink)'
        ).execute()

        existing_folders = results.get('files', [])

        if existing_folders:
            # Use existing folder
            subfolder = existing_folders[0]
            subfolder_id = subfolder.get('id')
            print(f"✓ Found existing subfolder: {subfolder.get('name')}")
            print(f"  Folder ID: {subfolder_id}")
            print(f"  Link: {subfolder.get('webViewLink')}")
        else:
            # Create new subfolder
            subfolder_metadata = {
                'name': date_folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folder_id]
            }

            subfolder = service.files().create(
                body=subfolder_metadata,
                fields='id, name, webViewLink'
            ).execute()

            subfolder_id = subfolder.get('id')
            print(f"✓ Created new subfolder: {subfolder.get('name')}")
            print(f"  Folder ID: {subfolder_id}")
            print(f"  Link: {subfolder.get('webViewLink')}")

        print(f"\nFiles to upload: {len(files_to_upload)}")
    except Exception as e:
        print(f"✗ Error handling subfolder: {e}")
        raise RuntimeError(f"Failed to handle subfolder '{date_folder_name}': {e}")

    uploaded_ids = []
    failed_files = []

    for file_path in files_to_upload:
        if file_path.is_file():
            print(f"\nProcessing: {file_path.name}...")

            media = MediaFileUpload(
                str(file_path),
                resumable=True
            )

            try:
                # Check if file already exists in this folder
                query = f"name='{file_path.name}' and '{subfolder_id}' in parents and trashed=false"
                results = service.files().list(
                    q=query,
                    spaces='drive',
                    fields='files(id, name)'
                ).execute()

                existing_files = results.get('files', [])

                if existing_files:
                    # Update existing file
                    file_id = existing_files[0]['id']
                    file = service.files().update(
                        fileId=file_id,
                        media_body=media,
                        fields='id, name, webViewLink'
                    ).execute()

                    uploaded_ids.append(file.get('id'))
                    print(f"  ✓ Updated existing file: {file.get('name')}")
                    print(f"    File ID: {file.get('id')}")
                    print(f"    Link: {file.get('webViewLink')}")
                else:
                    # Create new file
                    file_metadata = {
                        'name': file_path.name,
                        'parents': [subfolder_id]
                    }

                    file = service.files().create(
                        body=file_metadata,
                        media_body=media,
                        fields='id, name, webViewLink'
                    ).execute()

                    uploaded_ids.append(file.get('id'))
                    print(f"  ✓ Uploaded new file: {file.get('name')}")
                    print(f"    File ID: {file.get('id')}")
                    print(f"    Link: {file.get('webViewLink')}")

            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {e}")
                failed_files.append((file_path.name, str(e)))

    print(f"\n{'='*70}")
    print(f"Upload complete: {len(uploaded_ids)}/{len(files_to_upload)} files")
    print(f"{'='*70}")

    # Raise error if any files failed
    if failed_files:
        print(f"\n❌ Failed to upload {len(failed_files)} file(s):")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
        if len(uploaded_ids) == 0:
            raise RuntimeError(f"All {len(files_to_upload)} files failed to upload")
        else:
            raise RuntimeError(f"{len(failed_files)} out of {len(files_to_upload)} files failed to upload")

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
    parser.add_argument(
        '--oauth-credentials',
        help='OAuth credentials JSON as string (recommended for personal Drive)',
        default=os.getenv('GOOGLE_DRIVE_OAUTH_CREDENTIALS')
    )

    args = parser.parse_args()

    try:
        uploaded_ids = upload_directory_to_drive(
            local_dir=args.local_dir,
            folder_id=args.folder_id,
            credentials_path=args.credentials,
            credentials_json=args.credentials_json,
            oauth_credentials_json=args.oauth_credentials
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
