#!/usr/bin/env python
"""
Setup OAuth 2.0 credentials for Google Drive access

This script helps you generate OAuth credentials that can be used for
automated Google Drive uploads in GitHub Actions.

Steps:
1. Run this script locally: python scripts/setup_oauth.py
2. Follow the authorization flow in your browser
3. Copy the generated credentials JSON
4. Add it to GitHub Secrets as GOOGLE_DRIVE_OAUTH_CREDENTIALS

Requirements:
- You need to create OAuth 2.0 credentials in Google Cloud Console
- Download the client secrets JSON file
"""

import os
import json
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes required for Drive file upload
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def setup_oauth_credentials(client_secrets_path='client_secrets.json'):
    """
    Set up OAuth credentials for Google Drive access.

    Parameters
    ----------
    client_secrets_path : str
        Path to the OAuth client secrets JSON file downloaded from Google Cloud Console

    Returns
    -------
    dict
        OAuth credentials as a dictionary (to be stored in GitHub Secrets)
    """
    client_secrets_path = Path(client_secrets_path)

    if not client_secrets_path.exists():
        print(f"\n❌ Error: Client secrets file not found: {client_secrets_path}")
        print("\nTo create OAuth credentials:")
        print("1. Go to https://console.cloud.google.com/apis/credentials")
        print("2. Click 'Create Credentials' → 'OAuth client ID'")
        print("3. Choose 'Desktop app' as application type")
        print("4. Download the JSON file and save as 'client_secrets.json'")
        print("5. Run this script again")
        return None

    print(f"\n{'='*70}")
    print("Google Drive OAuth Setup")
    print(f"{'='*70}\n")

    # Check if we have existing credentials
    token_path = Path('token.json')
    creds = None

    if token_path.exists():
        print("Found existing credentials (token.json)")
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    # If credentials don't exist or are invalid, run the OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired credentials...")
            creds.refresh(Request())
        else:
            print("Starting OAuth authorization flow...")
            print("\nYour browser will open for authorization.")
            print("Please sign in with your Google account and grant access.\n")

            flow = InstalledAppFlow.from_client_secrets_file(
                str(client_secrets_path), SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
        print("✓ Credentials saved to token.json")

    # Convert credentials to dictionary format for GitHub Secrets
    creds_dict = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

    print(f"\n{'='*70}")
    print("✅ OAuth Setup Complete!")
    print(f"{'='*70}\n")

    print("Add the following to GitHub Secrets as GOOGLE_DRIVE_OAUTH_CREDENTIALS:")
    print(f"\n{'-'*70}")
    print(json.dumps(creds_dict, indent=2))
    print(f"{'-'*70}\n")

    print("Steps to add to GitHub:")
    print("1. Go to your repository Settings → Secrets and variables → Actions")
    print("2. Click 'New repository secret'")
    print("3. Name: GOOGLE_DRIVE_OAUTH_CREDENTIALS")
    print("4. Value: (paste the JSON above)")
    print("5. Update your workflow to use --oauth-credentials instead of --credentials-json\n")

    return creds_dict


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Setup OAuth credentials for Google Drive')
    parser.add_argument(
        '--client-secrets',
        default='client_secrets.json',
        help='Path to OAuth client secrets JSON file'
    )

    args = parser.parse_args()

    try:
        creds_dict = setup_oauth_credentials(args.client_secrets)

        if creds_dict:
            # Optionally save to a file
            output_file = 'oauth_credentials.json'
            with open(output_file, 'w') as f:
                json.dump(creds_dict, f, indent=2)
            print(f"Credentials also saved to: {output_file}")
            print("⚠️  Remember to add this file to .gitignore!")
            return 0
        else:
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
