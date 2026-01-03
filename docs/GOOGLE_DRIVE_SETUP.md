# Google Drive OAuth Setup Guide

This guide explains how to set up OAuth credentials for automated Google Drive uploads in GitHub Actions.

## Why OAuth instead of Service Account?

Service accounts cannot upload files to personal Google Drive folders due to storage quota limitations. OAuth authentication allows the automation to act on behalf of your Google account, enabling uploads to your personal Drive.

## Setup Steps

### 1. Create OAuth 2.0 Credentials in Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create a new one)
3. Enable the Google Drive API:
   - Go to **APIs & Services** → **Library**
   - Search for "Google Drive API"
   - Click **Enable**

4. Create OAuth credentials:
   - Go to **APIs & Services** → **Credentials**
   - Click **Create Credentials** → **OAuth client ID**
   - If prompted, configure the OAuth consent screen first:
     - User Type: **External**
     - App name: `Fieldprep Automation` (or any name)
     - User support email: Your email
     - Developer contact: Your email
     - Click **Save and Continue** through the remaining steps
   - Back to Create OAuth client ID:
     - Application type: **Desktop app**
     - Name: `Fieldprep Desktop Client`
     - Click **Create**
   - Click **Download JSON** and save as `client_secrets.json`

### 2. Generate OAuth Credentials Locally

Run the setup script on your local machine (not in GitHub Actions):

```bash
# Place the downloaded client_secrets.json in the project root
cp ~/Downloads/client_secret_*.json client_secrets.json

# Run the OAuth setup script
python scripts/setup_oauth.py
```

This will:
1. Open your browser for Google account authorization
2. Ask you to sign in and grant permissions
3. Generate OAuth credentials and save to `oauth_credentials.json`
4. Display the credentials JSON to copy

### 3. Add Credentials to GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

**Secret 1:**
- Name: `GOOGLE_DRIVE_OAUTH_CREDENTIALS`
- Value: (paste the entire JSON output from step 2)

**Secret 2:**
- Name: `GOOGLE_DRIVE_FOLDER_ID`
- Value: Your Google Drive folder ID (e.g., `17Eexa-x7fOIB0gOu63SWUkZlNSr5oyk8`)
  - To find the folder ID, open the folder in Google Drive and copy the ID from the URL:
  - URL format: `https://drive.google.com/drive/folders/{FOLDER_ID}`

### 4. Test the Workflow

Trigger the workflow manually to test:

```bash
gh workflow run weekly-plan-emit.yml
```

Or wait for the scheduled run (every Friday at 10:00 PM PST).

## Troubleshooting

### Token Refresh
OAuth tokens expire after 1 hour, but the refresh token is long-lived. The upload script automatically handles token refresh.

### Permission Errors
If you get permission errors:
1. Make sure you granted all requested permissions during OAuth authorization
2. Try re-running `python scripts/setup_oauth.py` to regenerate credentials
3. Verify the Google Drive folder exists and is accessible from your account

### Scope Issues
The required scope is `https://www.googleapis.com/auth/drive.file`, which allows:
- Creating files
- Modifying files created by the application
- Does NOT allow access to files created by other applications

## Files to Keep Secret

The following files contain sensitive credentials and should NEVER be committed to git:
- `client_secrets.json` - OAuth client configuration
- `oauth_credentials.json` - Your OAuth tokens
- `token.json` - Cached OAuth token

These are already listed in `.gitignore`.

## Maintenance

OAuth credentials are long-lived but may need to be refreshed if:
- You revoke access in your Google account
- Google changes security policies
- The refresh token expires (rare, but possible after 6 months of inactivity)

If uploads start failing, re-run the setup process to generate new credentials.
