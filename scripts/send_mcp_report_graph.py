import os
import sys

import requests


TOKEN_URL = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
GRAPH_SEND_MAIL_URL = "https://graph.microsoft.com/v1.0/me/sendMail"
SCOPES = "https://graph.microsoft.com/Mail.Send offline_access"


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set.")
    return value


def _access_token(client_id: str, refresh_token: str) -> str:
    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "scope": SCOPES,
        },
        timeout=30,
    )
    result = response.json()
    if not response.ok:
        error = result.get("error_description", "Microsoft token refresh failed.")
        raise RuntimeError(error)
    return result["access_token"]


def main() -> None:
    client_id = _required_env("MS_GRAPH_CLIENT_ID")
    refresh_token = _required_env("MS_GRAPH_REFRESH_TOKEN")
    recipients = [
        address.strip()
        for address in _required_env("MCP_REPORT_EMAIL_TO").split(",")
        if address.strip()
    ]
    if not recipients:
        raise RuntimeError("MCP_REPORT_EMAIL_TO must contain at least one email address.")

    repository = os.getenv("GITHUB_REPOSITORY", "unknown repository")
    branch = os.getenv("GITHUB_REF_NAME", "unknown branch")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    server_url = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    job_status = os.getenv("MCP_JOB_STATUS", "unknown")
    run_url = f"{server_url}/{repository}/actions/runs/{run_id}"

    payload = {
        "message": {
            "subject": f"[MCP tests] {job_status} - {repository}",
            "body": {
                "contentType": "Text",
                "content": (
                    "MCP test workflow completed.\n\n"
                    f"Result: {job_status}\n"
                    f"Repository: {repository}\n"
                    f"Branch: {branch}\n"
                    f"Workflow run: {run_url}\n\n"
                    "Download the mcp-test-report artifact from the workflow run for the JUnit report."
                ),
            },
            "toRecipients": [
                {"emailAddress": {"address": address}}
                for address in recipients
            ],
        },
        "saveToSentItems": True,
    }
    response = requests.post(
        GRAPH_SEND_MAIL_URL,
        headers={"Authorization": f"Bearer {_access_token(client_id, refresh_token)}"},
        json=payload,
        timeout=30,
    )
    if response.status_code != 202:
        raise RuntimeError(f"Microsoft Graph sendMail failed ({response.status_code}): {response.text}")
    print(f"MCP test notification accepted by Microsoft Graph for {len(recipients)} recipient(s).")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, RuntimeError, KeyError) as error:
        print(f"Email notification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error