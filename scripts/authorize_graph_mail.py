import time

import requests


AUTHORITY = "https://login.microsoftonline.com/consumers/oauth2/v2.0"
SCOPES = "https://graph.microsoft.com/Mail.Send offline_access"


def main() -> None:
    client_id = input("Microsoft app client ID: ").strip()
    if not client_id:
        raise SystemExit("A Microsoft app client ID is required.")

    device_response = requests.post(
        f"{AUTHORITY}/devicecode",
        data={"client_id": client_id, "scope": SCOPES},
        timeout=30,
    )
    device_data = device_response.json()
    if not device_response.ok:
        raise SystemExit(device_data.get("error_description", "Device authorization failed."))

    print(device_data["message"])
    deadline = time.monotonic() + device_data["expires_in"]
    interval = device_data.get("interval", 5)

    while time.monotonic() < deadline:
        time.sleep(interval)
        token_response = requests.post(
            f"{AUTHORITY}/token",
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": client_id,
                "device_code": device_data["device_code"],
            },
            timeout=30,
        )
        token_data = token_response.json()

        if token_response.ok:
            refresh_token = token_data.get("refresh_token")
            if not refresh_token:
                raise SystemExit("Microsoft did not return a refresh token. Try authorizing again.")
            print("Authorization succeeded. Add this value as the GitHub Actions secret MS_GRAPH_REFRESH_TOKEN:")
            print(refresh_token)
            return

        error = token_data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        raise SystemExit(token_data.get("error_description", f"Device authorization failed: {error}"))

    raise SystemExit("The device code expired. Run this script again to start a new authorization.")


if __name__ == "__main__":
    main()