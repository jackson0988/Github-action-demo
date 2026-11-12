"""Tell Render to run the new image, wait until it is live, then test the live URL.

The CI deploy job first pushes the image to GitHub's registry and sets IMAGE.
Needs three environment variables:
    IMAGE               the image to run, for example ghcr.io/your-name/taxi-api:ci-1a2b3c4
    RENDER_DEPLOY_HOOK  the service's deploy hook URL (secret: anyone with it can deploy)
    RENDER_URL          the service's public URL, for example https://taxi-api-xxxx.onrender.com

Uses only the Python standard library, so nothing to install.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

WAIT_MINUTES = 15  # Render pulls the image and starts it, usually a few minutes
TEST_TRIP = {"PULocationID": 74, "DOLocationID": 75, "trip_distance": 2.5, "hour": 8, "weekday": 1}

SETUP_HELP = """
One-time setup is not finished. The image is already pushed: {image}

1. GitHub > your profile > Packages > taxi-api > Package settings > Change visibility > Public
2. Render: New > Web Service > Existing Image > {latest}
   Instance type: Free. Create it.
3. Render service > Settings > Deploy Hook: copy the URL
4. GitHub repo > Settings > Secrets and variables > Actions
     Secrets tab:   RENDER_DEPLOY_HOOK = the deploy hook URL
     Variables tab: RENDER_URL         = the service URL, like https://taxi-api-xxxx.onrender.com
5. On this run: Re-run jobs > Re-run failed jobs
"""


def call(url, body=None):
    """GET the URL, or POST the body as JSON, and return the JSON answer."""
    headers = {"User-Agent": "taxi-class-5-deploy"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def trigger_deploy(hook, image):
    """Ask Render to deploy this exact image tag."""
    separator = "&" if "?" in hook else "?"
    url = f"{hook}{separator}imgURL={urllib.parse.quote(image, safe='')}"
    request = urllib.request.Request(url, method="POST", headers={"User-Agent": "taxi-class-5-deploy"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"Render accepted the deploy (HTTP {response.status})")
    except urllib.error.HTTPError as error:
        sys.exit(
            f"FAILED: Render refused the deploy (HTTP {error.code}). Check that RENDER_DEPLOY_HOOK is the "
            f"hook of this service, and that the service image is {image.rsplit(':', 1)[0]}:latest"
        )


def wait_until_live(url, version):
    """Wait until the live API reports the new model version."""
    deadline = time.time() + WAIT_MINUTES * 60
    while time.time() < deadline:
        try:
            live_version = call(f"{url}/health")["model_version"]
        except Exception:
            live_version = "not answering yet"
        print(f"live version: {live_version}")
        if live_version == version:
            return
        time.sleep(20)
    sys.exit(
        f"FAILED: version {version} was not live after {WAIT_MINUTES} minutes. "
        "Open the service on dashboard.render.com and read its Events and Logs"
    )


def write_summary(text):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")  # set by GitHub Actions
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as summary:
            summary.write(text)


def main():
    version_file = Path("model/version.txt")
    if not version_file.exists():
        sys.exit("No model/ folder. Run python src/check_model.py first")
    version = version_file.read_text().strip()

    image = os.environ.get("IMAGE", "")
    hook = os.environ.get("RENDER_DEPLOY_HOOK", "")
    url = os.environ.get("RENDER_URL", "").rstrip("/")
    if not image:
        sys.exit("Set IMAGE first. In CI the 'Build the image and push it' step sets it")
    if not hook or not url:
        sys.exit(SETUP_HELP.format(image=image, latest=image.rsplit(":", 1)[0] + ":latest"))

    print(f"Deploying model {version} as {image}")
    trigger_deploy(hook, image)
    wait_until_live(url, version)
    answer = call(f"{url}/predict", TEST_TRIP)
    print(f"LIVE: {url}/predict answered {answer}")

    write_summary(
        f"""### 🚀 Deployed model `{version}`

| | |
|---|---|
| Live API | {url} |
| Try it in the browser | {url}/docs |
| Postman | `POST {url}/predict` |
| Test prediction | {answer['minutes']} minutes for `{json.dumps(TEST_TRIP)}` |
| Image | `{image}` |
"""
    )


if __name__ == "__main__":
    main()
