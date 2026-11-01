import json
import urllib.request
import sys

URL = "http://127.0.0.1:8000/health"

def test_health():
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "HealthChecker/1.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"Health Check Passed: Status={data.get('status')}, Version={data.get('version')}")
                return 0
            else:
                print(f"Health Check Failed: HTTP {response.status}")
                return 1
    except Exception as e:
        print(f"Health Check Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(test_health())
