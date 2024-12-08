#!/usr/bin/env python3
"""
Script to fetch and print the location of a specific GitHub user.
"""

import requests
import sys
from datetime import datetime


def get_time_until_reset(reset_time):
    """
    Calculate the minutes until the rate limit resets.
    """
    current_time = datetime.now().timestamp()
    return int((reset_time - current_time) / 60)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./2-user_location.py <GitHub API URL>")
        sys.exit(1)

    api_url = sys.argv[1]

    try:
        response = requests.get(api_url)

        if response.status_code == 200:
            user_data = response.json()
            location = user_data.get("location")
            print(location if location else "Location not specified")
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            minutes_until_reset = get_time_until_reset(reset_time)
            print(f"Reset in {minutes_until_reset} min")
        else:
            print(f"Error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
