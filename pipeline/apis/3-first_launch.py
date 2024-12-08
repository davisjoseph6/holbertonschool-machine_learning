#!/usr/bin/env python3
"""
Script to display the first upcoming SpaceX launch information using the SpaceX API.
"""

import requests
from datetime import datetime


def fetch_first_upcoming_launch():
    """
    Fetches and displays the next upcoming SpaceX launch with the required details.
    """
    launches_url = "https://api.spacexdata.com/v4/launches/upcoming"
    rockets_url = "https://api.spacexdata.com/v4/rockets"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads"

    try:
        # Fetch upcoming launches
        launches_response = requests.get(launches_url)
        if launches_response.status_code != 200:
            print("Error fetching launches")
            return

        launches = launches_response.json()

        # Find the first upcoming launch based on date_unix
        first_launch = min(launches, key=lambda x: x["date_unix"])

        # Fetch rocket data
        rocket_id = first_launch["rocket"]
        rocket_response = requests.get(f"{rockets_url}/{rocket_id}")
        if rocket_response.status_code != 200:
            print("Error fetching rocket details")
            return
        rocket_name = rocket_response.json()["name"]

        # Fetch launchpad data
        launchpad_id = first_launch["launchpad"]
        launchpad_response = requests.get(f"{launchpads_url}/{launchpad_id}")
        if launchpad_response.status_code != 200:
            print("Error fetching launchpad details")
            return
        launchpad_data = launchpad_response.json()
        launchpad_name = launchpad_data["name"]
        launchpad_locality = launchpad_data["locality"]

        # Format date
        launch_date = datetime.fromtimestamp(first_launch["date_unix"]).isoformat()

        # Output the formatted result
        print(
                f"{first_launch['name']} ({launch_date}) {rocket_name} - "
                f"{launchpad_name} ({launchpad_locality})"
                )

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    fetch_first_upcoming_launch()
