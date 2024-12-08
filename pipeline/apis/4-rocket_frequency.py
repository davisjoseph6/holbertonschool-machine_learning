#!/usr/bin/env python3
"""
Script to display the number of launches per rocket using the SpaceX API.
"""

import requests
from collections import defaultdict


def main():
    """
    Fetch and display the number of launches per rocket in descending order.
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    try:
        # Fetch launches data
        launches_response = requests.get(launches_url)
        if launches_response.status_code != 200:
            print("Error fetching launches")
            return
        launches = launches_response.json()

        # Count launches per rocket ID
        rocket_launch_count = defaultdict(int)
        for launch in launches:
            rocket_launch_count[launch["rocket"]] += 1

        # Fetch rocket data
        rockets_response = requests.get(rockets_url)
        if rockets_response.status_code != 200:
            print("Error fetching rockets")
            return
        rockets = rockets_response.json()

        # Create a dictionary to map rocket IDs to their names
        rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

        # Create a list of tuples with rocket names and their launch counts
        rocket_data = [
                (rocket_names[rocket_id], count)
                for rocket_id, count in rocket_launch_count.items()
                if rocket_id in rocket_names
                ]

        # Sort the list by number of launches (descending) and then
        # alphabetically by name
        rocket_data.sort(key=lambda x: (-x[1], x[0]))

        # Print the results
        for name, count in rocket_data:
            print(f"{name}: {count}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    main()
