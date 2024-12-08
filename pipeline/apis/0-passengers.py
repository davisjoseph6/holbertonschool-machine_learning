#!/usr/bin/env python3
"""
Module to retrieve ships that can hold a given number of passengers
using the SWAPI API.
"""

import requests


def availableShips(passengerCount):
    """
    Retrieves a list of starships that can hold a given number of passengers.
    """
    base_url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []
    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            break
        data = response.json()
        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship.get("name"))
        base_url = data.get("next")  # Get the next page URL
    return ships
