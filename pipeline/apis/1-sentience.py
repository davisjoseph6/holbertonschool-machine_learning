#!/usr/bin/env python3
"""
Module to retrieve the home planets of all sentient species
using the SWAPI API.
"""

import requests


def sentientPlanets():
    """
    Retrieves the list of home planet names for all sentient species.
    """
    base_url = "https://swapi-api.hbtn.io/api/species/"
    planets = set()
    sentient_types = {"sentient"}  # Possible types for sentient beings
    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            break
        data = response.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if classification in sentient_types or designation in sentient_types:
                homeworld = species.get("homeworld")
                if homeworld:
                    planet_response = requests.get(homeworld)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.add(planet_data.get("name", "unknown"))
                    else:
                        planets.add("unknown")
        base_url = data.get("next")  # Get the next page URL
    return sorted(planets)  # Return sorted list for consistent results
