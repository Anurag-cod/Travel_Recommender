import pandas as pd

def get_unique_states(travel_data):
    """Get a list of unique states from the travel data."""
    return travel_data['State'].unique()
