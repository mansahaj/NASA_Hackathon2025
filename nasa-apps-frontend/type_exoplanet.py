import pandas as pd

def classify_exoplanet_type(row):
    """
    Classifies an exoplanet based on its radius (size) and equilibrium temperature.
    - Radius is in Earth radii (`koi_prad`).
    - Temperature is in Kelvin (`koi_teq`).
    """
    radius = row['koi_prad']
    temp = row['koi_teq']

    # Handle cases where data might be missing
    if pd.isna(radius) or pd.isna(temp):
        return "Unknown"

    # 1. Determine size category
    if radius > 6:
        size_class = "Gas Giant"
    elif 2 < radius <= 6:
        size_class = "Neptune-like"
    elif 1.25 < radius <= 2:
        size_class = "Super-Earth"
    else:  # radius <= 1.25
        size_class = "Terrestrial"

    # 2. Determine temperature prefix
    if temp > 1000:
        temp_class = "Hot"
    elif 250 <= temp <= 1000:
        temp_class = "Warm"
    else:  # temp < 250
        temp_class = "Cold"
        
    return f"{temp_class} {size_class}"