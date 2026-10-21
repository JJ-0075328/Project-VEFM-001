import pandas as pd
import numpy as np
import os

def generate_farm_data(file_path, num_samples=5000, seed=42):
    np.random.seed(seed)
    
    crop_types = ["Spinach", "Kale", "Lettuce", "Basil"]
    
    # Generate features
    tray_ids = np.arange(1, num_samples + 1)
    crop_type_choices = np.random.choice(crop_types, size=num_samples)
    plant_quantity = np.random.randint(30, 200, size=num_samples)
    growth_stage = np.random.randint(1, 4, size=num_samples)
    
    temperature = np.random.uniform(20.0, 30.0, size=num_samples)
    humidity = np.random.uniform(40.0, 80.0, size=num_samples)
    light = np.random.uniform(200.0, 900.0, size=num_samples)
    co2 = np.random.uniform(300.0, 1200.0, size=num_samples)
    ph = np.random.uniform(5.5, 7.0, size=num_samples)
    ec = np.random.uniform(1.2, 3.0, size=num_samples)
    
    # Generate targets with realistic physical dependencies and some noise
    # Crop Yield (kg) increases with Plant Quantity, Growth Stage, Light, and CO2
    crop_yield = (
        (plant_quantity * 0.08) * (growth_stage * 0.4) +
        (light * 0.003) + (co2 * 0.001) - (np.abs(temperature - 24.0) * 0.1) +
        np.random.normal(0, 1.0, size=num_samples)
    )
    # Clip yield to be positive
    crop_yield = np.clip(crop_yield, 0.5, None)
    
    # Water Usage (L) depends on Plant Quantity, Growth Stage, Temperature, and EC
    water_usage = (
        (plant_quantity * 0.5) * (growth_stage * 0.3) +
        (temperature * 1.5) + (humidity * -0.2) +
        np.random.normal(0, 5.0, size=num_samples)
    )
    water_usage = np.clip(water_usage, 5.0, None)
    
    # Energy Usage (kWh) depends on Light (from LED), Temperature, and CO2
    energy_usage = (
        (light * 0.01) + (np.abs(temperature - 22.0) * 0.3) +
        (co2 * 0.0005) + np.random.normal(0, 0.5, size=num_samples)
    )
    energy_usage = np.clip(energy_usage, 0.5, None)
    
    df = pd.DataFrame({
        "Tray_ID": tray_ids,
        "CropType": crop_type_choices,
        "Plant_Quantity": plant_quantity,
        "GrowthStage": growth_stage,
        "Crop_Yield_kg": crop_yield,
        "Water_Usage_L": water_usage,
        "Energy_Usage_kWh": energy_usage,
        "Temperature": temperature,
        "Humidity": humidity,
        "Light": light,
        "CO2": co2,
        "pH": ph,
        "EC": ec
    })
    
    df.to_csv(file_path, index=False)
    print(f"Generated dataset saved to: {file_path} ({num_samples} samples)")

if __name__ == "__main__":
    generate_farm_data("tend_agri_trays_set1.csv", num_samples=5000, seed=42)
    generate_farm_data("tend_agri_trays_set2.csv", num_samples=5000, seed=43)
    generate_farm_data("tend_agri_trays_set3.csv", num_samples=5000, seed=44)
