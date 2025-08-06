import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
BOLTZMANN_CONST = 0.695  # cm^-1/K (converted for our specific calculations)
PLANCK_CONST = 6.626e-34  # J·s
SPEED_OF_LIGHT = 2.998e10  # cm/s

############################################################################
# Section 1: Calculating lifetime from a single CSV file
############################################################################

def calculate_lifetime(csv_file,plot=False):
    """
    Calculate the lifetime (tau) from a single CSV file of ruby fluorescence data.
    
    Parameters:
    csv_file (str): Path to the CSV file
    
    Returns:
    float: Calculated lifetime (tau) in milliseconds
    """
    # Read CSV data
    data = pd.read_csv(csv_file,skiprows=22)
    
    # Get the file name without extension for labeling
    csv_basename = os.path.basename(csv_file)
    
    # Rename columns for clarity
    if len(data.columns) >= 2:
        data.columns = ['TIME', 'CH1']
    else:
        raise ValueError(f"CSV file {csv_file} doesn't have enough columns")
    
    # Find minimum value to offset data
    ch1_min = data['CH1'].min()
    
    # Add calculated columns
    data['CH1+'] = data['CH1'] + abs(ch1_min) + 1e-8
    #positive
    data['t(ms)'] = data['TIME'] * 1000  # Convert time to milliseconds
    data['CH1+(mV)'] = data['CH1+'] * 1000  # Convert voltage to millivolts
    data['ln(CH1+)'] = np.log(data['CH1+(mV)'])  # Natural log of the offset voltage
    
    # Filter data:
    # 1. Remove data before t = 0
    # 2. Remove data where ln(CH1+(mV)) < 1
    filtered_data = data[(data['t(ms)'] >= 0) & (data['ln(CH1+)'] >= 2)]
    
    # Extract x and y for linear regression
    x_values = filtered_data['t(ms)'].values
    y_values = filtered_data['ln(CH1+)'].values
    
    # Linear regression to calculate the slope
    slope, intercept, r_value, p_value, std_err = linregress(x_values, y_values)
    
    # Calculate tau (lifetime) in milliseconds
    tau = 1 / abs(slope)
    
    if plot:
        # Create plots (signal vs time and ln(signal) vs time)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Decay of Ruby Fluorescence
        ax1.plot(data['t(ms)'], data['CH1+(mV)'], 'k-', linewidth=2)
        ax1.set_title('Decay of Ruby Fluorescence', fontweight='bold')
        ax1.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Response (mV)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Natural log plot with linear fit
        ax2.plot(data['t(ms)'], data['ln(CH1+)'], 'k-', linewidth=2)
        ax2.plot(x_values, intercept + slope * x_values, 'r-', linewidth=2)
        ax2.set_title('ln(Response) vs Time', fontweight='bold')
        ax2.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('ln(Response)', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add equation and R² to the plot
        equation = f'y = {slope:.4f}x + {intercept:.4f}'
        r_squared = f'R² = {r_value**2:.4f}'
        ax2.text(0.7, 0.1, f'{equation}\n{r_squared}', 
                 transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
        print(f"File: {csv_basename}")
        print(f"Calculated lifetime (τ): {tau:.4f} ms")
    
    return tau

# Example usage:
# tau = calculate_lifetime('path/to/tek0004CH1.csv')

############################################################################
# Section 2: Processing multiple CSV files to get lifetime vs temperature data
############################################################################

def process_all_files(folder_path, initial_temperature):
    """
    Process all CSV files in a folder and calculate lifetime vs temperature data.
    
    Parameters:
    folder_path (str): Path to the folder containing CSV files
    temperature_map (dict): Dictionary mapping file numbers to temperatures in Celsius
    
    Returns:
    tuple: (temperatures in Kelvin, lifetimes in ms)
    """
    temperatures = [] 
    lifetimes = []

    
    # Get all CSV files in the folder
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    csv_files = [csv_file for csv_file in csv_files if csv_file.startswith('tek00') and not csv_file.startswith('tek0000')]
    
    temperatures_ = [initial_temperature + 5 * i for i in range(0,len(csv_files))]

    for i, csv_file in enumerate(csv_files):
        # Extract run number from filename (tek00XX format)
        try:
            if csv_file.startswith('tek00'):
                run_number = int(csv_file[5:7].lstrip('0'))
            else:
                # Skip files that don't match the expected format
                continue
                          
            # Convert to Kelvin
            temperature_kelvin = temperatures_[i]
                
            # Process the CSV file
            tau = calculate_lifetime(os.path.join(folder_path, csv_file))
            
            # Store results
            temperatures.append(temperature_kelvin)
            lifetimes.append(tau)
            
        except (ValueError, IndexError):
            print(f"Skipping file {csv_file}: could not extract run number")
            continue
    
    # Create a plot of lifetime vs temperature
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, lifetimes, c='blue', marker='o')
    plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Lifetime τ (ms)', fontsize=12, fontweight='bold')
    plt.title('Ruby Fluorescence Lifetime vs Temperature', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return np.array(temperatures), np.array(lifetimes)

# Example usage:
# Define a mapping of run numbers to temperatures
# temperature_map = {1: 25, 2: 30, 3: 35, ...}  # Run number: Temperature in Celsius
# temperatures, lifetimes = process_all_files('path/to/folder', temperature_map)

############################################################################
# Section 3: Calculating A_E and A_T
############################################################################

# def calculate_A_E(temperatures, lifetimes, max_temp=373.15):
#     """
#     Calculate A_E using low temperature data (up to 100°C / 373.15K)
    
#     Parameters:
#     temperatures (numpy.ndarray): Array of temperatures in Kelvin
#     lifetimes (numpy.ndarray): Array of lifetimes in milliseconds
#     max_temp (float): Maximum temperature to consider in Kelvin (default: 373.15K = 100°C)
    
#     Returns:
#     float: A_E value in ms^-1
#     """
#     # Filter data for low temperatures (up to specified max_temp)
#     low_temp_mask = temperatures <= max_temp
#     low_temp_data = temperatures[low_temp_mask]
#     low_lifetimes = lifetimes[low_temp_mask]
    
#     if len(low_lifetimes) == 0:
#         raise ValueError(f"No data points found below {max_temp}K ({max_temp-273.15}°C)")
    
#     # Calculate average lifetime at low temperatures to get A_E
#     tau_ave = np.mean(low_lifetimes)
#     A_E = 1 / tau_ave  # in ms^-1
    
#     print(f"A_E calculation:")
#     print(f"  Using {len(low_lifetimes)} data points below {max_temp}K ({max_temp-273.15}°C)")
#     print(f"  Average lifetime (τ): {tau_ave:.4f} ms")
#     print(f"  A_E = 1/τ = {A_E:.6f} ms^-1")
    
#     return A_E

def calculate_A_E(temperatures, lifetimes, max_temp=373.15):
    """
    Calculate A_E using low temperature data (up to 100°C / 373.15K)
    Also plots measured lifetimes vs model with only A_E
    
    Parameters:
    temperatures (numpy.ndarray): Array of temperatures in Kelvin
    lifetimes (numpy.ndarray): Array of lifetimes in milliseconds
    max_temp (float): Maximum temperature to consider in Kelvin (default: 373.15K = 100°C)
    
    Returns:
    float: A_E value in ms^-1
    """
    # Filter data for low temperatures (up to specified max_temp)
    low_temp_mask = temperatures <= max_temp
    low_temp_data = temperatures[low_temp_mask]
    low_lifetimes = lifetimes[low_temp_mask]
    
    if len(low_lifetimes) == 0:
        raise ValueError(f"No data points found below {max_temp}K ({max_temp-273.15}°C)")
    
    # Calculate average lifetime at low temperatures to get A_E
    tau_ave = np.mean(low_lifetimes)
    A_E = 1 / tau_ave  # in ms^-1
    
    print(f"A_E calculation:")
    print(f"  Using {len(low_lifetimes)} data points below {max_temp}K ({max_temp-273.15}°C)")
    print(f"  Average lifetime (τ): {tau_ave:.4f} ms")
    print(f"  A_E = 1/τ = {A_E:.6f} ms^-1")
    
    # Calculate lifetimes predicted by model with only A_E
    simple_model_lifetimes = []
    for temp in temperatures:
        tau_model = 1 / A_E 
        simple_model_lifetimes.append(tau_model)
    
    # Create plot comparing experimental and A_E-only model lifetimes
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, lifetimes, c='blue', marker='o', label='Experimental')
    plt.plot(temperatures, simple_model_lifetimes, 'g--', label='A_E only model')
    
    # Highlight data used to calculate A_E
    plt.scatter(low_temp_data, low_lifetimes, c='red', marker='o', 
                label=f'Data used for A_E (<{max_temp-273.15}°C)')
    
    plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Lifetime τ (ms)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Experimental and A_E-only Model', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation with A_E value
    plt.annotate(f'A_E = {A_E:.6f} ms⁻¹', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.show()
    
    return A_E


def calculate_A_T(temperatures, lifetimes, A_E, min_temp=373.15, max_temp=473.15):
    """
    Calculate A_T using medium temperature data (100°C to 200°C) and the calculated A_E
    
    Parameters:
    temperatures (numpy.ndarray): Array of temperatures in Kelvin
    lifetimes (numpy.ndarray): Array of lifetimes in milliseconds
    A_E (float): Previously calculated A_E value
    min_temp (float): Minimum temperature to consider in Kelvin (default: 373.15K = 100°C)
    max_temp (float): Maximum temperature to consider in Kelvin (default: 473.15K = 200°C)
    
    Returns:
    tuple: (A_T value, fitted lifetimes)
    """
    # Filter data for medium temperatures
    medium_temp_mask = (temperatures > min_temp) & (temperatures <= max_temp)
    medium_temp_data = temperatures[medium_temp_mask]
    medium_lifetimes = lifetimes[medium_temp_mask]
    
    if len(medium_lifetimes) == 0:
        raise ValueError(f"No data points found between {min_temp}K and {max_temp}K")
    
    # Calculate A_T using medium temperature data
    A_T_values = []
    
    for i, temp in enumerate(medium_temp_data):
        # Calculate n_T/n_E ratio according to the formula
        n_ratio = 8.311 * np.exp(-3380/temp)
        
        # Calculate A_T from the formula
        A_T = ((1 + 1/n_ratio) / medium_lifetimes[i]) - (A_E / n_ratio)
        A_T_values.append(A_T)
    
    # Use average A_T value
    A_T = np.mean(A_T_values)
    
    print(f"A_T calculation:")
    print(f"  Using {len(medium_lifetimes)} data points between {min_temp}K and {max_temp}K")
    print(f"  Individual A_T values: ", [f"{val:.6f}" for val in A_T_values])
    print(f"  A_T (average): {A_T:.6f} ms^-1")
    
    # Calculate fitted lifetimes using A_E and A_T for all temperatures
    fit_lifetimes = []
    for temp in temperatures:
        n_ratio = 8.311 * np.exp(-3380/temp)
        tau_fit = (1 + n_ratio) / (A_E + A_T * n_ratio)
        fit_lifetimes.append(tau_fit)
    
    # Create plot comparing experimental and fitted lifetimes
    plt.figure(figsize=(10, 6))
    plt.scatter(temperatures, lifetimes, c='blue', marker='o', label='Experimental')
    plt.plot(temperatures, fit_lifetimes, 'r-', label='Fitted (A_E and A_T only)')
    plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Lifetime τ (ms)', fontsize=12, fontweight='bold')
    plt.title('Comparison of Experimental and Fitted Lifetimes', fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return A_T, np.array(fit_lifetimes)

# Example usage:
# A_E = calculate_A_E(temperatures, lifetimes)
# A_T, fit_lifetimes = calculate_A_T(temperatures, lifetimes, A_E)

############################################################################
# Section 4: Calculating N(T), Activation Energy, and Phonon Wavenumber
############################################################################

def calculate_thermal_parameters(temperatures, lifetimes, A_E, A_T, min_temp=573.15):
    """
    Calculate N(T), activation energy, and phonon wavenumber.
    
    Parameters:
    temperatures (numpy.ndarray): Array of temperatures in Kelvin
    lifetimes (numpy.ndarray): Array of lifetimes in milliseconds
    A_E (float): Emission rate from E level in ms^-1
    A_T (float): Emission rate from T level in ms^-1
    min_temp (float): Minimum temperature to consider for N(T) in Kelvin (default: 573.15K = 300°C)
    
    Returns:
    tuple: (E_a, frequency_factor, wavenumber, full_fit_lifetimes)
    """
    # Calculate N(T) for each temperature
    nt_values = []
    valid_temps = []
    valid_nt = []
    
    for i, temp in enumerate(temperatures):
        # Calculate n_T/n_E ratio
        n_ratio = 8.311 * np.exp(-3380/temp)
        
        # Calculate N(T) using the formula
        nt = ((1 + 1/n_ratio) / lifetimes[i]) - (A_E / n_ratio) - A_T
        nt_values.append(nt)
        
        # Only keep positive N(T) values for Arrhenius fitting at higher temperatures
        if nt > 0 and temp > min_temp:
            valid_temps.append(1/temp)  # For Arrhenius plot (1/T)
            valid_nt.append(np.log(nt))  # For Arrhenius plot (ln(N(T)))
    
    print(f"N(T) calculation:")
    print(f"  Calculated {len(nt_values)} N(T) values")
    print(f"  Valid data points for Arrhenius plot: {len(valid_temps)}")
    
    # Convert to numpy arrays for linear regression
    valid_temps = np.array(valid_temps)
    valid_nt = np.array(valid_nt)
    
    # If we have enough valid data points, fit the Arrhenius equation
    if len(valid_temps) >= 2:
        # Linear regression for ln(N(T)) vs 1/T
        slope, intercept, r_value, p_value, std_err = linregress(valid_temps, valid_nt)
        
        # Calculate activation energy and frequency factor
        E_a = -slope * BOLTZMANN_CONST  # in cm^-1
        frequency_factor = np.exp(intercept)  # Pre-exponential factor
        
        # Calculate wavenumber associated with this activation energy
        wavenumber = E_a  # This is already in cm^-1
        
        print(f"Arrhenius analysis:")
        print(f"  Slope: {slope:.2f}")
        print(f"  Intercept: {intercept:.2f}")
        print(f"  R-squared: {r_value**2:.4f}")
        print(f"  Activation energy (E_a): {E_a:.2f} cm^-1")
        print(f"  Frequency factor (A): {frequency_factor:.2e} ms^-1")
        print(f"  Phonon wavenumber: {wavenumber:.2f} cm^-1")
        
        # Calculate fitted N(T) values for all temperatures
        nt_fit = []
        for temp in temperatures:
            if temp > min_temp:  # Only calculate for higher temperatures
                nt_val = frequency_factor * np.exp(-E_a / (BOLTZMANN_CONST * temp))
            else:
                nt_val = 0
            nt_fit.append(nt_val)
        
        # Calculate full fit including N(T)
        full_fit_lifetimes = []
        for i, temp in enumerate(temperatures):
            n_ratio = 8.311 * np.exp(-3380/temp)
            if temp > min_temp:  # Only include N(T) for higher temperatures
                tau_fit = (1 + n_ratio) / (A_E + (A_T + nt_fit[i]) * n_ratio)
            else:
                tau_fit = (1 + n_ratio) / (A_E + A_T * n_ratio)
            full_fit_lifetimes.append(tau_fit)
        
        # Create Arrhenius plot
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_temps, valid_nt, c='blue', marker='o')
        plt.plot(valid_temps, intercept + slope * valid_temps, 'r-')
        plt.xlabel('1/T (K^-1)', fontsize=12, fontweight='bold')
        plt.ylabel('ln(N(T))', fontsize=12, fontweight='bold')
        plt.title('Arrhenius Plot for N(T)', fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add equation to the plot
        equation = f'ln(N(T)) = {intercept:.2f} + ({slope:.2f})·(1/T)'
        plt.text(0.05, 0.9, equation, transform=plt.gca().transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
        plt.show()
        
        # Create final comparison plot with full model
        plt.figure(figsize=(10, 6))
        plt.scatter(temperatures, lifetimes, c='blue', marker='o', label='Experimental')
        plt.plot(temperatures, full_fit_lifetimes, 'r-', label='Full Model Fit')
        plt.xlabel('Temperature (K)', fontsize=12, fontweight='bold')
        plt.ylabel('Lifetime τ (ms)', fontsize=12, fontweight='bold')
        plt.title('Ruby Fluorescence Lifetime: Experiment vs Full Model', fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
        return E_a, frequency_factor, wavenumber, np.array(full_fit_lifetimes)
    else:
        print("Not enough valid data points for Arrhenius fitting")
        return None, None, None, None

# Example usage:
# E_a, frequency_factor, wavenumber, full_fit = calculate_thermal_parameters(
#     temperatures, lifetimes, A_E, A_T)
