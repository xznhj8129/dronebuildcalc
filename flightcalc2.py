import pandas as pd
import math
import numpy as np
import itertools

# Load motor data from CSV file
motor_data = pd.read_csv('motordata.csv')

# Function to calculate mean absolute percentage error (MAPE)
def calculate_mape(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

# MotorData class to store and execute polynomial functions
class MotorData:
    def __init__(self, brand, motor, prop):
        self.brand = brand
        self.motor = motor
        self.prop = prop
        self.polynomials = {}

    def add_polynomial(self, name, func):
        self.polynomials[name] = func

    def __getattr__(self, name):
        # Dynamically call the polynomial stored in the object
        if name in self.polynomials:
            return self.polynomials[name]
        raise AttributeError(f"{name} not found in MotorData.")

# Dictionary to store the best MotorData objects for each engine
motor_data_objects = {}

# List of target variables (those that we want to predict)
target_variables = ['rpm', 'current', 'watts', 'efficiency', 'throttle']

print('Computing Motor data')

# Iterate through each group of Brand, Motor, and Prop (i.e., each unique engine)
for (brand, motor, prop), group in motor_data.groupby(['Brand', 'Motor', 'Prop']):
    print(f"Processing {brand} {motor} {prop}")

    # Extract relevant data
    throttle = group['T'].str.rstrip('%').astype(float).values
    thrust = group['Thrust g'].values
    current = group['A'].values
    rpm = group['RPM'].values
    watts = group['W'].values
    efficiency = group['Efficiency'].values

    # Prepare thrust values based on the range (from 50 to maxt in steps of 50)
    maxt = thrust[-1]
    thrust_values = np.arange(50, maxt, 50)

    # Dictionary to hold polynomials for every combination
    poly_dict = {}

    # List of variables for combinations
    data_dict = {
        'throttle': throttle,
        'thrust': thrust,
        'current': current,
        'rpm': rpm,
        'watts': watts,
        'efficiency': efficiency
    }
    
    # Generate polynomial fit functions for every combination of variables
    for (x_var, y_var) in itertools.permutations(data_dict.keys(), 2):
        x_data = data_dict[x_var]
        y_data = data_dict[y_var]
        coeffs = np.polyfit(x_data, y_data, 2)
        poly_func = np.poly1d(coeffs)
        poly_dict[f'{x_var}_to_{y_var}'] = poly_func
    
    # Create a MotorData object for this engine
    motor_data_obj = MotorData(brand, motor, prop)

    # Iterate over each target variable (rpm, current, watts, efficiency, etc.)
    for target in target_variables:
        results = []

        # Chain the functions for 1-step, 2-step, 3-step, and 4-step combinations
        variables = list(data_dict.keys())
        variables.remove(target)  # We don't use the target variable as an input in the chain
        
        for length in range(1, 5):  # 1-step, 2-step, 3-step, and 4-step chains
            for combination in itertools.permutations(variables, length):
                try:
                    # Start with the thrust values for all chains
                    predicted_values = thrust_values
                    for i in range(len(combination) - 1):
                        x_var = combination[i]
                        y_var = combination[i+1]
                        key = f'{x_var}_to_{y_var}'
                        predicted_values = poly_dict[key](predicted_values)
                    
                    # Get the final predicted values using the last step of the combination
                    final_step_key = f'{combination[-1]}_to_{target}'
                    predicted_values = poly_dict[final_step_key](predicted_values)
                    
                    # Interpolate true values for comparison using thrust_values
                    true_values = np.interp(thrust_values, thrust, data_dict[target])
                    
                    # Calculate MAPE
                    mape = calculate_mape(true_values, predicted_values)
                    
                    # Store the result
                    results.append({
                        'chain': ' -> '.join(combination + (target,)),
                        'mape': mape,
                        'polynomial_chain': combination + (target,)
                    })
                except Exception as e:
                    print(f"Error with combination: {' -> '.join(combination + (target,))}, {e}")

        # Sort the results by MAPE to find the best chain for the current target
        if results:
            sorted_results = sorted(results, key=lambda x: x['mape'])
            best_result = sorted_results[0]

            # Print the selected best function for the current target
            print(f"\tSelected best function for {target}: {best_result['chain']} with MAPE: {best_result['mape']:.2f}%")

            # Create the callable function chain based on the best result
            def create_callable_function(poly_dict, chain):
                def func(thrust_value):
                    value = thrust_value
                    for i in range(len(chain) - 1):
                        x_var = chain[i]
                        y_var = chain[i+1]
                        key = f'{x_var}_to_{y_var}'
                        value = poly_dict[key](value)
                    return value
                return func

            # Add the best function to the MotorData object
            func_name = f'thrust_to_{target}'
            motor_data_obj.add_polynomial(func_name, create_callable_function(poly_dict, best_result['polynomial_chain']))

    # Store the MotorData object in the dictionary
    motor_data_objects[(brand, motor, prop)] = motor_data_obj

# Function to compute total weight
def compute_total_weight(motor_weight, include_payload):
    total_weight = frame_weight + battery_weight + (number_of_motors * motor_weight)
    if include_payload:
        total_weight += payload_weight
    return total_weight

def compute_performance(val): # thrust g
    global avg_battery_voltage
    global number_of_motors
    global motor_sim_data

    rpm = motor_sim_data.thrust_to_rpm(val)
    power_per_motor = motor_sim_data.thrust_to_watts(val)
    current_per_motor = motor_sim_data.thrust_to_current(val)
    eff = motor_sim_data.thrust_to_efficiency(val)
    total_current = current_per_motor * number_of_motors
    throttle = motor_sim_data.thrust_to_throttle(val)

    return power_per_motor, current_per_motor, eff, total_current, throttle
    #print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

# Step 1: Read the motor data from the CSV file
motor_data = pd.read_csv('motordata.csv')

# Battery parameters
batt_s = 6   # Number of series cells
batt_p = 2   # Number of parallel cells
cell_n = batt_s * batt_p
cell_w = float(70)  # Weight per cell in grams
battery_weight = cell_w * cell_n
cell_c = 10         # C-rating of the cells
cell_cap = float(4.5) # Capacity per cell in Amp-hours
battery_capacity = cell_cap * batt_p  # Total battery capacity in Amp-hours
max_battery_current = cell_c * cell_cap * batt_p  # Maximum battery current in Amps
max_battery_voltage = 4.2 * batt_s
avg_battery_voltage = 3.8 * batt_s
print(f"Battery {batt_s}S{batt_p}P {battery_capacity*1000} mAh {max_battery_current} A {battery_weight} g")

# Step 2: Define the input variables
#print("Using the following input variables:")
frame_weight = float(500)    # Frame weight in grams
payload_weight = float(2500)  # Payload weight in grams
max_esc_current = float(60)  # Maximum ESC current in Amps
thrust_weight_ratio = float(2)  # Necessary thrust-to-weight ratio
print("Frame weight:",frame_weight)
print("Payload weight:",payload_weight)
print("Maximum ESC current:",max_esc_current)
print("Minimum Thrust/Weight ratio:",thrust_weight_ratio)


max_engine_size = 31  # Maximum allowed engine stator diameter in mm
number_of_motors = 4 # Assume number of motors (default to quadcopter)

# Speed in km/h (convert to m/s for calculations)
flight_speed_kmh = 50  # Flight speed in km/h
flight_speed = flight_speed_kmh * (1000/3600)  # Convert km/h to m/s

# Safety margin and loiter time
safety_margin = 0.1  # 10% safety margin
loiter_time_min = 1  # Loaded hover time in minutes

valid_combinations = []

# Brand	Motor	Weight g	KV	Prop	T	V	A	Thrust g	Efficiency	W	RPM	Min Prop	Max Prop	Stator Diameter	Stator height	Data P size	Data P pitch	Prop area	Stator Vol	Stator Volume/Prop Area ratio
for (brand, motor, prop), group in motor_data.groupby(['Brand', 'Motor', 'Prop']):
    motor_weight = group['Weight g'].iloc[0]
    stator_diameter = group['Stator Diameter'].iloc[0]
    stator_height = group['Stator height'].iloc[0]
    kv = group['KV'].iloc[0]
    prop_diameter = group['Data P size'].iloc[0]
    prop_pitch = group['Data P pitch'].iloc[0]
    motor_sim_data = motor_data_objects[(brand, motor, prop)]
    
    
    # Skip motors larger than the maximum allowed size
    if stator_diameter > max_engine_size:
        continue
    
    # Fit quadratic polynomials
    throttle = group['T'].str.rstrip('%').astype(float).values  # Convert 'T' column to float
    thrust = group['Thrust g'].values
    current = group['A'].values
    rpm = group['RPM'].values
    watts = group['W'].values
    efficiency = group['Efficiency'].values
    maxt = thrust[len(thrust)-1]
    print(f"{brand} {motor} {prop} {kv}KV Max thrust: {thrust[len(thrust)-1]}")

    # Format results
    #print(f"\tStator Volume: {stator_volume:.2f} mm³\n\tProp Area: {prop_area:.2f} sq/in\n\tV/A Ratio: {volume_to_area_ratio:.2f} mm³/in²")
    #print(f"\tProp Tip Speed MPH: {tip_speed_mph:.2f}\n\tMach: {mach_number:.2f}")

    # Generate throttle values from min to max in 1% increments
    throttle_values = np.arange(min(throttle), max(throttle)+1, 1)

    # Initialize flags and variables
    meets_all_requirements = True
    best_option = None

    # For both loaded and unloaded conditions
    total_weights = {}
    throttle_settings = {}
    current_draws = {}

    for include_payload in [True, False]:
        condition = 'Loaded' if include_payload else 'Unloaded'
        total_weight = compute_total_weight(motor_weight, include_payload)

        # **Compute necessary thrust for thrust-to-weight ratio**
        necessary_tw_thrust = total_weight * thrust_weight_ratio  # Total necessary thrust
        required_tw_thrust_per_motor = necessary_tw_thrust / number_of_motors

        # Find throttle setting where thrust equals required_tw_thrust_per_motor
        if maxt < required_tw_thrust_per_motor:
            print(f'Not enough thrust ({maxt} vs {required_tw_thrust_per_motor})')
            meets_all_requirements = False
            break

        # minimum RPM that meets the requirement
        tw_rpm = motor_sim_data.thrust_to_rpm(required_tw_thrust_per_motor)
        
        print(f'\t {condition}')
        print(f"\t\tfor thrust {round(necessary_tw_thrust)} RPM {round(tw_rpm)}")

        # Calculate required current at this RPM
        #### 
        tw_power_per_motor, tw_current_per_motor, tw_eff, total_current_tw, throttle_tw = compute_performance(required_tw_thrust_per_motor)
        print(f"\t\ttw {round(thrust_weight_ratio)}:1 {round(tw_power_per_motor)} W, {round(tw_eff,3)} eff, {round(tw_current_per_motor)} A/4, {round(total_current_tw)} A, {round(throttle_tw)}% throt")

        # **Check current limits**
        if tw_current_per_motor > max_esc_current or total_current_tw > max_battery_current:
            meets_all_requirements = False
            print('TW Over current')
            break


        # **Proceed to calculate hover and flight settings**
        # Required thrust per motor for hover is total weight divided by number of motors
        required_thrust_per_motor_hover = total_weight / number_of_motors

        # Find throttle setting where thrust equals the weight per motor (hover)
        hover_rpm = motor_sim_data.thrust_to_rpm(required_thrust_per_motor_hover)

        # Calculate current draw at hover throttle setting
        #####
        hover_power_per_motor, hover_current_per_motor, hover_eff, total_current_hover, throttle_hover = compute_performance(required_thrust_per_motor_hover)
        print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

        # For flight, calculate required thrust per motor at 30-degree pitch angle
        required_thrust_per_motor_flight = required_thrust_per_motor_hover / math.cos(math.radians(30))
        flight_rpm = motor_sim_data.thrust_to_rpm(required_thrust_per_motor_flight)

        # Calculate current draw at flight throttle setting
        flight_power_per_motor, current_per_motor_flight, flight_eff, total_current_flight, throttle_flight = compute_performance(required_thrust_per_motor_flight)
        print(f"\t\tflight {round(flight_power_per_motor)} W, {round(flight_eff,3)} eff, {round(current_per_motor_flight)} A/4, {round(total_current_flight)} A, {round(throttle_flight)}% throt")

        # Store the data
        total_weights[condition] = total_weight
        throttle_settings[condition] = {
            'Max': throttle_tw,
            'TW': throttle_tw,
            'Hover': throttle_hover,
            'Flight': throttle_flight
        }
        current_draws[condition] = {
            'Max': total_current_tw,
            'TW': throttle_tw,
            'Hover': total_current_hover,
            'Flight': total_current_flight
        }

    if meets_all_requirements and len(current_draws) == 2:
        # Calculate total flight times for loaded and unloaded conditions at flight throttle
        flight_time_loaded = battery_capacity * 60 / current_draws['Loaded']['Flight']  # in minutes
        flight_time_unloaded = battery_capacity * 60 / current_draws['Unloaded']['Flight']  # in minutes

        # **Iterative Calculation of Maximum Delivery Radius**
        # Initialize variables
        max_distance = 0
        max_flight_time = 0

        # Total battery capacity in Amp-minutes
        battery_capacity_Am = battery_capacity * 60

        # Iterate over possible outbound flight times (in minutes)
        for t_out in np.arange(1, flight_time_loaded - loiter_time_min, 0.1):
            # Energy consumed during outbound flight, loiter, and return flight
            E_out = current_draws['Loaded']['Flight'] * t_out
            E_loiter = current_draws['Loaded']['Hover'] * loiter_time_min
            E_return = current_draws['Unloaded']['Flight'] * t_out  # Assuming same time for return flight

            total_energy = E_out + E_loiter + E_return

            # Check if total energy is within battery capacity minus safety margin
            if total_energy <= battery_capacity_Am * (1 - safety_margin):
                max_flight_time = t_out
                radius = flight_speed_kmh * (t_out / 60)  # Convert time to hours for distance
            else:
                break  # Exceeded battery capacity, stop iteration
        max_distance = flight_speed_kmh * (flight_time_loaded/60.0)
        # Store the best option with shortened variable names
        best_option = {
            'Motor': motor,
            'Prop': prop,
            'Loaded W': total_weights['Loaded'],
            'Clean W': total_weights['Unloaded'],
            'Throt (Max)': f"{throttle_settings['Loaded']['Max']:.1f}%",
            'Throt (L-H)': f"{throttle_settings['Loaded']['Hover']:.1f}%",
            'Throt (L-F)': f"{throttle_settings['Loaded']['Flight']:.1f}%",
            'Throt (U-F)': f"{throttle_settings['Unloaded']['Flight']:.1f}%",
            'Cur A (L-H)': current_draws['Loaded']['Hover'],
            'Cur A (L-F)': current_draws['Loaded']['Flight'],
            'Cur A (U-F)': current_draws['Unloaded']['Flight'],
            'Flight min (L-F)': flight_time_loaded,
            'Flight min (U-F)': flight_time_unloaded,
            'Max Outbound min': max_flight_time,
            'Loiter min': loiter_time_min,
            'Radius km': radius,
            'Max range km': max_distance
        }

        valid_combinations.append(best_option)

# Narrow down to top 5 options based on delivery radius
if valid_combinations:
    df_valid = pd.DataFrame(valid_combinations)
    df_top10 = df_valid.nlargest(10, 'Radius km', keep='first')
    print("\nTop 10 Motor-Propeller Combinations Based on Delivery Radius:")
    print(df_top10)
else:
    print("No suitable motor-propeller combinations found that fulfill the requirements.")
