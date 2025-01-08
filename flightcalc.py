import pandas as pd
import math
import numpy as np

# Function to compute total weight
def compute_total_weight(motor_weight, include_payload):
    total_weight = frame_weight + battery_weight + (number_of_motors * motor_weight)
    if include_payload:
        total_weight += payload_weight
    return total_weight

def compute_performance(val): # thrust g
    global avg_battery_voltage
    global number_of_motors

    rpm = poly_thrust_rpm(val)
    power_per_motor = poly_thrust_power(val)
    current_per_motor = power_per_motor / avg_battery_voltage
    eff = poly_thrust_eff(val)
    total_current = current_per_motor * number_of_motors
    throttle = poly_thrust_throttle(val)

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
frame_weight = float(400)    # Frame weight in grams
payload_weight = float(1000)  # Payload weight in grams
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
loiter_time_min = 2  # Loaded hover time in minutes

valid_combinations = []

# Brand	Motor	Weight g	KV	Prop	T	V	A	Thrust g	Efficiency	W	RPM	Min Prop	Max Prop	Stator Diameter	Stator height	Data P size	Data P pitch	Prop area	Stator Vol	Stator Volume/Prop Area ratio
for (brand, motor, prop), group in motor_data.groupby(['Brand', 'Motor', 'Prop']):
    motor_weight = group['Weight g'].iloc[0]
    stator_diameter = group['Stator Diameter'].iloc[0]
    stator_height = group['Stator height'].iloc[0]
    kv = group['KV'].iloc[0]
    prop_diameter = group['Data P size'].iloc[0]
    prop_pitch = group['Data P pitch'].iloc[0]
    
    
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
    print(f"{brand} {motor} {prop} {kv}KV Max thrust: {thrust[len(thrust)-1]}")

    # Fit polynomials
    coeffs_thrust = np.polyfit(throttle, thrust, 2)
    coeffs_current = np.polyfit(throttle, current, 2)
    coeffs_throt_rpm = np.polyfit(throttle, rpm, 2)
    coeffs_thrust_rpm = np.polyfit(thrust, rpm, 2)
    coeffs_thrust_throt = np.polyfit(thrust, throttle, 2)
    coeffs_thrust_eff = np.polyfit(thrust, efficiency, 2)
    coeffs_rpm_thrust = np.polyfit(rpm, thrust, 2)
    coeffs_rpm_eff = np.polyfit(rpm, efficiency, 2)
    coeffs_rpm_throt = np.polyfit(rpm, throttle, 2)
    coeffs_thrust_current = np.polyfit(thrust, current, 2)
    coeffs_thrust_power = np.polyfit(thrust, watts, 2)

    # Define polynomial functions
    poly_throt_thrust = np.poly1d(coeffs_thrust)
    poly_throt_current = np.poly1d(coeffs_current)
    poly_throt_rpm = np.poly1d(coeffs_throt_rpm)
    poly_thrust_rpm = np.poly1d(coeffs_thrust_rpm)
    poly_thrust_eff = np.poly1d(coeffs_thrust_eff)
    poly_thrust_current = np.poly1d(coeffs_thrust_current)
    poly_thrust_throttle = np.poly1d(coeffs_thrust_throt)
    poly_thrust_power = np.poly1d(coeffs_thrust_power)
    poly_rpm_thrust = np.poly1d(coeffs_rpm_thrust)
    poly_rpm_eff = np.poly1d(coeffs_rpm_eff)
    poly_rpm_throt = np.poly1d(coeffs_rpm_throt)

    maxrpm = poly_throt_rpm(100)
    maxt = poly_rpm_thrust(maxrpm)
    maxw = poly_thrust_power(maxt)
    maxa = maxw / avg_battery_voltage#poly_rpm_eff(maxrpm)
    devt =  abs(((maxt / thrust[len(thrust)-1]) * 100) - 100)
    deva =  abs(((maxa / current[len(current)-1]) * 100) - 100)
    print(f"\tRPM: {round(maxrpm)}, Thrust: {round(maxt)}g, {round(maxw)}W, {round(maxa)}A")
    print(f"\tT Deviation: {round(devt,2)}%, A Deviation: {round(deva,2)}%")


    rpm = maxrpm#kv * avg_battery_voltage

    # Calculate tip speed in surface feet per minute (SFPM)
    tip_speed_sfpm = (math.pi * prop_diameter * rpm) / 12  # Convert inches to feet

    # Convert SFPM to miles per hour (mph)
    tip_speed_mph = tip_speed_sfpm * 0.011363636  # Conversion factor from SFPM to mph

    # Calculate tip speed in Mach number
    mach_number = tip_speed_mph / 767  # Speed of sound at sea level in mph

    # Calculate stator volume
    stator_volume = math.pi * (stator_diameter / 2) ** 2 * stator_height

    # Calculate propeller area
    prop_area = math.pi * (prop_diameter / 2) ** 2

    # Calculate volume to area ratio (mm^3 per square inch)
    volume_to_area_ratio = stator_volume / prop_area  # No need for conversion

    # Calculate motor size (stator diameter * stator height)
    motor_size = f"{int(stator_diameter)}{'0' if stator_height < 10 else ''}{int(stator_height)}"

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
        print(f'\t {condition}')
        total_weight = compute_total_weight(motor_weight, include_payload)

        # **Compute necessary thrust for thrust-to-weight ratio**
        necessary_tw_thrust = total_weight * thrust_weight_ratio  # Total necessary thrust
        required_tw_thrust_per_motor = necessary_tw_thrust / number_of_motors

        # Find throttle setting where thrust equals required_tw_thrust_per_motor
        if maxt < required_tw_thrust_per_motor:
            meets_all_requirements = False
            break

        # minimum RPM that meets the requirement
        tw_rpm = poly_thrust_rpm(required_tw_thrust_per_motor)
        print(f"\t\tfor thrust {round(necessary_tw_thrust)} RPM {round(tw_rpm)}")

        # Calculate required current at this RPM
        #### 
        tw_power_per_motor, tw_current_per_motor, tw_eff, total_current_tw, throttle_tw = compute_performance(required_tw_thrust_per_motor)
        print(f"\t\ttw {round(thrust_weight_ratio)}:1 {round(tw_power_per_motor)} W, {round(tw_eff,3)} eff, {round(tw_current_per_motor)} A/4, {round(total_current_tw)} A, {round(throttle_tw)}% throt")

        # **Check current limits**
        if tw_current_per_motor > max_esc_current or total_current_tw > max_battery_current:
            meets_all_requirements = False
            break


        # **Proceed to calculate hover and flight settings**
        # Required thrust per motor for hover is total weight divided by number of motors
        required_thrust_per_motor_hover = total_weight / number_of_motors

        # Find throttle setting where thrust equals the weight per motor (hover)
        hover_rpm = poly_thrust_rpm(required_thrust_per_motor_hover)

        # Calculate current draw at hover throttle setting
        #####
        hover_power_per_motor, hover_current_per_motor, hover_eff, total_current_hover, throttle_hover = compute_performance(required_thrust_per_motor_hover)
        print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

        #current_per_motor_hover = poly_current(throttle_hover)
        #total_current_hover = current_per_motor_hover * number_of_motors

        # Check if current draw is within limits
        if hover_current_per_motor > max_esc_current or total_current_hover > max_battery_current:
            meets_all_requirements = False
            break

        # For flight, calculate required thrust per motor at 30-degree pitch angle
        required_thrust_per_motor_flight = required_thrust_per_motor_hover / math.cos(math.radians(30))
        flight_rpm = poly_thrust_rpm(required_thrust_per_motor_flight)

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
                max_distance = flight_speed_kmh * (t_out / 60)  # Convert time to hours for distance
            else:
                break  # Exceeded battery capacity, stop iteration

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
            'Radius km': max_distance
        }

        valid_combinations.append(best_option)

# Narrow down to top 5 options based on delivery radius
if valid_combinations:
    df_valid = pd.DataFrame(valid_combinations)
    df_top10 = df_valid.sort_values(by='Radius km', ascending=False).head(10)
    print("\nTop 10 Motor-Propeller Combinations Based on Delivery Radius:")
    print(df_top10)
else:
    print("No suitable motor-propeller combinations found that fulfill the requirements.")
