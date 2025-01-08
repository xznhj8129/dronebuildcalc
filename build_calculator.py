import pandas as pd
import math
import numpy as np
import itertools
from matplotlib import pyplot as plt 

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

        motor_data = pd.read_csv('motordata.csv')
        group = motor_data[(motor_data['Brand'] == brand) & 
            (motor_data['Motor'] == motor) & 
            (motor_data['Prop'] == prop)]

        #print(f"Processing {brand} {motor} {prop}")
        #print(group)
        if group.empty:
            print(f"No data found for {brand} {motor} {prop}")
            exit(0)
            
        self.throttle = group['T'].str.rstrip('%').astype(float).values
        self.thrust = group['Thrust g'].values
        self.max_thrust = self.thrust[len(self.thrust)-1]
        self.current = group['A'].values
        self.rpm = group['RPM'].values
        self.watts = group['W'].values
        self.motor_batt_s = group['Data Batt S'].iloc[0]
        self.efficiency = group['Efficiency'].values
        self.motor_weight = group['Weight g'].iloc[0]
        self.stator_diameter = group['Stator Diameter'].iloc[0]
        self.stator_height = group['Stator height'].iloc[0]
        self.kv = group['KV'].iloc[0]
        self.prop_diameter = group['Data P size'].iloc[0]
        self.prop_pitch = group['Data P pitch'].iloc[0]

        thrust_values = np.arange(50, self.max_thrust, 10)

        poly_dict = {}

        data_dict = {
            'throttle': self.throttle,
            'thrust': self.thrust,
            'current': self.current,
            'rpm': self.rpm,
            'watts': self.watts,
            'efficiency': self.efficiency
        }

        # Generate polynomial fit functions for every combination of variables
        for (x_var, y_var) in itertools.permutations(data_dict.keys(), 2):
            x_data = data_dict[x_var]
            y_data = data_dict[y_var]
            coeffs = np.polyfit(x_data, y_data, 2)
            poly_func = np.poly1d(coeffs)
            poly_dict[f'{x_var}_to_{y_var}'] = poly_func

        #motor_data_obj = MotorData(brand, motor, prop)
        target_variables = ['rpm', 'current', 'watts', 'efficiency', 'throttle']

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

                        final_step_key = f'{combination[-1]}_to_{target}'
                        predicted_values = poly_dict[final_step_key](predicted_values)
                        true_values = np.interp(thrust_values, self.thrust, data_dict[target])
                        mape = calculate_mape(true_values, predicted_values)
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
                self.add_polynomial(func_name, create_callable_function(poly_dict, best_result['polynomial_chain']))

                # Plot predicted vs true values
                predicted_values = self.__getattr__(f'thrust_to_{target}')(thrust_values)
                true_values = np.interp(thrust_values, self.thrust, data_dict[target])

                # Plot the data
                plt.figure(figsize=(8, 6))
                plt.plot(thrust_values, true_values, 'o-', label='True Values', color='blue')
                plt.plot(thrust_values, predicted_values, 'x-', label='Predicted Values', color='red')
                plt.title(f'Predicted vs True Values for {target} ({brand} {motor} {prop})')
                plt.xlabel('Thrust (g)')
                plt.ylabel(target.capitalize())
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{brand} {motor} {prop} {target}.png')
                #plt.show()

    def add_polynomial(self, name, func):
        self.polynomials[name] = func

    def __getattr__(self, name):
        # Dynamically call the polynomial stored in the object
        if name in self.polynomials:
            return self.polynomials[name]
        raise AttributeError(f"{name} not found in MotorData.")

    def motor_perf_thrust(self, val): # thrust g
        global avg_battery_voltage
        global number_of_motors

        rpm = self.thrust_to_rpm(val)
        power_per_motor = self.thrust_to_watts(val)
        current_per_motor = self.thrust_to_current(val)
        eff = self.thrust_to_efficiency(val)
        total_current = current_per_motor * number_of_motors
        throttle = self.thrust_to_throttle(val)

        return power_per_motor, current_per_motor, eff, total_current, throttle
        #print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

    def drone_weight_perf(self, weight):
        throttle_values = np.arange(min(self.throttle), max(self.throttle)+1, 1)
        throttle_settings = {}
        current_draws = {}
        meets_all_requirements = True
        total_weight = weight + battery_weight + total_motors_weight

        #print()
        #print("Frame weight:",weight)
        #print("Motors weight:",total_motors_weight)
        #print("Maximum ESC current:",max_esc_current)
        #print("Minimum Thrust/Weight ratio:",thrust_weight_ratio)

        # **Compute necessary thrust for thrust-to-weight ratio**
        necessary_tw_thrust = total_weight * thrust_weight_ratio  # Total necessary thrust
        required_tw_thrust_per_motor = necessary_tw_thrust / number_of_motors

        # Find throttle setting where thrust equals required_tw_thrust_per_motor
        if self.max_thrust < required_tw_thrust_per_motor:
            #print(f'Not enough thrust ({self.max_thrust} vs {required_tw_thrust_per_motor})')
            meets_all_requirements = False
            return None

        # minimum RPM that meets the requirement
        tw_rpm = self.thrust_to_rpm(required_tw_thrust_per_motor)
        #print(f"\tfor thrust {round(necessary_tw_thrust/4)} RPM {round(tw_rpm)}")
        tw_power_per_motor, tw_current_per_motor, tw_eff, total_current_tw, throttle_tw = self.motor_perf_thrust(required_tw_thrust_per_motor)
        #print(f"\ttw {round(thrust_weight_ratio)}:1 {round(tw_power_per_motor)} W, {round(tw_eff,3)} eff, {tw_current_per_motor:.2f} A/4, {total_current_tw:.2f} A, {round(throttle_tw)}% throt")

        if tw_current_per_motor > max_esc_current or total_current_tw > max_battery_current:
            meets_all_requirements = False
            #print('TW Over current')
            return None

        required_thrust_per_motor_hover = total_weight / number_of_motors
        hover_rpm = self.thrust_to_rpm(required_thrust_per_motor_hover)
        hover_power_per_motor, hover_current_per_motor, hover_eff, total_current_hover, throttle_hover = self.motor_perf_thrust(required_thrust_per_motor_hover)
        #print(f"\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {hover_current_per_motor:.2f}, A/4 {total_current_hover:.2f}, A {round(throttle_hover)}% throt")

        # For flight, calculate required thrust per motor at 30-degree pitch angle
        required_thrust_per_motor_flight = required_thrust_per_motor_hover / math.cos(math.radians(30))
        flight_rpm = self.thrust_to_rpm(required_thrust_per_motor_flight)
        flight_power_per_motor, current_per_motor_flight, flight_eff, total_current_flight, throttle_flight = self.motor_perf_thrust(required_thrust_per_motor_flight)
        #print(f"\tflight {round(flight_power_per_motor)} W, {round(flight_eff,3)} eff, {current_per_motor_flight:.2f} A/4, {total_current_flight:.2f} A, {round(throttle_flight)}% throt")

        throttle_settings = {
            'TW': throttle_tw,
            'Hover': throttle_hover,
            'Flight': throttle_flight
        }
        current_draws = {
            'TW': total_current_tw,
            'Hover': total_current_hover,
            'Flight': total_current_flight
        }

        flight_time = battery_capacity * 60 / current_draws['Flight']  # in minutes
        hover_time = battery_capacity * 60 / current_draws['Hover']  # in minutes
        tw_time = battery_capacity * 60 / current_draws['TW']  # in minutes
        max_distance = 0
        max_flight_time = 0
        battery_capacity_Am = battery_capacity * 60

        # Iterate over possible outbound flight times (in minutes)
        for t_out in np.arange(1, flight_time, 0.1):
            # Energy consumed during outbound flight, loiter, and return flight
            E_out = current_draws['Flight'] * t_out
            E_loiter = current_draws['Hover'] * loiter_time_min
            E_return = current_draws['Flight'] * t_out  # Assuming same time for return flight
            total_energy = E_out + E_loiter + E_return

            # Check if total energy is within battery capacity minus safety margin
            if total_energy <= battery_capacity_Am * (1 - safety_margin):
                max_flight_time = t_out
                radius = flight_speed_kmh * (t_out / 60)  # Convert time to hours for distance
            else:
                break  # Exceeded battery capacity, stop iteration
        max_distance = flight_speed_kmh * (flight_time/60.0)

        #print("Performance:")
        #print("Total weight:",total_weight)
        #print(f"Hover:\t\t {round(throttle_hover)}% throt\t {current_draws['Hover']:.2f} A\t {hover_time:.2f} min")
        #print(f"TW {round(thrust_weight_ratio)}:1 ratio:\t {round(throttle_tw)}% throt\t {current_draws['TW']:.2f} A\t {tw_time:.2f} min")
        #print(f"Flight 30deg:\t {round(throttle_flight)}% throt\t {current_draws['Flight']:.2f} A\t {flight_time:.2f} min")
        #print(f"Max distance at {flight_speed_kmh}km/h: {max_distance:.2f} km")
        #print(f'For {loiter_time_min:.2f} min loiter time:')
        #print(f'Max Outbound time: {max_flight_time:.2f} min')
        #print(f'Max flight radius: {radius:.2f} km')
        return{
            'frame_weight': weight,
            "total_weight": total_weight,
            'throt_h':throttle_hover,
            'throt_f': throttle_flight,
            'throt_max': throttle_tw,
            'rpm_h': hover_rpm,
            'rpm_f': flight_rpm,
            'rpm_tw': tw_rpm,
            'time_h': hover_time,
            'time_f': flight_time,
            'time_m': tw_time,
            'eff_h': hover_eff,
            'eff_f': flight_eff,
            'eff_tw': tw_eff,
            'ah': current_draws['Hover'],
            'af': current_draws['Flight'],
            'am': current_draws['TW'],
            'maxd': max_distance,
            'radius': radius,
        }


#brand, motor, prop = ('Brotherhobby', 'F2004', 'F6030X2')
#brand, motor, prop = ('Brotherhobby', 'F2008', 'F9453X2')
brand, motor, prop = ('Brotherhobby', 'F1503', 'F4726X2')

motor_model = MotorData(brand, motor, prop)

max_esc_current = float(60)  # Maximum ESC current in Amps
thrust_weight_ratio = float(2)  # Necessary thrust-to-weight ratio
batt_s = motor_model.motor_batt_s   # Number of series cells
batt_p = 1   # Number of parallel cells
cell_w = float(70)  # Weight per cell in grams
cell_c = 12         # C-rating of the cells
cell_cap = float(3) # Capacity per cell in Amp-hours

cell_n = batt_s * batt_p
battery_capacity = cell_cap * batt_p  # Total battery capacity in Amp-hours
max_battery_current = cell_c * cell_cap * batt_p  # Maximum battery current in Amps
max_battery_voltage = 4.2 * batt_s
avg_battery_voltage = 3.8 * batt_s
battery_weight = cell_w * cell_n
number_of_motors = 4 # Assume number of motors (default to quadcopter)
total_motors_weight = motor_model.motor_weight * number_of_motors

# Speed in km/h (convert to m/s for calculations)
flight_speed_kmh = 50  # Flight speed in km/h
flight_speed = flight_speed_kmh * (1000/3600)  # Convert km/h to m/s

# Safety margin and loiter time
safety_margin = 0.1  # 10% safety margin
loiter_time_min = 10  # Loaded hover time in minutes

print()
print('#'*32)
print('Results')
print('#'*32)
print(f"Battery {batt_s}S{batt_p}P {battery_capacity*1000} mAh {max_battery_current} A {battery_weight}g")
print(f"{brand} {motor} {prop} {motor_model.kv}KV Max thrust: {motor_model.max_thrust}g")
print("Total motors weight:", total_motors_weight,'g')
print(f"Loiter time:", loiter_time_min,"min")

mot_frame_w_ratio = 0.30
bat_total_w_ratio = 0.32

est_frame_w = total_motors_weight/mot_frame_w_ratio
est_mtow_from_frame = est_frame_w + total_motors_weight + battery_weight
#est_mtow_from_batt = battery_weight/bat_total_w_ratio

#avg_est_mtow = (est_mtow_from_frame + est_mtow_from_batt) / 2.0
#avg_est_frame_w = avg_est_mtow - battery_weight - total_motors_weight
p = motor_model.drone_weight_perf(est_frame_w)

print(f"Estimated frame weight for {mot_frame_w_ratio*100} % Motor/Frame ratio: {est_frame_w:.2f}g, Total: {est_mtow_from_frame:.2f}g")
#print(f"Estimated MTOW for {bat_total_w_ratio*100} % Battery/MTOW ratio: {est_mtow_from_batt:.2f}g")
#print(f"Median estimated MTOW: {avg_est_mtow:.2f}g")    
#print(f"Estimated median frame weight: {avg_est_frame_w:.2f}g ")
print(f"Hover time: {p['time_h']:.2f} minutes")
print(f"Flight time: {p['time_f']:.2f} minutes")
print(f"Max throttle time: {p['time_m']:.2f} minutes")
print()

results = []
frame_weight = float(100)    # (minimum) frame weight in grams
for frame_w in np.arange(50, 5000, 1):
    p = motor_model.drone_weight_perf(frame_w)
    if p: 
        results.append(p)
    else: break
maxw = results[len(results)-1]
print()
print(f"Maximum weight")
print(f"Highest Frame weight: {maxw['frame_weight']}g")
print(f"Highest MTOW: {maxw['total_weight']}g")
print(f"Hover time: {maxw['time_h']:.2f} minutes")
print(f"Flight time: {maxw['time_f']:.2f} minutes")
print(f"Max throttle time: {maxw['time_m']:.2f} minutes")
print(f"Battery/MTOW weight ratio: {battery_weight/maxw['total_weight']*100:.2f} %")
print(f"Motor/frame weight ratio: {total_motors_weight/maxw['frame_weight']*100:.2f} %")
print()

thresholds = [60, 40, 30, 20]
df = pd.DataFrame(results)
filtered_df = df[df['total_weight'] < 250]
if not filtered_df.empty:
    frame_weight_row = filtered_df.loc[filtered_df['frame_weight'].idxmax()]
    
    print(f"Maximum 249g")
    print(f"Highest Frame weight: {frame_weight_row['frame_weight']}g")
    print(f"Highest MTOW: {frame_weight_row['total_weight']:.2f}g")
    print(f"Hover time: {frame_weight_row['time_h']:.2f} minutes")
    print(f"Flight time: {frame_weight_row['time_f']:.2f} minutes")
    print(f"Max throttle time: {frame_weight_row['time_m']:.2f} minutes")
    mfr = total_motors_weight/frame_weight_row['frame_weight']
    bmwr = battery_weight/frame_weight_row['total_weight']
    print(f"Motor/frame weight ratio: {mfr*100:.2f} %")
    print(f"Battery/MTOW weight ratio: {bmwr*100:.2f} %")
    print()
else:
    print('Cannot meet <249g\n')

for threshold in thresholds:
    filtered_df = df[df['time_f'] > threshold]

    if not filtered_df.empty:
        frame_weight_row = filtered_df.loc[filtered_df['frame_weight'].idxmax()]
        
        print(f"Threshold: {threshold} minutes")
        print(f"Highest Frame weight: {frame_weight_row['frame_weight']}g")
        print(f"Highest MTOW: {frame_weight_row['total_weight']}g")
        print(f"Hover time: {frame_weight_row['time_h']:.2f} minutes")
        print(f"Flight time: {frame_weight_row['time_f']:.2f} minutes")
        print(f"Max throttle time: {frame_weight_row['time_m']:.2f} minutes")
        mfr = total_motors_weight/frame_weight_row['frame_weight']
        bmwr = battery_weight/frame_weight_row['total_weight']
        print(f"Motor/frame weight ratio: {mfr*100:.2f} %")
        print(f"Battery/MTOW weight ratio: {bmwr*100:.2f} %")
        print()



weights = [result['total_weight'] for result in results]
flight_times = [result['time_f'] for result in results] 
plt.figure(figsize=(8, 6))
plt.plot(weights, flight_times, label='MTOW vs Flight time')
plt.title('MTOW vs Flight Time')
plt.xlabel('MTOW (g)')
plt.ylabel('Flight Time (min)')
plt.grid(True)
plt.legend()
plt.savefig(f'{brand} {motor} {prop} MTOW vs Flight time.png')
plt.show()