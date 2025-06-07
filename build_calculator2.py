import pandas as pd
import math
import numpy as np
import itertools
from matplotlib import pyplot as plt 

class MotorData:
    def __init__(self, brand, motor, prop, kv, serie):
        self.brand = brand
        self.motor = motor
        self.prop = prop

        motor_data = pd.read_csv('motordata.csv')
        group = motor_data[(motor_data['Brand'] == brand) & 
                           (motor_data['Motor'] == motor) & 
                           (motor_data['Prop'] == prop) & 
                           (motor_data['KV'] == kv) & 
                           (motor_data['Data Batt S'] == serie)]

        if group.empty:
            print(f"No data found for {brand} {motor} {prop} {kv} {serie}S")
            exit(0)
            
        self.throttle = group['T'].str.rstrip('%').astype(float).values
        self.thrust = group['Thrust g'].values
        self.max_thrust = self.thrust[len(self.thrust)-1]
        self.current = group['A'].values
        self.max_draw = self.current[len(self.current)-1]
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
        for i in ['throttle','current','rpm','watts','efficiency']:
            self.plot_thrust_to(i, thrust_values)

    # Interpolation function for each variable
    def thrust_to_throttle(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.throttle)

    def thrust_to_current(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.current)

    def thrust_to_rpm(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.rpm)

    def thrust_to_watts(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.watts)

    def thrust_to_efficiency(self, thrust_values):
        return np.interp(thrust_values, self.thrust, self.efficiency)

    # Example plot function
    def plot_thrust_to(self, target_variable, thrust_values):
        true_values = np.interp(thrust_values, self.thrust, getattr(self, target_variable))
        predicted_values = getattr(self, f'thrust_to_{target_variable}')(thrust_values)

        # Plot true vs predicted values
        #plt.figure(figsize=(8, 6))
        #plt.plot(thrust_values, true_values, 'o-', label='True Values', color='blue')
        #plt.plot(thrust_values, predicted_values, 'x-', label='Interpolated Values', color='red')
        #plt.title(f'Interpolated vs True Values for {target_variable} ({self.brand} {self.motor} {self.prop})')
        #plt.xlabel('Thrust (g)')
        #plt.ylabel(target_variable.capitalize())
        #plt.legend()
        #plt.grid(True)
        #plt.savefig(f'{brand} {motor} {prop} {target_variable}.png')
        #plt.show()


    def motor_perf_thrust(self, val): # thrust g
        global avg_battery_voltage
        global number_of_motors
        if val > self.max_thrust:
            print('Error: over maximum thrust')
            return None
        rpm = self.thrust_to_rpm(val)
        power_per_motor = self.thrust_to_watts(val)
        current_per_motor = self.thrust_to_current(val)
        eff = self.thrust_to_efficiency(val)
        total_current = current_per_motor * number_of_motors
        throttle = self.thrust_to_throttle(val)
        #print(f'CALC FOR {val:.2f}g THRUST: {rpm:.2f} RPM {current_per_motor:.2f} A ({total_current:.2f} A total) {throttle:.2f}% THROT')

        return power_per_motor, current_per_motor, eff, total_current, throttle
        #print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

    def drone_weight_perf(self, weight, batt_weight, tot_motors_weight, batt_cap):
        throttle_values = np.arange(min(self.throttle), max(self.throttle)+1, 1)
        throttle_settings = {}
        current_draws = {}
        meets_all_requirements = True
        total_weight = weight + batt_weight + tot_motors_weight

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

        flight_time = batt_cap * 60 / current_draws['Flight']  # in minutes
        hover_time = batt_cap * 60 / current_draws['Hover']  # in minutes
        tw_time = batt_cap * 60 / current_draws['TW']  # in minutes
        max_distance = 0
        max_flight_time = 0
        batt_cap_Am = batt_cap * 60
        radius = 0
        max_flight_time = 0

        # Iterate over possible outbound flight times (in minutes)
        for t_out in np.arange(0.1, flight_time, 0.1):
            # Energy consumed during outbound flight, loiter, and return flight
            E_out = current_draws['Flight'] * t_out
            E_loiter = current_draws['Hover'] * loiter_time_min
            E_return = current_draws['Flight'] * t_out  # Assuming same time for return flight
            total_energy = E_out + E_loiter + E_return

            # Check if total energy is within battery capacity minus safety margin
            if total_energy <= batt_cap_Am * (1 - safety_margin):
                max_flight_time = t_out
                radius = flight_speed_kmh * (t_out / 60)  # Convert time to hours for distance
            else:
                break
        max_distance = flight_speed_kmh * (flight_time/60.0)

        print("Performance:")
        print("Total weight:",total_weight)
        print(f"Hover:\t\t {round(throttle_hover)}% throt\t {current_draws['Hover']:.2f} A\t {hover_time:.2f} min")
        print(f"TW {round(thrust_weight_ratio)}:1 ratio:\t {round(throttle_tw)}% throt\t {current_draws['TW']:.2f} A\t {tw_time:.2f} min")
        print(f"Flight 30deg:\t {round(throttle_flight)}% throt\t {current_draws['Flight']:.2f} A\t {flight_time:.2f} min")
        print(f"Max distance at {flight_speed_kmh}km/h: {max_distance:.2f} km")
        print(f'For {loiter_time_min:.2f} min loiter time:')
        print(f'Max Outbound time: {max_flight_time:.2f} min')
        print(f'Max flight radius: {radius:.2f} km')
        print()
        return{
            'error': None,
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
brand, motor, prop, kv, battS = ('Uangel', 'A2807', 'GF7035', 1300, 6)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F2008', 'F8330X2', 1400, 4)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F2008', 'F9453X2', 1000, 4)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F2008', 'F9453X2', 1000, 3)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F1503', 'F4726X2', 3850, 2)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F2206', 'F6030X2', 1370, 4)
#brand, motor, prop, kv, battS = ('Brotherhobby', 'F2216', 'F9453X2', 880, 4)


motor_model = MotorData(brand, motor, prop, kv, battS)


batt_format = 21700
batt_s = motor_model.motor_batt_s   # Number of series cells
batt_p = 1   # Number of parallel cells
cell_w = float(70)  # Weight per cell in grams (21700: 70, 18650: 45)
cell_c = 10         # C-rating of the cells
cell_cap = float(5) # Capacity per cell in Amp-hours

cell_n = batt_s * batt_p
battery_capacity = cell_cap * batt_p  # Total battery capacity in Amp-hours
max_battery_current = cell_c * cell_cap * batt_p  # Maximum battery current in Amps
max_battery_voltage = 4.2 * batt_s
avg_battery_voltage = 3.8 * batt_s
battery_weight = cell_w * cell_n

max_esc_current = float(60)  # Maximum ESC current in Amps
thrust_weight_ratio = float(2.5)  # Necessary thrust-to-weight ratio
number_of_motors = 4 # Assume number of motors (default to quadcopter)
total_motors_weight = motor_model.motor_weight * number_of_motors

# Speed in km/h (convert to m/s for calculations)
flight_speed_kmh = 50  # Flight speed in km/h
flight_speed = flight_speed_kmh * (1000/3600)  # Convert km/h to m/s

# Safety margin and loiter time
safety_margin = 0.1  # 10% safety margin
loiter_time_min = 1  # Loaded hover time in minutes
takeoff_weight = 1400.0 # 1P
drone_weight = takeoff_weight - battery_weight
frame_weight = drone_weight - total_motors_weight
required_thrust = (thrust_weight_ratio * takeoff_weight)
max_total_thrust = motor_model.max_thrust * number_of_motors

print()
print('#'*32)
print(f'Results for {takeoff_weight}g takeoff weight')
print('#'*32)
print(f"Motor: {brand} {motor} {prop} {motor_model.kv}KV Max thrust: {motor_model.max_thrust}g")
print(f"Total motors weight: {total_motors_weight}g")
print(f"Drone weight (no battery): {drone_weight}g")
print(f"Frame weight (no battery or motors): {frame_weight:.2f}g")
print(f"Maximum total thrust: {max_total_thrust}g")
print(f"Maximum total Amp draw (x1): {motor_model.max_draw*4} ({motor_model.max_draw}) A")

print('\n#########')
print(f"Battery: {batt_format} {batt_s}S1P {battery_capacity*1000} mAh {max_battery_current}A {battery_weight}g")
print("#########")
p = motor_model.drone_weight_perf(frame_weight, total_motors_weight, battery_weight, battery_capacity)

print(f"Required thrust for {thrust_weight_ratio:.1f}:1 thrust/weight ratio at {takeoff_weight}g TOW: {required_thrust:.2f}g")
print(f"Actual Thrust/Weight ratio: {max_total_thrust/takeoff_weight:.2f}:1")
if max_total_thrust < required_thrust:
    print(f"\nERROR: Insufficient thrust for thrust/weight ratio")
    exit(0)
print(f"Amp draw hover: {p['ah']:.2f} A")
print(f"Amp draw flight: {p['af']:.2f} A")
print(f"Hover time: {p['time_h']:.2f} minutes")
print(f"Flight time: {p['time_f']:.2f} minutes")
print(f"Max throttle time: {p['time_m']:.2f} minutes")
print()

p2 = motor_model.drone_weight_perf(frame_weight, total_motors_weight, battery_weight*2, battery_capacity*2)
if p2 is None:
    print("Error for 2P")
else:
    takeoff_weight = drone_weight + (battery_weight*2)
    required_thrust = (thrust_weight_ratio * takeoff_weight)
    print('\n#########')
    print(f"Battery: {batt_format} {batt_s}S2P {battery_capacity*2000} mAh {max_battery_current*2}A {battery_weight*2}g")
    print("#########")
    print(f"2P Takeoff weight: {takeoff_weight}g")
    print(f"Required thrust for {thrust_weight_ratio:.1f}:1 thrust/weight ratio at {takeoff_weight}g TOW: {required_thrust:.2f}g")
    print(f"Actual Thrust/Weight ratio: {max_total_thrust/takeoff_weight:.2f}:1")
    if max_total_thrust < required_thrust:
        print(f"\nERROR: Insufficient thrust for thrust/weight ratio")
        exit(0)
    print(f"Amp draw hover: {p2['ah']:.2f} A")
    print(f"Amp draw flight: {p2['af']:.2f} A")
    print(f"Hover time: {p2['time_h']:.2f} minutes")
    print(f"Flight time: {p2['time_f']:.2f} minutes")
    print(f"Max throttle time: {p2['time_m']:.2f} minutes")
    print()

def from_weight_ratio():
    mot_frame_w_ratio = 0.275
    bat_total_w_ratio = 0.32

    est_frame_w = total_motors_weight/mot_frame_w_ratio
    est_mtow_from_frame = est_frame_w + total_motors_weight + battery_weight
    #est_mtow_from_batt = battery_weight/bat_total_w_ratio

    #avg_est_mtow = (est_mtow_from_frame + est_mtow_from_batt) / 2.0
    #avg_est_frame_w = avg_est_mtow - battery_weight - total_motors_weight
    p = motor_model.drone_weight_perf(est_frame_w)

    print(f"Estimated frame weight for {mot_frame_w_ratio*100:.2f} % Motor/Frame ratio: {est_frame_w:.2f}g, Total: {est_mtow_from_frame:.2f}g")
    #print(f"Estimated MTOW for {bat_total_w_ratio*100} % Battery/MTOW ratio: {est_mtow_from_batt:.2f}g")
    #print(f"Median estimated MTOW: {avg_est_mtow:.2f}g")    
    #print(f"Estimated median frame weight: {avg_est_frame_w:.2f}g ")
    print(f"Amp draw hover: {p['ah']:.2f} A")
    print(f"Amp draw flight: {p['af']:.2f} A")
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
    print(f"Highest Frame weight: {maxw['frame_weight']:.2f}g")
    print(f"Highest MTOW: {maxw['total_weight']:.2f}g")
    #print(f"Thrust to Weight ratio: {}:{}")
    print(f"Amp draw hover: {maxw['ah']:.2f} A")
    print(f"Amp draw flight: {maxw['af']:.2f} A")
    print(f"Hover time: {maxw['time_h']:.2f} minutes")
    print(f"Flight time: {maxw['time_f']:.2f} minutes")
    print(f"Max throttle time: {maxw['time_m']:.2f} minutes")
    print(f"Battery/MTOW weight ratio: {battery_weight/maxw['total_weight']*100:.2f} %")
    print(f"Motor/frame weight ratio: {total_motors_weight/maxw['frame_weight']*100:.2f} %")
    print()

    df = pd.DataFrame(results)
    filtered_df = df[df['total_weight'] < 250]
    if not filtered_df.empty:
        frame_weight_row = filtered_df.loc[filtered_df['frame_weight'].idxmax()]
        
        print(f"Maximum 249g")
        print(f"Highest Frame weight: {frame_weight_row['frame_weight']}g")
        print(f"Highest MTOW: {frame_weight_row['total_weight']:.2f}g")
        print(f"Amp draw hover: {frame_weight_row['ah']:.2f} A")
        print(f"Amp draw flight: {frame_weight_row['af']:.2f} A")
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

    thresholds = [60, 50, 45, 40, 35, 30, 25, 20]
    for threshold in thresholds:
        filtered_df = df[df['time_f'] > threshold]

        if not filtered_df.empty:
            frame_weight_row = filtered_df.loc[filtered_df['frame_weight'].idxmax()]
            if frame_weight_row['frame_weight'] >= maxw['frame_weight']:
                break

            print(f"Threshold: {threshold} minutes")
            print(f"Highest Frame weight: {frame_weight_row['frame_weight']:.2f}g")
            print(f"Highest MTOW: {frame_weight_row['total_weight']:.2f}g")
            print(f"Amp draw hover: {frame_weight_row['ah']:.2f} A")
            print(f"Amp draw flight: {frame_weight_row['af']:.2f} A")
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