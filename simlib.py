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
        self.kv = int(kv)
        self.serie = int(serie)

        motor_data = pd.read_csv('motordata.csv')
        group = motor_data[(motor_data['Brand'] == self.brand) & 
                           (motor_data['Motor'] == self.motor) & 
                           (motor_data['Prop'] == self.prop) & 
                           (motor_data['KV'] == self.kv) & 
                           (motor_data['Data Batt S'] == self.serie)]

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
        global number_of_motors
        if val > self.max_thrust:
            print('Error: over maximum thrust')
            return None
        rpm = self.thrust_to_rpm(val)
        power_per_motor = self.thrust_to_watts(val)
        current_per_motor = self.thrust_to_current(val)
        eff = self.thrust_to_efficiency(val)
        throttle = self.thrust_to_throttle(val)
        #print(f'CALC FOR {val:.2f}g THRUST: {rpm:.2f} RPM {current_per_motor:.2f} A ({total_current:.2f} A total) {throttle:.2f}% THROT')

        return power_per_motor, current_per_motor, eff, throttle
        #print(f"\t\thover {round(hover_power_per_motor)} W, {round(hover_eff,3)} eff, {round(hover_current_per_motor)}, A/4 {round(total_current_hover)}, A {round(throttle_hover)}% throt")

class LiIonBatteryData:    
    def __init__(self, brand, model, cellformat, serie, parallel):
        self.brand = brand
        self.model = model
        self.format = int(cellformat)
        self.parallel = int(parallel)
        self.serie = int(serie)
        self.max_v = 4.2
        self.nominal_v = 3.6
        self.min_v = 2.5

        batt_data = pd.read_csv('batterydata.csv')
        group = batt_data[(batt_data['Brand'] == self.brand) & 
                           (batt_data['Name'] == self.model) & 
                           (batt_data['Format'] == self.format)]

        if group.empty:
            print(f"No data found for {brand} {model} {cellformat}")
            exit(0)
        self.cell_weight = group['Weight'].iloc[0] / 1000.0
        self.cell_c = group['C-rating'].iloc[0]
        self.cell_cap_a = group['CapacityA'].iloc[0]
        self.cell_cap_wh = group['CapacityWh'].iloc[0]
        self.pack_weight = self.cell_weight * serie * parallel
        self.pack_amps = self.cell_cap_a * parallel
        self.pack_wh = self.cell_cap_wh * parallel
        self.pack_current = self.cell_c * self.pack_amps
        self.cell_len = 0.070 if self.format=="21700" else 0.065
        self.cell_width = 0.021 if self.format=="21700" else 0.018 
        self.pack_len = self.cell_len * self.parallel
        self.pack_width = -1