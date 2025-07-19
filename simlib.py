import pandas as pd
import math
import numpy as np
import itertools
from matplotlib import pyplot as plt 
from scipy.interpolate import interp1d

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
        for i in range(1,50):
            mot_watt, mot_curr, mot_gW, mot_throt, mot_rpm = self.motor_perf_thrust(i*100)
            if mot_watt==math.inf:
                break
            print(f"Tg:{i*100:<15}  W: {mot_watt:<15.2f} g/W:{mot_gW:<15.2f} A:{mot_curr:<15.2f} RPM:{mot_rpm:<15.2f}")



    def thrust_to_throttle(self, thrust_g):
        f = interp1d(self.thrust, self.throttle, fill_value='extrapolate')
        return f(thrust_g)

    def thrust_to_current(self, thrust_g):
        f = interp1d(self.thrust, self.current, fill_value='extrapolate')
        return f(thrust_g)

    def thrust_to_rpm(self, thrust_g):
        f = interp1d(self.thrust, self.rpm, fill_value='extrapolate')
        return f(thrust_g)

    def thrust_to_watts(self, thrust_g):
        f = interp1d(self.thrust, self.watts, fill_value='extrapolate')
        return f(thrust_g)

    def rpm_to_efficiency(self, rpm):
        f = interp1d(self.rpm, self.efficiency, fill_value='extrapolate')
        return f(rpm)

    def thrust_to_efficiency(self, thrust_g):
        f = interp1d(self.thrust, self.efficiency, fill_value='extrapolate')
        return f(thrust_g)

    # Example plot function
    def plot_thrust_to(self, target_variable, thrust_values):
        true_values = interp1d(thrust_values, self.thrust, getattr(self, target_variable))
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


    def motor_perf_thrust(self, thr_g): # interpolation for thrust g
        if thr_g > self.max_thrust:
            #print('Error: over maximum thrust')
            return math.inf, math.inf, math.inf, math.inf, math.inf
        rpm = max(0.0,self.thrust_to_rpm(thr_g)) #rpm
        current_per_motor = max(0.0,self.thrust_to_current(thr_g)) #a
        elec_eff = max(0.0,self.rpm_to_efficiency(rpm)) #g/w
        power_per_motor = max(0.0,thr_g / elec_eff) #max(0.0,self.thrust_to_watts(thr_g)) #w
        throttle = max(0.0,self.thrust_to_throttle(thr_g)) # %

        return power_per_motor, current_per_motor, elec_eff, throttle, rpm

    def motor_perf_thrust2(self, thr_g, v, airspd): # interpolation for thrust g
        if thr_g > self.max_thrust:
            #print('Error: over maximum thrust')
            return math.inf, math.inf, math.inf, math.inf, math.inf
        rpm = rpm_from_thrust(thr_g, self.prop_diameter, self.prop_pitch, airspd)
        #current_per_motor = max(0.0,self.thrust_to_current(thr_g)) #a
        elec_eff = max(0.0,self.rpm_to_efficiency(rpm)) #g/w
        power_per_motor = max(0.0,thr_g / elec_eff) #max(0.0,self.thrust_to_watts(thr_g)) #w
        current_per_motor = power_per_motor / v
        throttle = max(0.0,self.thrust_to_throttle(thr_g)) # %

        return power_per_motor, current_per_motor, elec_eff, throttle, rpm

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


def prop_thrust(RPM: float, diameter_in: float, pitch_in: float, v: float, clip_negative: bool = True) -> float:
    """
    https://www.tytorobotics.com/blogs/articles/how-to-calculate-propeller-thrust
    """
    if pitch_in <= 0:
        raise ValueError("pitch_in must be > 0")

    thrust_n = 4.392e-8 * (
        RPM * (math.pow(diameter_in, 3.5)) / math.sqrt(pitch_in)
        * (4.233e-4 * RPM * pitch_in - v)
        ) 
    thrust_g = thrust_n / 9.81 * 1000
    
    if clip_negative:
        thrust_g = max(thrust_g, 0.0)
    return thrust_g


def rpm_from_thrust(thrust_g: float, diameter_in: float, pitch_in: float, v: float) -> float:
    """
    Invert prop_thrust() to get RPM from desired thrust (in grams).

    Forward model (thrust in N):
        T = K * RPM * (C * RPM - v)
        where:
            K = 4.392e-8 * d^3.5 / sqrt(pitch)
            C = 4.233e-4 * pitch
            d = diameter_in (in)
            pitch = pitch_in (in)
            v = forward airspeed (m/s)

    After algebra:
        RPM = ( v + sqrt( v^2 + 4 * C * T / K ) ) / ( 2 * C )

    Parameters
    ----------
    thrust_g : float
        Desired thrust in grams ( ≥ 0 ).
    diameter_in : float
        Prop diameter in inches.
    pitch_in : float
        Prop pitch in inches (> 0).
    v : float
        Forward airspeed (m/s).

    Returns
    -------
    float
        Required RPM (rev/min).
    """
    if pitch_in <= 0:
        raise ValueError("pitch_in must be > 0")
    if diameter_in <= 0:
        raise ValueError("diameter_in must be > 0")
    if thrust_g < 0:
        raise ValueError("thrust_g must be >= 0")

    # Convert thrust grams -> Newtons
    thrust_n = thrust_g / 1000.0 * 9.81
    if thrust_n == 0:
        return 0.0

    K = 4.392e-8 * (diameter_in ** 3.5) / math.sqrt(pitch_in)
    C = 4.233e-4 * pitch_in

    # RPM = (v + sqrt(v^2 + 4*C*T/K)) / (2*C)
    inside = v * v + 4.0 * C * thrust_n / K
    RPM = (v + math.sqrt(inside)) / (2.0 * C)
    return RPM
