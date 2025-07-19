from simlib import *

motor_model = MotorData("Axisflying", "AX2812", "HQ9045", int(900), int(6))
v = 0
print('Airspeed:',v)
for rpm in np.arange(0, 16000 + 1000, 1000):
    t = prop_thrust(rpm, motor_model.prop_diameter, motor_model.prop_pitch, v) * 1.25
    r = rpm_from_thrust(t, motor_model.prop_diameter, motor_model.prop_pitch, v)
    eff = motor_model.rpm_to_efficiency(rpm) # more accurate
    w = t/eff # more accurate
    print(f"RPM: {rpm:<7.0f}\tT(g): {t:.1f}\tRPM_r: {r:.0f}\tg/W: {eff:.2f}\tT/g/W: {w:.2f}")
    #print(rpm, t, r, w1, w2)