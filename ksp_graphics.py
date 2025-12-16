from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

df = pd.read_csv('ksp_flight_data.csv')
df['Time'] = df['Time'] - df['Time'].min() #время с 0

time_data_ksp = df['Time'].values #время в секундах
height_data_ksp = df['Altitude'].values #высота в метрах
velocity_data_ksp = df['Total Velocity'].values #полная скорость
horizontal_velocity_ksp = df['Horizontal Velocity'].values #горизонтальная скорость
vertical_velocity_ksp = df['Vertical Velocity'].values #вертикальная скорость
horizontal_displacement_ksp = df['Displacement'].values #горизонтальное смещение
mass_data_ksp = df['Mass'].values #масса ракеты
drag_data_ksp = df['Drag'].values #сила сопротивления
pitch_data_ksp = df['Pitch'].values #угол тангажа
thrust_data_ksp = df['Engine Thrust'].values #тяга двигателя

G = 6.67430e-11 #гравитационная постоянная
M = 5.2915158 * 10 ** 22 #масса Кербина
R = 600_000 #радиус Кербина

R_g = 8.314 #универсальная газовая постоянная
M_m = 0.029 #молярная масса воздуха

F0 = thrust_data_ksp[0] #начальная тяга
F1 = max(thrust_data_ksp) #максимальная тяга
T = time_data_ksp[np.where(thrust_data_ksp == 0)[0][0]] #время работы двигателя по умолчанию

T_k = 309.67 #температура на поверхности Кербина
C = 0.5 #коэффициент лобового сопротивления
S = math.pi * 4.2 ** 2 #площадь поперечного сечения
p0 = 1.2255 #плотность воздуха на поверхности
H0 = height_data_ksp[0] #начальная высота

phi0 = math.radians(pitch_data_ksp[0]) #начальный угол в радианах
beta = math.radians((pitch_data_ksp[-1] - pitch_data_ksp[0]) / (time_data_ksp[-1] - time_data_ksp[0])) #скорость изменения угла (рад/с)

m0 = mass_data_ksp[0] #начальная масса
m_final = mass_data_ksp[-1] #конечная масса
m_fuel = m0 - m_final #масса топлива
k = m_fuel / T #расход топлива

g = 9.81  #ускорение свободного падения на поверхности
dt = 0.01 #шаг интегрирования

def p_h(h):
    #плотность атмосферы
    return p0 * math.exp(-(g * M_m * h) / (R_g * T_k))

def gravity(h):
    #гравитация
    return G * M / (R + h) ** 2

def thrust(t):
    #Тяга двигателя
    if t <= T:
        alpha = (F1 - F0) / T
        return F0 + alpha * t
    return 0

def phi_t(t):
    #Угол тангажа
    return phi0 + beta * t

def rocket_system(t, state):
    x, y, vx, vy, m = state
    #масса ракеты
    if t <= T:
        m_current = m0 - k * t
    else:
        m_current = m0 - m_fuel
    if m_current < (m0 - m_fuel):
        m_current = m0 - m_fuel
    h = y
    phi = phi_t(t)
    #скорость и направление вектора скорости
    v_sq = vx ** 2 + vy ** 2
    if v_sq > 0:
        v = math.sqrt(v_sq)
        dir_vx = vx / v
        dir_vy = vy / v
    else:
        v = 0.01
        dir_vx = math.cos(phi)
        dir_vy = math.sin(phi)
    F_thrust = thrust(t)
    F_drag = 0.5 * C * S * p_h(h) * v ** 2
    F_gravity = m_current * gravity(h)

    F_thrust_x = F_thrust * math.cos(phi)
    F_thrust_y = F_thrust * math.sin(phi)

    F_drag_x = -F_drag * dir_vx
    F_drag_y = -F_drag * dir_vy

    F_gravity_x = 0
    F_gravity_y = -F_gravity

    F_total_x = F_thrust_x + F_drag_x + F_gravity_x
    F_total_y = F_thrust_y + F_drag_y + F_gravity_y

    #ускорения(второй закон Ньютона)
    ax = F_total_x / m_current
    ay = F_total_y / m_current

    # производная массы
    if t <= T:
        dm = -k
    else:
        dm = 0
    return [vx, vy, ax, ay, dm]

initial_state = [0, H0, 0, 0, m0]
t_start = 0
t_end = min(180, max(time_data_ksp))
t_eval = np.arange(t_start, t_end, dt)

#система уравнений методом Рунге-Кутта 4-5 порядка
solution = solve_ivp(
    rocket_system, #функция с уравнениями
    [t_start, t_end], #интервал времени
    initial_state, #начальные условия
    method='RK45', #метод интегрирования
    t_eval=t_eval, #в какие моменты выводить результат
    rtol=1e-8, #относительная точность
    atol=1e-10, #абсолютная точность
    max_step=0.1 #максимальный шаг интегрирования
)

time_model = solution.t #время
x_model = solution.y[0]
y_model = solution.y[1]
vx_model = solution.y[2]
vy_model = solution.y[3]
mass_model = solution.y[4]

velocity_model = [math.sqrt(vx_model[i] ** 2 + vy_model[i] ** 2)
                  for i in range(len(time_model))] #полная скорость из компонент
pitch_model = [math.degrees(phi_t(t)) for t in time_model] #угол
thrust_model = [thrust(t) for t in time_model] #тяга
drag_model = [0.5 * C * S * p_h(y_model[i]) * velocity_model[i] ** 2
              for i in range(len(time_model))] #сопротивление

plt.figure(figsize=(18, 12))

#высота от времени
plt.subplot(2, 3, 1)
plt.plot(time_data_ksp, height_data_ksp, 'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(time_model, y_model, 'b-', linewidth=2, label='Математическая модель')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.title('Высота от времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(150, max(time_data_ksp)))

#скорость от времени
plt.subplot(2, 3, 2)
plt.plot(time_data_ksp, velocity_data_ksp, 'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(time_model, velocity_model, 'b-', linewidth=2, label='Математическая модель')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.title('Полная скорость от времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(150, max(time_data_ksp)))

#траектория полета
plt.subplot(2, 3, 3)
max_disp = min(max(horizontal_displacement_ksp), max(x_model))
idx_ksp = horizontal_displacement_ksp <= max_disp
idx_model = x_model <= max_disp
plt.plot(horizontal_displacement_ksp[idx_ksp], height_data_ksp[idx_ksp],
         'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(x_model[idx_model], y_model[idx_model],
         'b-', linewidth=2, label='Математическая модель')
plt.xlabel('Горизонтальное смещение (м)')
plt.ylabel('Высота (м)')
plt.title('Траектория полета')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

#масса от времени
plt.subplot(2, 3, 4)
plt.plot(time_data_ksp, mass_data_ksp, 'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(time_model, mass_model, 'b-', linewidth=2, label='Математическая модель')
plt.axvline(x=T, color='gray', linestyle='--', alpha=0.5, label=f'Отсечка двигателя (T={T:.1f} с)')
plt.xlabel('Время (с)')
plt.ylabel('Масса (кг)')
plt.title('Масса ракеты от времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(150, max(time_data_ksp)))

#угол тангажа от времени
plt.subplot(2, 3, 5)
plt.plot(time_data_ksp, pitch_data_ksp, 'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(time_model, pitch_model, 'b-', linewidth=2, label='Математическая модель')
plt.xlabel('Время (с)')
plt.ylabel('Тангаж (градусы)')
plt.title('Угол наклона от времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(150, max(time_data_ksp)))
plt.axhline(y=90, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

#аэродинамическое сопротивление от времени
plt.subplot(2, 3, 6)
plt.plot(time_data_ksp, drag_data_ksp, 'r-', linewidth=1.5, alpha=0.7, label='KSP данные')
plt.plot(time_model, drag_model, 'b-', linewidth=2, label='Математическая модель')
plt.xlabel('Время (с)')
plt.ylabel('Сопротивление (Н)')
plt.title('Аэродинамическое сопротивление от времени')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(150, max(time_data_ksp)))
plt.yscale('log')

plt.tight_layout()
plt.savefig('graphics.png', dpi=300, bbox_inches='tight')
plt.show()