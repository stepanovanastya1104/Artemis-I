import krpc
import time
import csv
from math import sqrt
import numpy as np
import pathlib

turn_start_altitude = 250  #начало гравитационного поворота
turn_end_altitude = 45000  #завершение поворота
min_apoapsis = 84000  #минимальный апоцентр

conn = krpc.connect(name='Artemis-1')
vessel = conn.space_center.active_vessel

ut = conn.add_stream(getattr, conn.space_center, 'ut')
altitude = conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
periapsis = conn.add_stream(getattr, vessel.orbit, 'periapsis_altitude')
mass = conn.add_stream(getattr, vessel, 'mass')  # Масса

PATH = str(pathlib.Path(__file__).parent.joinpath("ksp_flight_data.csv"))
with open(PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Altitude", "Vertical Velocity", "Horizontal Velocity",
                     "Total Velocity", "Drag", "Displacement", "Engine Thrust",
                     "Vacuum Thrust", "Air Temperature", "Mass", "Pitch",
                     "Heading", "Roll"])

    vessel.control.throttle = 1.0
    vessel.auto_pilot.engage()
    vessel.auto_pilot.target_pitch_and_heading(90, 90)
    vessel.auto_pilot.target_roll = 0

    start_time = conn.space_center.ut
    initial_position = vessel.position(vessel.orbit.body.reference_frame)
    initial_position_vec_length = np.linalg.norm(initial_position)

    print('3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)
    print('Пуск!')

    vessel.control.activate_next_stage()

    initial_thrust = vessel.thrust  #текущая тяга
    vacuum_thrust = sum(
        [engine.max_vacuum_thrust for engine in vessel.parts.engines])  #тяга в вакууме
    print(f"Начальная тяга: {initial_thrust} Н, Тяга в вакууме: {vacuum_thrust} Н")
    print(f"Начальная масса: {mass():.2f} кг")

    turn_angle = 0
    srb_dropped = False
    las_dropped = False
    core_stage_dropped = False
    tli_transfer = False

    while True:
        ut = conn.space_center.ut
        elapsed_time = ut - start_time
        #гравитационный поворот
        if turn_start_altitude < altitude() < turn_end_altitude:
            frac = ((altitude() - turn_start_altitude) /
                    (turn_end_altitude - turn_start_altitude))
            new_turn_angle = frac * 90
            if abs(new_turn_angle - turn_angle) > 0.5:
                turn_angle = new_turn_angle
                vessel.auto_pilot.target_pitch_and_heading(90 - (turn_angle), 90)
        if not srb_dropped and vessel.resources.amount('SolidFuel') < 400:
            srb_dropped = True
            vessel.control.activate_next_stage()
            print("Сброс ускорителей")
            print(f"Масса после сброса ускорителей: {mass():.2f} кг")
        #температура воздуха на текущей высоте
        try:
            air_temperature = vessel.flight().static_air_temperature
        except AttributeError:
            air_temperature = None  #если температура недоступна
        #получение углов
        flight_info = vessel.flight()
        current_pitch = flight_info.pitch  #тангаж
        current_heading = flight_info.heading  #курс
        current_roll = flight_info.roll  #крен

        #запись данных в файл
        altitude_val = altitude()
        speed = vessel.flight(vessel.orbit.body.reference_frame).speed
        drag_x, drag_y, drag_z = vessel.flight().drag
        drag = sqrt(drag_x ** 2 + drag_y ** 2 + drag_z ** 2)
        current_position = vessel.position(vessel.orbit.body.reference_frame)
        current_position = current_position / \
                           np.linalg.norm(current_position) * initial_position_vec_length
        horizontal_displacement = np.linalg.norm(
            current_position - initial_position)
        vertical_speed = vessel.flight(
            vessel.orbit.body.reference_frame).vertical_speed
        horizontal_speed = vessel.flight(
            vessel.orbit.body.reference_frame).horizontal_speed

        #получение тяги двигателя
        current_thrust = vessel.thrust
        #получение массы
        current_mass = mass()
        writer.writerow([
            elapsed_time, altitude_val, vertical_speed, horizontal_speed,
            speed, drag, horizontal_displacement, current_thrust,
            vacuum_thrust, air_temperature, current_mass,
            current_pitch, current_heading, current_roll  # Добавили углы ориентации
        ])

        #достижение целевых орбитальных параметров
        if not tli_transfer and apoapsis() > min_apoapsis:
            tli_transfer = True
            vessel.control.throttle = 0

            print(f"Финальная масса: {current_mass:.2f} кг")
            print(f"Текущие углы: Тангаж={current_pitch:.1f}°, Курс={current_heading:.1f}°, Крен={current_roll:.1f}°")
            print("Орбита достигнута. Завершение работы.")
            break

        time.sleep(0.1)