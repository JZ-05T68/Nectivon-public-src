"""Independent numerical reference checks for R3; these do not grade an unseen model answer.

The samples derive relations from phase sums, definitions and balances. Passing
them validates the review references, not all future generated prose or formulas.
The original private Word and runtime outputs are deliberately not test fixtures.
"""

from cmath import exp
from math import cos, isclose, pi, sin, sqrt

import pytest


@pytest.mark.parametrize("connection", ["star", "delta"])
def test_line_power_and_kw_conversion_from_independent_phase_sum(connection):
    line_voltage, line_current, angle = 400.0, 12.0, 0.6
    phase_voltage = line_voltage / sqrt(3) if connection == "star" else line_voltage
    phase_current = line_current if connection == "star" else line_current / sqrt(3)
    phase_shifts = [0.0, -2 * pi / 3, 2 * pi / 3]
    power = sum((phase_voltage * exp(1j * shift)
                 * (phase_current * exp(1j * (shift - angle))).conjugate()).real
                for shift in phase_shifts)
    assert isclose(power, sqrt(3) * line_voltage * line_current * cos(angle))
    power_kw = power / 1000
    assert isclose(1000 * power_kw / (sqrt(3) * line_voltage * cos(angle)), line_current)
    assert not isclose(3 * line_voltage * line_current * cos(angle), power)


@pytest.mark.parametrize("pairs,slip", [(2, 0.04), (3, -0.06), (4, 0.25)])
def test_slip_speed_and_rotor_frequency_have_different_quantity_and_unit_scales(pairs, slip):
    supply_hz = 50.0
    synchronous_rpm = supply_hz / pairs * 60
    actual_rpm = synchronous_rpm * (1 - slip)
    speed_difference = synchronous_rpm - actual_rpm
    rotor_cycles = abs(pairs * speed_difference / 60)
    assert isclose(speed_difference, slip * synchronous_rpm)
    assert isclose(speed_difference, 60 * slip * supply_hz / pairs)
    assert isclose(rotor_cycles, abs(slip) * supply_hz)
    assert not isclose(speed_difference, slip * supply_hz)
    assert rotor_cycles >= 0


def test_constant_core_loss_is_a_fixed_supply_approximation_not_voltage_invariance():
    def core_loss(voltage, frequency):
        flux = voltage / frequency
        return 0.03 * frequency * flux**1.6 + 0.002 * frequency**2 * flux**2

    # Load is not in this fixed-excitation approximation; voltage explicitly is.
    fixed = [core_loss(230, 50) for _load in (0, 0.5, 1)]
    assert len(set(fixed)) == 1
    assert core_loss(250, 50) > core_loss(230, 50)


@pytest.mark.parametrize("angle", [pi / 6, -pi / 6])
def test_regulation_sign_and_percent_scale_from_complex_voltage_drop(angle):
    resistance, reactance, load = 0.008, 0.035, 0.7
    current = load * exp(-1j * angle)
    drop = current * complex(resistance, reactance)
    # Rated secondary phase-voltage base is 1 pu; percentage is 100*pu.
    projected_percent = 100 * drop.real
    approximation = load * (100 * resistance * cos(angle)
                            + 100 * reactance * sin(angle))
    exact_percent = 100 * (abs(1 + drop) - 1)
    assert isclose(projected_percent, approximation)
    assert abs(exact_percent - approximation) < 0.05
    assert (approximation > 0) == (angle > 0)


def test_dc_mechanical_characteristic_uses_developed_not_shaft_torque():
    voltage, current, resistance, flux = 220.0, 20.0, 0.5, 1.0
    emf_constant = 0.2  # n in r/min
    torque_constant = emf_constant * 60 / (2 * pi)
    emf = voltage - current * resistance
    rpm = emf / (emf_constant * flux)
    omega = rpm * 2 * pi / 60
    electromagnetic_torque = torque_constant * flux * current
    assert isclose(electromagnetic_torque * omega, emf * current)
    from_characteristic = (voltage / (emf_constant * flux)
                           - resistance * electromagnetic_torque
                           / (emf_constant * torque_constant * flux**2))
    assert isclose(from_characteristic, rpm)
    shaft_torque = electromagnetic_torque - 200 / omega
    assert shaft_torque < electromagnetic_torque
    assert not isclose(voltage / (emf_constant * flux)
                       - resistance * shaft_torque
                       / (emf_constant * torque_constant * flux**2), rpm)


@pytest.mark.parametrize("connection", ["star", "delta"])
def test_total_three_phase_copper_loss_uses_actual_winding_currents(connection):
    line_current, phase_resistance = 9.0, 0.7
    winding_current = line_current if connection == "star" else line_current / sqrt(3)
    total = sum(winding_current**2 * phase_resistance for _ in range(3))
    assert isclose(total, 3 * winding_current**2 * phase_resistance)
    assert total > winding_current**2 * phase_resistance


def test_voltage_limit_separates_constant_ratio_region_from_field_weakening():
    maximum_voltage, base_hz = 400.0, 50.0

    def voltage(hz):
        return min(maximum_voltage, maximum_voltage * hz / base_hz)

    assert isclose(voltage(25) / 25, voltage(50) / 50)
    assert voltage(100) == maximum_voltage
    assert voltage(100) / 100 < voltage(base_hz) / base_hz
    # At low frequency a nonzero stator drop consumes more of the available voltage.
    assert (voltage(5) - 10) / 5 < voltage(5) / 5
