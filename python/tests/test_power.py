from rackscale import power


def test_thermal_limit_and_junction():
    assert power.thermal_power_limit(50, 0.04, 90) == 1000
    assert power.junction_temp_c(1000, 50, 0.04) == 90
    assert power.thermal_power_limit(95, 0.04, 90) == 0


def test_distribution_scales_as_one_over_v_squared():
    a, b = power.distribution(140e3, 54), power.distribution(140e3, 800)
    assert abs(a["loss_w"] / b["loss_w"] - (800 / 54) ** 2) < 1e-6


def test_coolant_and_cdu():
    f = power.coolant_flow(140e3, 10)
    assert 200 < f["l_per_min"] < 220  # 140 kW / (3950 J/kg/K * 10 K) = 3.54 kg/s
    assert power.racks_per_cdu(2e6, 140e3) == 14


def test_smoothing_buffer_of_a_square_wave():
    # 1 s at 2 kW, 1 s at 0: the store absorbs 1 kW for 1 s, 1 kJ.
    s = power.smoothing([2000.0, 0.0], bin_us=1e6)
    assert abs(s["buffer_j"] - 1000.0) < 1e-6
    assert s["swing_w"] == 2000.0
    assert abs(power.smoothing([2000.0, 0.0], 1e6, floor_fraction=0.5)["floor_burn_j"] - 1000.0) < 1e-6


def test_square_wave_closed_form_matches_the_trace():
    # 3 bins high, 1 low: duty 0.75 of a 4 s period.
    w = [2000.0, 2000.0, 2000.0, 0.0]
    assert abs(power.smoothing(w, 1e6)["buffer_j"] - power.square_wave_buffer_j(2000.0, 4.0, 0.75)) < 1e-6


def test_cap_tradeoff():
    t = power.cap_tradeoff(600, 1100, 261)
    assert abs(t["clock"] - (339 / 839) ** (1 / 3)) < 1e-12
    assert t["time"] > 1.3 and t["energy"] < 0.75
    assert power.cap_tradeoff(1100, 1100, 261) == {"clock": 1.0, "time": 1.0, "energy": 1.0}
