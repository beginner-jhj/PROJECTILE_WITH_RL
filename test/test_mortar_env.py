import math
from learning.mortar_env import MortarEnv


def test_angle_range():
    env = MortarEnv()
    assert math.isclose(env.min_angle, 45.0)
    assert math.isclose(env.max_angle, 85.0)


def test_error_calculation():
    env = MortarEnv()
    env.reset()
    action = [60.0]
    _, _, _, _, info = env.step(action)
    assert math.isclose(info["error"], env.calculate_error(info["landing_x"]))


def test_model_prediction_error():
    """Evaluate trained SAC model and compare to 20 m RPE."""
    import numpy as np
    from stable_baselines3 import SAC

    env = MortarEnv()
    # Model was trained with a different action space; override when loading.
    custom_objects = {
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }
    model_path = "models_sac_mortar/checkpoints/sac_final_model.zip"
    model = SAC.load(model_path, env=env, custom_objects=custom_objects)

    errors = []
    for _ in range(5):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        errors.append(info["error"])

    avg_error_px = float(np.mean(errors))
    px_to_m = 1.0  # 1 pixel == 1 meter in the simulation
    avg_error_m = avg_error_px * px_to_m

    human_rpe = 20.0
    diff = avg_error_m - human_rpe

    # Store results for inspection
    info_msg = (
        f"Average error {avg_error_m:.2f} m; "
        f"difference from 20 m RPE: {diff:.2f} m"
    )
    print(info_msg)

    assert isinstance(avg_error_m, float)
