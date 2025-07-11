from stable_baselines3 import SAC
from learning.env import ProjectileEnv
from learning.mortar_env import MortarEnv
from simulation.render import Renderer
import os


class Test:
    def __init__(self):
        self.models = dict()

    def test(self, max_steps=10, mode="no_air_resistance", record_trajectory=False,
             test_one_model=False, test_model_name=""):

        if mode == "mortar":
            env = MortarEnv()
        else:
            air_resistance = mode == "with_air_resistance"
            env = ProjectileEnv(
                simulation_config={
                    "apply_air_resistance": air_resistance,
                    "record_trajectory": record_trajectory,
                }
            )

        if test_one_model:
            models = self.get_models(mode)[test_model_name]
        else:
            models = self.get_models(mode)

        for model_item in models.values():
            model = SAC.load(model_item['path'], env=env)
            obs, _ = env.reset()
            step_count = 0

            terminated = False

            print(f"CURRENT MODEL:{model_item['name']}")

            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                if mode == "mortar":
                    obs, reward, terminated, truncated, info = env.step(action)
                    angle = float(action[0])
                    speed = env.speed
                    trajectory = None
                else:
                    obs, reward, terminated, truncated, info = env.step(action, max_steps=max_steps)
                    angle = info["angle"]
                    speed = info["speed"]
                    trajectory = info["trajectory"]

                step_count += 1
                record = {
                    "landing_x": info["landing_x"],
                    "angle": angle,
                    "speed": speed,
                    "error": info["error"],
                    "trajectory": trajectory,
                }

                model_item['record_history'].append(record)

                if info['landing_x'] > model_item['best_record_by_landing_x']['landing_x']:
                    model_item['best_record_by_landing_x'] = record

                model_item['average_record'] = {
                    "landing_x": model_item['average_record']['landing_x'] + info["landing_x"],
                    "angle": model_item['average_record']['angle'] + angle,
                    "speed": model_item['average_record']['speed'] + speed,
                    "error": model_item['average_record']['error'] + info["error"],
                }

            model_item['average_record'] = {
                "landing_x": model_item['average_record']['landing_x'] / step_count,
                "angle": model_item['average_record']['angle'] / step_count,
                "speed": model_item['average_record']['speed'] / step_count,
                "error": model_item['average_record']['error'] / step_count,
            }

        return models

    def get_models(self, mode="no_air_resistance"):
        self.models = {}

        if mode == "with_air_resistance":
            path = "models_sac_with_air_resistance/checkpoints/"
            prefix = "model_with_air_resistance"
            pattern = "sac_projectile_with_air_resistance"
        elif mode == "mortar":
            path = "models_sac_mortar/checkpoints/"
            prefix = "model_mortar"
            pattern = "sac_mortar"
        else:
            path = "models_sac/checkpoints/"
            prefix = "model"
            pattern = "sac_projectile"

        files = [f for f in os.listdir(path) if pattern in f]
        for file in files:
            iteration = file.split("_")[-2]
            self.models[f"{prefix}_{iteration}"] = {
                "path": os.path.join(path, file),
                "name": file,
                "best_record_by_landing_x": {
                    "landing_x": 0,
                    "angle": 0,
                    "speed": 0,
                    "error": 0,
                },
                "average_record": {
                    "landing_x": 0,
                    "angle": 0,
                    "speed": 0,
                    "error": 0,
                },
                "record_history": [],
            }

        return self.models

    def print_model_best_record(self, models, sort="desc"):
        if sort == "desc":
            models = sorted(models.values(), key=lambda x: x['best_record_by_landing_x']['landing_x'], reverse=True)
        else:
            models = sorted(models.values(), key=lambda x: x['best_record_by_landing_x']['landing_x'])
        for model_item in models:
            print(f"Model: {model_item['name']}")
            print(f"Best landing x: {model_item['best_record_by_landing_x']['landing_x']}")
            print(f"Best angle: {model_item['best_record_by_landing_x']['angle']}")
            print(f"Best speed: {model_item['best_record_by_landing_x']['speed']}")
            print(f"Best error: {model_item['best_record_by_landing_x']['error']}")
            print("\n")

    def print_model_average_record(self, models, sort="desc"):
        if sort == "desc":
            models = sorted(models.values(), key=lambda x: x['average_record']['landing_x'], reverse=True)
        else:
            models = sorted(models.values(), key=lambda x: x['average_record']['landing_x'])
        for model_item in models:
            print(f"Model: {model_item['name']}")
            print(f"Average landing x: {model_item['average_record']['landing_x']}")
            print(f"Average angle: {model_item['average_record']['angle']}")
            print(f"Average speed: {model_item['average_record']['speed']}")
            print(f"Average error: {model_item['average_record']['error']}")
            print("\n")

    def print_record_history(self, models):
        for model_item in models.values():
            print(f"Model: {model_item['name']}")
            for record in model_item['record_history']:
                print(f"Landing x: {record['landing_x']}")
                print(f"Angle: {record['angle']}")
                print(f"Speed: {record['speed']}")
                print(f"Error: {record['error']}")
                print("\n")



if __name__ == "__main__":
    test = Test()
    result = test.test(max_steps=30, mode="no_air_resistance", record_trajectory=False)
    
    test.print_model_best_record(result, sort="desc")
    test.print_model_average_record(result)
    test.print_record_history(result)
    

