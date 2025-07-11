from stable_baselines3 import SAC
from learning.env import ProjectileEnv
from simulation.render import Renderer
import os


class Test:
    def __init__(self):
        self.models = dict()

    def test(self, max_steps=10, air_resistance=False, record_trajectory=False,test_one_model=False,test_model_name=""):

        env = ProjectileEnv(simulation_config={"apply_air_resistance": air_resistance, "record_trajectory": record_trajectory})

        if test_one_model:
            models = self.get_models(air_resistance=air_resistance)[test_model_name]
        else:
            models = self.get_models(air_resistance=air_resistance)

        for model_item in models.values():
            model = SAC.load(model_item['path'], env=env)
            obs,_ = env.reset()

            terminated = False

            print(f"CURRENT MODEL:{model_item['name']}")

            while not terminated:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action, max_steps=max_steps)

                record = {
                    "landing_x": info['landing_x'],
                    "angle": info['angle'],
                    "speed": info['speed'],
                    "error": info['error'],
                    "trajectory": info['trajectory']
                }

                model_item['record_history'].append(record)

                if info['landing_x'] > model_item['best_record_by_landing_x']['landing_x']:
                    model_item['best_record_by_landing_x'] = record

                model_item['average_record'] = {
                    "landing_x": model_item['average_record']['landing_x'] + info['landing_x'],
                    "angle": model_item['average_record']['angle'] + info['angle'],
                    "speed": model_item['average_record']['speed'] + info['speed'],
                    "error": model_item['average_record']['error'] + info['error']
                }

            model_item['average_record'] = {
                "landing_x": model_item['average_record']['landing_x'] / max_steps,
                "angle": model_item['average_record']['angle'] / max_steps,
                "speed": model_item['average_record']['speed'] / max_steps,
                "error": model_item['average_record']['error'] / max_steps
            }

        return models

    def get_models(self, air_resistance=False):

        if air_resistance:
            files = [f for f in os.listdir("models_sac_with_air_resistance/checkpoints/") if "sac_projectile_with_air_resistance" in f]
            for file in files:
                iteration = file.split("_")[-2]
                self.models[f"model_with_air_resistance_{iteration}"] = {
                    "path": os.path.join("models_sac_with_air_resistance/checkpoints/", file), 
                    "name": file,
                    "best_record_by_landing_x": {
                        "landing_x": 0,
                        "angle": 0,
                        "speed": 0,
                        "error": 0
                    },
                    "average_record": {
                        "landing_x": 0,
                        "angle": 0,
                        "speed": 0,
                        "error": 0
                    },
                    "record_history": []
                }
        else:
            files = [f for f in os.listdir("models_sac/checkpoints/") if "sac_projectile" in f]
            for file in files:
                iteration = file.split("_")[-2]
                self.models[f"model_{iteration}"] = {
                    "path": os.path.join("models_sac/checkpoints/", file), 
                    "name": file,
                    "best_record_by_landing_x": {
                        "landing_x": 0,
                        "angle": 0,
                        "speed": 0,
                        "error": 0
                    },
                    "average_record": {
                        "landing_x": 0,
                        "angle": 0,
                        "speed": 0,
                        "error": 0
                    },
                    "record_history": []
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



if __name__ == "__main__":
    test = Test()
    result = test.test(max_steps=100, air_resistance=False, record_trajectory=False)
    test.print_model_best_record(result, sort="desc")
    test.print_model_average_record(result, sort="desc")
    

