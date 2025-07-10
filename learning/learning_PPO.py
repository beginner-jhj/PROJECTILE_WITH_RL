from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from learning.env import ProjectileEnv
import os
from torch import nn

# 모델 저장 디렉토리 생성
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

def create_env():
    """환경 생성 함수"""
    return ProjectileEnv()

# 환경 설정
env = Monitor(create_env(), "logs/")
eval_env = Monitor(create_env(), "logs/eval/")

# PPO 모델 생성 (TD3 대신 PPO 사용 - sparse reward에 더 적합)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,  # 적절한 학습률
    n_steps=2048,        # 배치 크기
    batch_size=64,       # 미니배치 크기
    n_epochs=10,         # 에폭 수
    gamma=0.99,          # 할인율
    gae_lambda=0.95,     # GAE 람다
    clip_range=0.2,      # 클리핑 범위
    ent_coef=0.07,       # 엔트로피 계수 (탐험 장려)
    vf_coef=0.5,         # 가치 함수 계수
    max_grad_norm=0.5,   # 그래디언트 클리핑
    policy_kwargs={
        "net_arch": [dict(pi=[128, 128], vf=[128, 128])],  # 더 큰 네트워크
        "activation_fn": nn.ReLU
    },
    tensorboard_log="logs/tensorboard/"
)

# 콜백 설정
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/best_model/",
    log_path="logs/eval/",
    eval_freq=10000,
    deterministic=True,
    render=False,
    n_eval_episodes=10
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="models/checkpoints/",
    name_prefix="ppo_projectile"
)

# 훈련 설정
total_timesteps_per_iter = 100_000
num_iterations = 10

print("=== 학습 시작 ===")
print(f"총 반복 수: {num_iterations}")
print(f"반복당 timesteps: {total_timesteps_per_iter}")
print(f"총 timesteps: {total_timesteps_per_iter * num_iterations}")

# 학습 실행
for i in range(num_iterations):
    print(f"\n=== Iteration {i+1}/{num_iterations} ===")
    
    # 학습
    model.learn(
        total_timesteps=total_timesteps_per_iter,
        reset_num_timesteps=False,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=f"PPO_iter_{i+1}"
    )
    
    # 체크포인트 저장
    model.save(f"models/ppo_projectile_iter_{i+1}")
    
    # 성능 평가
    print(f"Iteration {i+1} 완료. 모델 저장됨.")
    
    # 간단한 테스트
    if (i + 1) % 2 == 0:  # 2번째 반복마다 테스트
        print("=== 중간 테스트 ===")
        test_env = create_env()
        obs,_ = test_env.reset()
        
        total_reward = 0
        best_error = float('inf')
        
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            
            if info['error'] < best_error:
                best_error = info['error']
            
            print(f"Step {step+1}: Error={info['error']:.2f}, "
                  f"Angle={info['angle']:.2f}, Speed={info['speed']:.2f}, "
                  f"Reward={reward:.3f}")
            
            if done:
                break
        
        print(f"테스트 결과 - 총 보상: {total_reward:.3f}, 최고 정확도: {best_error:.2f}")
        print("-" * 50)

print("\n=== 학습 완료 ===")
print("최종 모델이 저장되었습니다.")

# 최종 테스트
print("\n=== 최종 성능 테스트 ===")
final_env = create_env()

for test_num in range(5):
    print(f"\n테스트 {test_num + 1}:")
    obs = final_env.reset()
    
    total_reward = 0
    best_error = float('inf')
    
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = final_env.step(action)
        total_reward += reward
        
        if info['error'] < best_error:
            best_error = info['error']
        
        print(f"  Step {step+1}: Error={info['error']:.2f}, "
              f"Angle={info['angle']:.2f}, Speed={info['speed']:.2f}")
        
        if done:
            break
    
    print(f"  결과: 총 보상={total_reward:.3f}, 최고 정확도={best_error:.2f}")

print("\n모든 모델이 'models/' 디렉토리에 저장되었습니다.")
print("TensorBoard 로그는 'logs/tensorboard/' 디렉토리에서 확인할 수 있습니다.")
print("실행 명령: tensorboard --logdir logs/tensorboard/")