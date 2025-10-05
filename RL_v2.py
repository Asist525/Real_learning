import gymnasium as gym
import kymnasium as kym
import matplotlib.pyplot as plt
import random
import pickle

# =========================
# 환경 생성
# =========================
env = gym.make(
    id='kymnasium/GridAdventure-Crossing-26x26-v0',
    render_mode='human',
    bgm=True
)

obs, info = env.reset()
print(f'# Observation shape: {obs.shape}')
print(f'# Info: {info}')

print('Action space: ', env.action_space)
print('State space: ', env.observation_space)

# =========================
# 에이전트 정의
# =========================
class YourAgent(kym.Agent):
    ACTIONS = [0, 1, 2]  # 0=좌회전, 1=우회전, 2=전진

    def __init__(self, PI=None):
        self.PI = PI

    def act(self, obs, info):
        if self.PI is None:  # 학습 전이면 랜덤
            return random.choice(self.ACTIONS)
        else:  # 학습된 정책이 있으면 사용
            W, H = obs.shape[1], obs.shape[0]
            # 에이전트 위치 및 방향 추출
            for code, d in {1000:0, 1001:1, 1002:2, 1003:3}.items():
                ys, xs = (obs == code).nonzero()
                if len(xs) > 0:
                    x, y = int(xs[0]), int(ys[0])
                    state = xy_to_state(x, y, d, W, H)
                    return self.PI[state]
            return random.choice(self.ACTIONS)

    def save(self, path):
        with open(path, mode='wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, mode='rb') as f:
            return pickle.load(f)

# =========================
# 상태 변환 유틸
# =========================
def xy_to_state(x, y, d, W=26, H=26):
    return d * (W * H) + y * W + x

def state_to_xyd(state, W=26, H=26):
    area = W * H
    d, rem = divmod(state, area)
    y, x = divmod(rem, W)
    return x, y, d

# =========================
# 행동 정의
# =========================
DIRS = {
    0: (1, 0),   # →
    1: (0, 1),   # ↓
    2: (-1, 0),  # ←
    3: (0, -1),  # ↑
}

def step_xy(x, y, d, action, W, H, env_map):
    if action == 0:   # 좌회전
        d = (d + 3) % 4
    elif action == 1: # 우회전
        d = (d + 1) % 4
    elif action == 2: # 전진
        dx, dy = DIRS[d]
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H:
            tile = env_map[ny, nx]
            if tile != 250:  # 벽(250)이 아니면 이동
                x, y = nx, ny
    return x, y, d

def estimate_value(state, action, gamma, V, env_map):
    x, y, d = state_to_xyd(state, 26, 26)
    nx, ny, nd = step_xy(x, y, d, action, 26, 26, env_map)
    new_state = xy_to_state(nx, ny, nd)

    tile = env_map[ny, nx]
    if tile == 900:   # lava
        reward = -1
    elif tile == 810: # goal
        reward = 1
    else:
        reward = 0

    if tile in (900, 810):
        return reward
    else:
        return reward + gamma * V[new_state]

# =========================
# Value Iteration
# =========================
# =========================
# Policy Iteration (정책 반복)
# =========================
PHI   = 1e-3   # 정책평가 수렴 임계치
GAMMA = 0.99
n_states = 26 * 26 * 4

V  = [0.0 for _ in range(n_states)]
PI = [random.choice([0,1,2]) for _ in range(n_states)]  # 랜덤 정책 초기화

while True:
    # --- Policy Evaluation ---
    while True:
        delta = 0.0
        for s in range(n_states):
            v_old = V[s]
            V[s] = estimate_value(s, PI[s], GAMMA, V, obs)
            delta = max(delta, abs(V[s] - v_old))
        if delta < PHI:
            break

    # --- Policy Improvement ---
    policy_stable = True
    for s in range(n_states):
        old_a = PI[s]
        q_values = [estimate_value(s, a, GAMMA, V, obs) for a in [0,1,2]]
        best_a = max(range(3), key=lambda a: q_values[a])
        PI[s] = best_a
        if best_a != old_a:
            policy_stable = False

    if policy_stable:
        break


# =========================
# 정책 저장
# =========================
agent = YourAgent(PI)
agent.save('./value_iter_agent.pkl')
print("학습 완료 및 저장: ./value_iter_agent.pkl")

# =========================
# 실행 예시
# =========================
obs, info = env.reset()
done = False
while not done:
    action = agent.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
