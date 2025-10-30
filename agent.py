# agent.py
# Python 3.12
# KNU RL Competition Round 3 - Avoid Blurp
# - Expected SARSA 고정
# - obs_type='custom' 기준 관찰축약
# - 액션마스크(경계 + 1스텝 충돌예측 + stay위험차단 + 공집합복구)
# - 2000 에피소드마다 체크포인트 저장(.pkl)
# - .pkl 안에 reward_cfg / mask_cfg / env_cfg / 학습메타 포함 → 중간에도 evaluate.py로 실행 가능
# - 로그: 스텝은 카운트만, 에피소드마다 1줄, 2000ep마다 상세
# - 추가: --log-file 주면 콘솔 + 파일 둘 다에 같은 형식으로 남김 → pandas 분석 스크립트가 그대로 읽을 수 있음

from __future__ import annotations
import argparse
import os
import pickle
import time
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import gymnasium as gym
import kymnasium as kym  # noqa: F401  # env 등록용

# =============================
# 전역 상수
# =============================
ENV_ID = "kymnasium/AvoidBlurp-Normal-v0"
OBS_TYPE = "custom"
ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3

# 디스크리타이즈 설정
DISCRETE_X_BUCKETS = 15      # 플레이어 x축 버킷 수
DISCRETE_DX_BUCKETS = 5      # 적과의 x차이 버킷 수
DISCRETE_DY_BUCKETS = 3      # 수직거리 버킷 수


# =============================
# 유틸 함수
# =============================
def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def is_enemy_active(e: np.ndarray) -> bool:
    """적 1줄이 전부 0인지 확인"""
    return not np.allclose(e, 0.0)


def overlap(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> bool:
    """1차원 구간 겹침 여부"""
    (a1, a2), (b1, b2) = seg1, seg2
    return not (a2 < b1 or b2 < a1)


# =============================
# 상태 인코더
# =============================
class StateEncoder:
    """
    관찰(obs) → 의사결정용 이산 상태로 바꿔주는 모듈
    - 플레이어 x를 화면폭으로 나눠서 버킷화
    - 가장 가까운 적 1명을 사용
    - 적이 없으면 '위협없음' 상태
    """

    def __init__(self, width: int = 750, height: int = 600, danger_y: float = 200.0):
        self.width = width
        self.height = height
        self.danger_y = danger_y

    def encode(self, obs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        반환:
        - state_id: 이산화된 상태 인덱스 (int)
        - extra: 마스크·보상 계산에 다시 쓸 수 있도록 원시/축약값을 같이 돌려준다
        """
        player = obs["player"]
        enemies = obs["enemies"]

        px, py, pw, ph, pspeed = player
        # 활성 적만 필터
        active_enemies = [e for e in enemies if is_enemy_active(e)]
        # 플레이어 위쪽에 있는 적만 (py보다 y가 작음)
        active_enemies = [e for e in active_enemies if 0 < (py - e[1])]
        # 수직거리 기준 오름차순
        active_enemies.sort(key=lambda e: (py - e[1]))

        # 기본값: 위협 없음
        has_danger = 0
        rel_dx_bucket = 2  # 가운데
        rel_dy_bucket = 2  # 멀리

        # 플레이어 중심
        c_p = px + pw / 2.0

        # 가장 가까운 적 1개만 본다
        if len(active_enemies) > 0:
            e = active_enemies[0]
            ex, ey, ew, eh, espeed, eacc = e
            c_e = ex + ew / 2.0
            dy = py - ey            # 수직거리 (양수면 위에 있음)
            dx = c_e - c_p          # +면 오른쪽 적, -면 왼쪽 적

            # 위협 여부: 수직거리 안에 있고 x구간이 겹치는 경우
            p_seg = (px, px + pw)
            e_seg = (ex, ex + ew)
            if (0 < dy < self.danger_y) and overlap(p_seg, e_seg):
                has_danger = 1

            # dx 이산화: 왼머/왼가까/중앙/오가까/오머 = 5칸
            # 화면폭 750 기준으로 임계치 대충 잡음
            dx_abs = abs(dx)
            close_thr = self.width / 10.0  # 75px 근처
            mid_thr = self.width / 5.0     # 150px 근처
            if dx < -mid_thr:
                rel_dx_bucket = 0  # 왼쪽 멀리
            elif dx < -close_thr:
                rel_dx_bucket = 1  # 왼쪽 가까이
            elif dx_abs <= close_thr:
                rel_dx_bucket = 2  # 중앙
            elif dx > close_thr and dx <= mid_thr:
                rel_dx_bucket = 3  # 오른쪽 가까이
            else:
                rel_dx_bucket = 4  # 오른쪽 멀리

            # dy 이산화: 0~80 / 80~160 / 160~
            if dy < 80:
                rel_dy_bucket = 0
            elif dy < 160:
                rel_dy_bucket = 1
            else:
                rel_dy_bucket = 2

            extra = {
                "player": player,
                "enemy": e,
                "has_danger": has_danger,
                "dx": dx,
                "dy": dy,
            }
        else:
            # 적이 전혀 없는 경우
            extra = {
                "player": player,
                "enemy": None,
                "has_danger": 0,
                "dx": 0.0,
                "dy": 9999.0,
            }

        # 플레이어 x를 버킷화
        bucket_w = self.width / DISCRETE_X_BUCKETS
        x_bucket = int(np.clip((px // bucket_w), 0, DISCRETE_X_BUCKETS - 1))

        # 상태 인덱스 합치기
        # 구조: x_bucket | rel_dx_bucket | rel_dy_bucket | has_danger
        state_id = (
            x_bucket * (DISCRETE_DX_BUCKETS * DISCRETE_DY_BUCKETS * 2)
            + rel_dx_bucket * (DISCRETE_DY_BUCKETS * 2)
            + rel_dy_bucket * 2
            + has_danger
        )
        extra["x_bucket"] = x_bucket
        extra["rel_dx_bucket"] = rel_dx_bucket
        extra["rel_dy_bucket"] = rel_dy_bucket

        return int(state_id), extra


# =============================
# 액션 마스크
# =============================
def make_action_mask(extra: Dict[str, Any], mask_cfg: Dict[str, Any]) -> np.ndarray:
    """
    extra: encode()에서 나온 dict (player, enemy 포함)
    mask_cfg: pkl에 저장되는 마스크 설정
    반환: shape (3,)의 bool 배열. True면 사용가능.
    """
    player = extra["player"]
    enemy = extra.get("enemy", None)

    px, py, pw, ph, pspeed = player
    width = mask_cfg.get("width", 750)
    left_margin = mask_cfg.get("left_margin", 10)
    right_margin = mask_cfg.get("right_margin", 10)
    danger_y = mask_cfg.get("danger_y", 200.0)
    predict_collision = mask_cfg.get("predict_collision", True)
    allow_stay_if_all_blocked = mask_cfg.get("allow_stay_if_all_blocked", True)

    can_stay = True
    can_left = True
    can_right = True

    # 1) 경계 마스크
    if px <= left_margin:
        can_left = False
    if (px + pw) >= (width - right_margin):
        can_right = False

    # 2) 예측 충돌 마스크
    if predict_collision and enemy is not None:
        ex, ey, ew, eh, es, ea = enemy
        # 수직으로 가까울 때만 검사
        if 0 < (py - ey) < danger_y:
            e_seg = (ex, ex + ew)

            # stay
            p_seg_now = (px, px + pw)
            if overlap(p_seg_now, e_seg):
                can_stay = False

            # 이동량: 플레이어 속도 없으면 20px로 가정
            dx = pspeed if pspeed > 0 else 20.0

            # left
            p_seg_left = (px - dx, px + pw - dx)
            if overlap(p_seg_left, e_seg):
                can_left = False

            # right
            p_seg_right = (px + dx, px + pw + dx)
            if overlap(p_seg_right, e_seg):
                can_right = False

    mask = np.array([can_stay, can_left, can_right], dtype=bool)

    # 3) 모두 막힌 경우 복구
    if not mask.any() and allow_stay_if_all_blocked:
        mask[0] = True

    return mask


# =============================
# 보상 계산
# =============================
def compute_reward(
    extra: Dict[str, Any],
    next_extra: Dict[str, Any],
    terminated: bool,
    truncated: bool,
    reward_cfg: Dict[str, Any],
    prev_action: Optional[int],
    action: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    이론서에서 만든 합성보상 그대로 구현
    """
    alive_reward = reward_cfg.get("alive_reward", 1.0)
    crash_penalty = reward_cfg.get("crash_penalty", 1000.0)
    danger_penalty = reward_cfg.get("danger_penalty", 0.5)
    switch_penalty = reward_cfg.get("switch_penalty", 0.1)
    danger_y = reward_cfg.get("danger_y", 200.0)

    reward = 0.0
    logs = {
        "danger": 0,
        "crash": 0,
        "switched": 0,
    }

    # 1) 살아있으면 +1
    reward += alive_reward

    # 2) 위협 근접 패널티
    enemy = extra.get("enemy", None)
    player = extra.get("player")
    if enemy is not None and player is not None:
        px, py, pw, ph, _ = player
        ex, ey, ew, eh, _, _ = enemy
        if 0 < (py - ey) < danger_y:
            if overlap((px, px + pw), (ex, ex + ew)):
                reward -= danger_penalty
                logs["danger"] = 1

    # 3) 행동변경 패널티
    if prev_action is not None and prev_action != action:
        reward -= switch_penalty
        logs["switched"] = 1

    # 4) 충돌 시 큰 패널티
    if terminated or truncated:
        reward -= crash_penalty
        logs["crash"] = 1

    return reward, logs


# =============================
# Expected SARSA 기반 에이전트
# =============================
class YourAgent(kym.Agent):
    """
    kym.evaluate(...) 에서 불러서 사용할 수 있는 형태
    """

    def __init__(
        self,
        q_table: Dict[int, np.ndarray] | None = None,
        reward_cfg: Dict[str, Any] | None = None,
        mask_cfg: Dict[str, Any] | None = None,
        env_cfg: Dict[str, Any] | None = None,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.2,
    ):
        super().__init__()
        self.q_table: Dict[int, np.ndarray] = q_table if q_table is not None else {}
        self.reward_cfg = reward_cfg if reward_cfg is not None else {
            "alive_reward": 1.0,
            "crash_penalty": 1000.0,
            "danger_penalty": 0.5,
            "switch_penalty": 0.1,
            "danger_y": 200.0,
        }
        self.mask_cfg = mask_cfg if mask_cfg is not None else {
            "width": 750,
            "left_margin": 10,
            "right_margin": 10,
            "danger_y": 200.0,
            "predict_collision": True,
            "allow_stay_if_all_blocked": True,
        }
        self.env_cfg = env_cfg if env_cfg is not None else {
            "env_id": ENV_ID,
            "obs_type": OBS_TYPE,
        }
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # 관찰 인코더
        width = self.mask_cfg.get("width", 750)
        height = self.env_cfg.get("height", 600)
        danger_y = self.mask_cfg.get("danger_y", 200.0)
        self.encoder = StateEncoder(width=width, height=height, danger_y=danger_y)

        # 평가 때 행동변경 패널티 계산용
        self.prev_action_runtime: Optional[int] = None

    # -------------------------
    # kym.Agent 필수 구현
    # -------------------------
    def act(self, observation: Any, info: Dict) -> int:
        state, extra = self.encoder.encode(observation)
        mask = make_action_mask(extra, self.mask_cfg)
        action = self._select_action(state, mask, eval_mode=True)
        self.prev_action_runtime = action
        return int(action)

    @classmethod
    def load(cls, path: str) -> "kym.Agent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(
            q_table=data["model"]["q_table"],
            reward_cfg=data["reward_cfg"],
            mask_cfg=data["mask_cfg"],
            env_cfg=data["env_cfg"],
            alpha=data["model"]["alpha"],
            gamma=data["model"]["gamma"],
            epsilon=data["model"]["epsilon"],
        )
        return agent

    def save(self, path: str, episodes_trained: int = 0, training_meta: Dict[str, Any] | None = None):
        data = {
            "version": 1,
            "env_cfg": self.env_cfg,
            "reward_cfg": self.reward_cfg,
            "mask_cfg": self.mask_cfg,
            "model": {
                "q_table": self.q_table,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
            },
            "training_meta": training_meta if training_meta is not None else {
                "episodes_trained": episodes_trained,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # -------------------------
    # 내부 메서드
    # -------------------------
    def _get_q_row(self, state: int) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(NUM_ACTIONS, dtype=np.float32)
        return self.q_table[state]

    def _select_action(self, state: int, mask: np.ndarray, eval_mode: bool = False) -> int:
        q_row = self._get_q_row(state)
        masked_q = q_row.copy()
        masked_q[~mask] = -1e9

        if eval_mode:
            return int(np.argmax(masked_q))

        if np.random.rand() < self.epsilon:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions))
        else:
            return int(np.argmax(masked_q))

    def update_expected_sarsa(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_mask: np.ndarray,
    ):
        q_next = self._get_q_row(next_state)
        valid_actions = np.where(next_mask)[0]
        probs = np.zeros(NUM_ACTIONS, dtype=np.float32)

        if len(valid_actions) == 1:
            probs[valid_actions[0]] = 1.0
        else:
            best_a = valid_actions[np.argmax(q_next[valid_actions])]
            eps = self.epsilon
            for a in valid_actions:
                if a == best_a:
                    probs[a] = 1.0 - eps
                else:
                    probs[a] = eps / (len(valid_actions) - 1)

        expected_next = float(np.sum(probs * q_next))

        q_row = self._get_q_row(state)
        td_target = reward + self.gamma * expected_next
        td_error = td_target - q_row[action]
        q_row[action] += self.alpha * td_error


# =============================
# 학습 루프
# =============================
def train_agent(
    episodes: int,
    save_dir: str,
    ckpt_every: int = 2000,
    render: str = "none",
    seed: int | None = None,
    log_file: str | None = None,
):
    """
    Expected SARSA로 학습
    - 2000에피마다 pkl 저장
    - 에피소드마다 한 줄 로그
    - log_file이 있으면 파일에도 같은 형식으로 남김
    """
    ensure_dir(save_dir)

    # 로그파일 열기 (append)
    log_fh = None
    if log_file:
        # 파일만 주고 디렉터리를 안 준 경우도 있으니까 보호
        log_dir = os.path.dirname(log_file)
        if log_dir:
            ensure_dir(log_dir)
        log_fh = open(log_file, "a", encoding="utf-8")

    def log_line(msg: str):
        print(msg)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    # env 생성
    env = gym.make(
        id=ENV_ID,
        render_mode=None if render == "none" else render,
        bgm=False,
        obs_type=OBS_TYPE,
    )

    # 폭/높이 추출
    width = 750
    height = 600
    if hasattr(env.unwrapped, "width"):
        width = int(env.unwrapped.width)
    if hasattr(env.unwrapped, "height"):
        height = int(env.unwrapped.height)

    # 에이전트 생성
    agent = YourAgent(
        q_table=None,
        reward_cfg={
            "alive_reward": 1.0,
            "crash_penalty": 1000.0,
            "danger_penalty": 0.5,
            "switch_penalty": 0.1,
            "danger_y": 200.0,
        },
        mask_cfg={
            "width": width,
            "left_margin": 10,
            "right_margin": 10,
            "danger_y": 200.0,
            "predict_collision": True,
            "allow_stay_if_all_blocked": True,
        },
        env_cfg={
            "env_id": ENV_ID,
            "obs_type": OBS_TYPE,
            "width": width,
            "height": height,
        },
        alpha=0.1,
        gamma=0.99,
        epsilon=0.2,
    )

    if seed is not None:
        np.random.seed(seed)

    episodes_trained = 0
    recent_lengths: List[int] = []
    recent_rewards: List[float] = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state, extra = agent.encoder.encode(obs)
        done = False
        ep_len = 0
        ep_reward_sum = 0.0
        danger_cnt = 0
        blocked_cnt = 0
        prev_action = None

        while not done:
            # 1) 마스크
            mask = make_action_mask(extra, agent.mask_cfg)
            if not mask.any():
                blocked_cnt += 1

            # 2) 행동 선택
            action = agent._select_action(state, mask, eval_mode=False)

            # 3) 환경 진행
            next_obs, _, terminated, truncated, info = env.step(action)

            # 4) 다음 상태 인코딩
            next_state, next_extra = agent.encoder.encode(next_obs)

            # 5) 보상 계산
            reward, r_logs = compute_reward(
                extra,
                next_extra,
                terminated,
                truncated,
                agent.reward_cfg,
                prev_action,
                action,
            )

            ep_reward_sum += reward
            ep_len += 1
            danger_cnt += r_logs["danger"]

            # 6) Expected SARSA 업데이트
            next_mask = make_action_mask(next_extra, agent.mask_cfg)
            agent.update_expected_sarsa(state, action, reward, next_state, next_mask)

            # 7) 종료 체크
            done = bool(terminated or truncated)
            state, extra = next_state, next_extra
            prev_action = action

        # 에피소드 끝
        episodes_trained += 1
        recent_lengths.append(ep_len)
        recent_rewards.append(ep_reward_sum)
        if len(recent_lengths) > 200:
            recent_lengths.pop(0)
        if len(recent_rewards) > 200:
            recent_rewards.pop(0)

        # 에피소드 로그 (한 줄, 파일에도)
        log_line(
            f"[EP {episodes_trained}] len={ep_len} R={ep_reward_sum:.1f} "
            f"danger={danger_cnt} blocked={blocked_cnt}"
        )

        # ε 감소
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        # 2000ep마다 저장
        if episodes_trained % ckpt_every == 0:
            avg_len = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            max_len = int(np.max(recent_lengths)) if recent_lengths else 0
            ckpt_name = f"avoid_{episodes_trained:05d}.pkl"
            ckpt_path = os.path.join(save_dir, ckpt_name)
            training_meta = {
                "episodes_trained": episodes_trained,
                "avg_len_recent": avg_len,
                "max_len_recent": max_len,
                "time": time.time(),
            }
            agent.save(ckpt_path, episodes_trained=episodes_trained, training_meta=training_meta)
            log_line(
                f"[CKPT] ep={episodes_trained} avg_len={avg_len:.1f} max_len={max_len} "
                f"epsilon={agent.epsilon:.3f} saved={ckpt_path}"
            )

    env.close()
    if log_fh is not None:
        log_fh.close()


# =============================
# CLI
# =============================
def main():
    parser = argparse.ArgumentParser(description="Avoid Blurp Expected SARSA agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="학습 실행")
    p_train.add_argument("--episodes", type=int, default=30000, help="학습할 에피소드 수")
    p_train.add_argument("--save-dir", type=str, default="ckpt", help="체크포인트 저장 폴더")
    p_train.add_argument("--ckpt-every", type=int, default=2000, help="몇 에피소드마다 저장할지")
    p_train.add_argument("--render", type=str, default="none", choices=["none", "human", "rgb_array"])
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--log-file", type=str, default=None, help="로그를 저장할 파일 경로")

    # eval (중간 pkl 테스트용)
    p_eval = sub.add_parser("eval", help="저장된 pkl을 불러와 환경에서 실행")
    p_eval.add_argument("--pkl", type=str, required=True, help="불러올 pkl 경로")
    p_eval.add_argument("--render", type=str, default="human", choices=["human", "rgb_array"])
    p_eval.add_argument("--episodes", type=int, default=2, help="몇 에피소드 시연할지")

    args = parser.parse_args()

    if args.cmd == "train":
        train_agent(
            episodes=args.episodes,
            save_dir=args.save_dir,
            ckpt_every=args.ckpt_every,
            render=args.render,
            seed=args.seed,
            log_file=args.log_file,
        )
    elif args.cmd == "eval":
        agent = YourAgent.load(args.pkl)
        for _ in range(args.episodes):
            kym.evaluate(
                env_id=ENV_ID,
                agent=agent,
                render_mode=args.render,
                bgm=True if args.render == "human" else False,
                obs_type=OBS_TYPE,
            )


if __name__ == "__main__":
    main()
