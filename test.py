import kymnasium as kym
from RL import YourAgent   # RL.py와 같은 폴더여야 함

# 학습된 에이전트 불러오기
agent = YourAgent.load(r"C:\Users\Samsung\Desktop\일\알고리즘\value_iter_agent.pkl")

# 평가 실행
kym.evaluate(
    env_id="kymnasium/GridAdventure-Crossing-26x26-v0",
    agent=agent,
    render_mode="human",  # 또는 "rgb_array"
    bgm=True
)
