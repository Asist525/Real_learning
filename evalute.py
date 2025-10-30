# evaluate.py
import kymnasium as kym
from agent import YourAgent   # agent.py랑 같은 폴더라고 가정
import sys

def main():
    if len(sys.argv) < 2:
        print("usage: python evaluate.py <pkl_path>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    agent = YourAgent.load(pkl_path)

    kym.evaluate(
        env_id='kymnasium/AvoidBlurp-Normal-v0',
        agent=agent,
        render_mode='human',   # 창 띄워서 확인
        bgm=True,
        obs_type='custom'
    )

if __name__ == "__main__":
    main()
