from kaggle_environments import make
from agents import agent_2_mcts  # agent_2_mcts.py 파일이 agents 폴더에 있어야 합니다.
import random

# Kaggle chess environment 생성
env = make("chess", debug=True)
env.configuration["seed"] = random.randint(0,10000)
# 에이전트 실행
result = env.run([lambda obs: agent_2_mcts.chess_bot(obs), "random"])

# 에이전트의 상태 및 결과 출력
print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

# 게임 렌더링
env.render(mode='ipython', width=600, height=600)
