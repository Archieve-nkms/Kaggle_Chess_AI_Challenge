import random
import os.path

from agents import test_agent_3
from kaggle_environments import make

print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))

env = make("chess", debug=True)
env.configuration["seed"] = random.randint(0,10000)
result = env.run([lambda obs: test_agent_3.chess_bot(obs), "agents/agent_1.py"])

print("Agent exit status/reward/time left: ")
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

env.render(mode='ipython', width=600, height=600)
