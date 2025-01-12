from kaggle_environments import make

env = make("chess", debug=True)
result = env.run(["agents/agent_1.py", "random"])
print("Agent exit status/reward/time left: ")

# look at the generated replay.json and print out the agent info
for agent in result[-1]:
    print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)
print("\n")

# render the game
env.render(mode='ipython', width=600, height=600)

