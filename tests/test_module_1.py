from kaggle_environments import make

env = make("chess", debug=True)
result = env.run(["random", "random"])
env.render(mode='ipython', width=600, height=600)
