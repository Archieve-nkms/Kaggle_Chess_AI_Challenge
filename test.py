from Chessnut import Game


game = Game("6k1/5pp1/7N/2K5/2B5/8/8/8 b - - 0 1")

for move in game.get_moves():
    print(f"{move}")