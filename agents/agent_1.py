from Chessnut import Game
import random

def chess_bot(obs):
    game = Game(obs.board)
    moves = list(game.get_moves())
    for move in moves[:10]:
        g = Game(obs.board)
        g.apply_move(move)
        if g.status == Game.CHECKMATE:
            return move
        
    for move in moves:
        if game.board.get_piece(Game.xy2i(move[2:4])) != ' ':
            return move
            
    for move in moves:
        if "q" in move.lower():
            return move
            
    return random.choice(moves)
