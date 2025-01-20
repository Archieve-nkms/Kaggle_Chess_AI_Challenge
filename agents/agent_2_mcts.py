from Chessnut import Game
from math import sqrt, log

def chess_bot(obs):
    mcts = MCTS(obs.board, obs.mark) # FEN
    return mcts.search(20)

class Node:
    def __init__(self, fen, move, parent=None):
        game = Game(fen)
        self.fen = fen
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children = []
        self.untried_moves = list(game.get_moves())

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_leaf(self):
        return len(self.children) == 0

    def get_best_child(self):
        c = 1.41421356237  # root 2
        max_value = float("-inf")
        best_node = None
        for child in self.children:
            uct = child.wins / child.visits + c * sqrt(log(self.visits) / child.visits)
            if uct > max_value:
                max_value = uct
                best_node = child
        return best_node

    def expand(self, move):
        game = Game(self.fen)
        game.apply_move(move)
        child_node = Node(game.get_fen(), move, self)
        self.children.append(child_node)
        return child_node

    def update(self, result):
        self.visits += 1
        self.wins += result

class MCTS:
    def __init__(self, fen, player):
        self.root = Node(fen, None)
        self.player = player
        self.evaluator = ChessEvaluator()

    def _select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.get_best_child()
        return node

    def _expand(self, node):
        move = node.untried_moves.pop()
        return node.expand(move)

    def _simulate(self, fen):
        def default_policy(fen):
            game = Game(fen)

            score = self.evaluator.eval(game.board._position, 0 if self.player == "white" else 1) 

            return score
    
        score = default_policy(fen)
        return score


    def _backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def search(self, iterations):
        for _ in range(iterations):
            node = self._select(self.root)
            if not node.is_fully_expanded():
                node = self._expand(node)
            result = self._simulate(node.fen)
            self._backpropagate(node, result)
        print("\n")
        return self.root.get_best_child().move

class ChessEvaluator:

    def __init__(self):
        self.PAWN = 0
        self.KNIGHT = 1
        self.BISHOP = 2
        self.ROOK = 3
        self.QUEEN = 4 
        self.KING = 5
        self.mg_value = [82, 337, 365, 477, 1025, 0]
        self.eg_value = [94, 281, 297, 512, 936, 0]
        self.mg_pawn_table = [
              0,   0,   0,   0,   0,   0,  0,   0,
             98, 134,  61,  95,  68, 126, 34, -11,
             -6,   7,  26,  31,  65,  56, 25, -20,
            -14,  13,   6,  21,  23,  12, 17, -23,
            -27,  -2,  -5,  12,  17,   6, 10, -25,
            -26,  -4,  -4, -10,   3,   3, 33, -12,
            -35,  -1, -20, -23, -15,  24, 38, -22,
              0,   0,   0,   0,   0,   0,  0,   0
        ]
        self.eg_pawn_table = [
              0,   0,   0,   0,   0,   0,   0,   0,
            178, 173, 158, 134, 147, 132, 165, 187,
             94, 100,  85,  67,  56,  53,  82,  84,
             32,  24,  13,   5,  -2,   4,  17,  17,
             13,   9,  -3,  -7,  -7,  -8,   3,  -1,
              4,   7,  -6,   1,   0,  -5,  -1,  -8,
             13,   8,   8,  10,  13,   0,   2,  -7,
              0,   0,   0,   0,   0,   0,   0,   0
        ]
        self.mg_knight_table = [
            -167, -89, -34, -49,  61, -97, -15, -107,
             -73, -41,  72,  36,  23,  62,   7,  -17,
             -47,  60,  37,  65,  84, 129,  73,   44,
              -9,  17,  19,  53,  37,  69,  18,   22,
             -13,   4,  16,  13,  28,  19,  21,   -8,
             -23,  -9,  12,  10,  19,  17,  25,  -16,
             -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23
        ]
        self.eg_knight_table = [
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25,  -8, -25,  -2,  -9, -25, -24, -52,
            -24, -20,  10,   9,  -1,  -9, -19, -41,
            -17,   3,  22,  22,  22,  11,   8, -18,
            -18,  -6,  16,  25,  16,  17,   4, -18,
            -23,  -3,  -1,  15,  10,  -3, -20, -22,
            -42, -20, -10,  -5,  -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64  
        ]
        self.mg_bishop_table = [
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
             -4,   5,  19,  50,  37,  37,   7,  -2,
             -6,  13,  13,  26,  34,  12,  10,   4,
              0,  15,  15,  15,  14,  27,  18,  10,
              4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21
        ]
        self.eg_bishop_table = [
            -14, -21, -11,  -8, -7,  -9, -17, -24,
             -8,  -4,   7, -12, -3, -13,  -4, -14,
              2,  -8,   0,  -1, -2,   6,   0,   4,
             -3,   9,  12,   9, 14,  10,   3,   2,
             -6,   3,  13,  19,  7,  10,  -3,  -9,
            -12,  -3,   8,  10, 13,   3,  -7, -15,
            -14, -18,  -7,  -1,  4,  -9, -15, -27,
            -23,  -9, -23,  -5, -9, -16,  -5, -17
        ]
        self.mg_rook_table = [
             32,  42,  32,  51, 63,  9,  31,  43,
             27,  32,  58,  62, 80, 67,  26,  44,
             -5,  19,  26,  36, 17, 45,  61,  16,
            -24, -11,   7,  26, 24, 35,  -8, -20,
            -36, -26, -12,  -1,  9, -7,   6, -23,
            -45, -25, -16, -17,  3,  0,  -5, -33,
            -44, -16, -20,  -9, -1, 11,  -6, -71,
            -19, -13,   1,  17, 16,  7, -37, -26
        ]
        self.eg_rook_table = [
            13, 10, 18, 15, 12,  12,   8,   5,
            11, 13, 13, 11, -3,   3,   8,   3,
             7,  7,  7,  5,  4,  -3,  -5,  -3,
             4,  3, 13,  1,  2,   1,  -1,   2,
             3,  5,  8,  4, -5,  -6,  -8, -11,
            -4,  0, -5, -1, -7, -12,  -8, -16,
            -6, -6,  0,  2, -9,  -9, -11,  -3,
            -9,  2,  3, -1, -5, -13,   4, -20
        ]
        self.mg_queen_table = [
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
             -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
             -1, -18,  -9,  10, -15, -25, -31, -50
        ]
        self.eg_queen_table = [
             -9,  22,  22,  27,  27,  19,  10,  20,
            -17,  20,  32,  41,  58,  25,  30,   0,
            -20,   6,   9,  49,  47,  35,  19,   9,
              3,  22,  24,  45,  57,  40,  57,  36,
            -18,  28,  19,  47,  31,  34,  39,  23,
            -16, -27,  15,   6,   9,  17,  10,   5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43,  -5, -32, -20, -41
        ]
        self.mg_king_table = [
            -65,  23,  16, -15, -56, -34,   2,  13,
             29,  -1, -20,  -7,  -8,  -4, -38, -29,
             -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
              1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14 
        ]
        self.eg_king_table = [
            -74, -35, -18, -18, -11,  15,   4, -17,
            -12,  17,  14,  17,  17,  38,  23,  11,
             10,  17,  23,  15,  20,  45,  44,  13,
             -8,  22,  24,  27,  26,  33,  26,   3,
            -18,  -4,  21,  24,  27,  23,   9, -11,
            -19,  -3,  11,  21,  23,  16,   7,  -9,
            -27, -11,   4,  13,  14,   4,  -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43
        ]
        self.mg_pesto_table = [
            self.mg_pawn_table,
            self.mg_knight_table,
            self.mg_bishop_table,
            self.mg_rook_table,
            self.mg_queen_table,
            self.mg_king_table
        ]
        self.eg_pesto_table = [
            self.eg_pawn_table,
            self.eg_knight_table,
            self.eg_bishop_table,
            self.eg_rook_table,
            self.eg_queen_table,
            self.eg_king_table 
        ]
        self.gamephaseInc = [0, 0, 1, 1, 1, 1, 2, 2, 4, 4, 0, 0]
        self.mg_table = [[0] * 64 for _ in range(12)]
        self.eg_table = [[0] * 64 for _ in range(12)]

        self.init_tables()

    @staticmethod
    def FLIP(sq):
        return sq ^ 56

    @staticmethod
    def OTHER(side):
        return side ^ 1

    def init_tables(self):
        for p in range(self.PAWN, self.KING + 1):
            pc_white = 2 * p + 0
            pc_black = 2 * p + 1
            for sq in range(64):
                self.mg_table[pc_white][sq] = self.mg_value[p] + self.mg_pesto_table[p][sq]
                self.eg_table[pc_white][sq] = self.eg_value[p] + self.eg_pesto_table[p][sq]
                self.mg_table[pc_black][sq] = self.mg_value[p] + self.mg_pesto_table[p][self.FLIP(sq)]
                self.eg_table[pc_black][sq] = self.eg_value[p] + self.eg_pesto_table[p][self.FLIP(sq)]

    def eval(self, board, side2move):
        mg = [0, 0]
        eg = [0, 0]
        gamePhase = 0

        for i, piece in enumerate(board):
            if piece != " ":
                pc = piece.isupper()
                mg[pc] += self.mg_table[pc][i]
                eg[pc] += self.eg_table[pc][i]
                gamePhase += self.gamephaseInc[pc]

        mgScore = mg[side2move] - mg[self.OTHER(side2move)]
        egScore = eg[side2move] - eg[self.OTHER(side2move)]
        mgPhase = min(gamePhase, 24)
        egPhase = 24 - mgPhase
        return (mgScore * mgPhase + egScore * egPhase) // 24
