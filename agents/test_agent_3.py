import torch
import torch.nn as nn
import torch.nn.functional as f
from Chessnut import Game
from math import sqrt, log

def chess_bot(obs):
    mcts = MCTS(obs.board, obs.mark) # FEN
    return mcts.search(10)

def board_to_bitboards(game_board):
    position = game_board._position
    bitboards = {piece: 0 for piece in "PNBRQKpnbrqk"}

    for i, piece in enumerate(position):
        if piece != " ":
            bitboards[piece] |= (1 << (63 - i))

    return bitboards

def bitboards_to_tensor(bitboards):
    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for i, (piece, bitboard) in enumerate(bitboards.items()):
        for j in range(64):
            row, col = divmod(j, 8)
            tensor[i, row, col] = (bitboard >> (63 - j)) & 1

    return tensor

class Node:
    def __init__(self, fen, move, parent=None):
        game = Game(fen)
        self.fen = fen
        self.parent = parent
        self.move = move
        self.visits = 1
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
        self.value_network = ValueNetwork()

    def _select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.get_best_child()
        return node

    def _expand(self, node):
        move = node.untried_moves.pop()
        return node.expand(move)

    def _simulate(self, fen):
        game = Game(fen)

        bitboards = board_to_bitboards(game.board)
        input_tensor = bitboards_to_tensor(bitboards).unsqueeze(0)
        score = self.value_network.forward(input_tensor)
        self.value_network.save_to_buffer(input_tensor, score.item())

        return score.item()


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
        return self.root.get_best_child().move
    

"""
Input: bitboard로 표현된 기물의 위치.(백폰, 백나이트, 백룩, ...) 12 x 기물 위치(8바이트)
Output: 보드 평가치 (-1.0 - 1.0)
"""
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.buffer = []  # 게임이 종료되기 전까지 state와 y_hat을 보관할 버퍼
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
    
    def save_to_buffer(self, input, score):
        self.buffer.append({input, score})