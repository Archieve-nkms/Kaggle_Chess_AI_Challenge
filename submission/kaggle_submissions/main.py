import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt, log
from Chessnut import Game
from torch.ao.quantization import QuantStub, DeQuantStub

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(19, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.dequant(x)
        return torch.tanh(self.fc2(x))
    
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
        self.win_prop = None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def is_leaf(self):
        return len(self.children) == 0

    def get_best_child(self, exploration_weight=1.41421356237):
        parent_visits_log = log(self.visits)

        return max(
            self.children,
            key=lambda child: child.wins / child.visits + exploration_weight * child.win_prop * sqrt(parent_visits_log / child.visits)
        )

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
    def __init__(self, value_network, is_opponent = False, exploration_weight = 2):
        self.root = None
        self.value_network = value_network
        self.is_opponent = is_opponent
        self.exploration_weight = exploration_weight

    def _select(self, node):
        while node.is_fully_expanded() and not node.is_leaf():
            node = node.get_best_child(exploration_weight = self.exploration_weight)
        return node

    def _expand(self, node):
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        return node.expand(move)

    def _simulate(self, fen):
        with torch.no_grad():
            input_tensor = Create_input_tensor(Game(fen))
            value = self.value_network.forward(input_tensor).item()
        return value
        
    def _backpropagate(self, node, result):
        while node is not None:
            node.update(result)
            node = node.parent

    def search(self, iterations, fen):
        self.root = Node(fen, None)
        for _ in range(iterations):
            # Selection
            node = self._select(self.root)

            # Expansion
            if not node.is_fully_expanded():
                node = self._expand(node)

            # Simulation
            result = self._simulate(node.fen)
            node.win_prop = result

            # Backpropagation
            self._backpropagate(node, result)

        best_node = self.root.get_best_child(exploration_weight=0)

        return best_node.move

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

def Create_input_tensor(game: Game):
    bitboards = board_to_bitboards(game.board)
    input_tensor = bitboards_to_tensor(bitboards).unsqueeze(0)
    
    _, _, castling_fen, en_passant_fen = game.get_fen().split()[:4]

    player_turn = torch.tensor([[[[1] * 8] * 8]]) if game.state.player == 'b' else torch.tensor([[[[0] * 8] * 8]])
    en_passant = torch.tensor([[[[1] * 8] * 8]]) if en_passant_fen != '-' else torch.tensor([[[[0] * 8] * 8]])
    move_count = torch.full((1, 1, 8, 8), game.state.ply / 50.0, dtype=torch.float32)
    castling = torch.tensor([
        [[[1] * 8] * 8] if 'K' in castling_fen else [[[0] * 8] * 8],
        [[[1] * 8] * 8] if 'Q' in castling_fen else [[[0] * 8] * 8],
        [[[1] * 8] * 8] if 'k' in castling_fen else [[[0] * 8] * 8],
        [[[1] * 8] * 8] if 'q' in castling_fen else [[[0] * 8] * 8],
    ], dtype=torch.float32)

    special_rules = torch.cat((player_turn, en_passant, move_count, castling), dim=0).squeeze(1)
    
    input_tensor = torch.cat((input_tensor.squeeze(0), special_rules), dim=0)

    return input_tensor.unsqueeze(0)

# 모델 및 MCTS 초기화
model = ValueNetwork()
model.load_state_dict(torch.load("/kaggle_simulations/agent/q_value_network_episode_5.pth"), strict=False)
model.eval()
mcts = MCTS(model)

def chess_bot(obs, config):
    move = mcts.search(1, obs.board)
    return move
