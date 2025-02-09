import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.quantization

from Chessnut import Game
from kaggle_environments import make
from math import sqrt, log

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return f.relu(out)
    
class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.buffer = []
        
        self.conv1 = nn.Conv2d(19, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.res_block = ResidualBlock(256)
        self.conv2 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1)

        self.fc1 = nn.Linear(8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.to(device)
        x = f.relu(self.bn(self.conv1(x)))
        x = self.res_block(x)
        x = f.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.tanh(self.fc2(x))
    
    def save_to_buffer(self, board):
        self.buffer.append(board)

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
            if self.is_opponent:
                value = 1.0 - value
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

        if(not self.is_opponent):
            input_tensor = Create_input_tensor(Game(best_node.fen))
            self.value_network.save_to_buffer(input_tensor)

        return best_node.move

class SnapshotManager:
    def __init__(self, window: int = 4, replace_interval: int = 5):
        self.window = window
        self.snapshots = []
        self.oldest_index = -1
        self.replace_interval = replace_interval

    def addSnapshot(self, value_network: ValueNetwork):
        if len(self.snapshots) < self.window:
            new_snapshot = type(value_network)().to(device)
            new_snapshot.load_state_dict(value_network.state_dict())
            self.snapshots.append(new_snapshot)
        else:
            self.snapshots[self.oldest_index].load_state_dict(value_network.state_dict())
        self.oldest_index = (self.oldest_index + 1) % self.window

    def select(self) -> ValueNetwork:
        if not self.snapshots:
            raise ValueError("No snapshots available for selection.")

        weights = [i + 1 for i in range(len(self.snapshots))]
        selected_index = random.choices(range(len(self.snapshots)), weights=weights, k=1)[0]
        return self.snapshots[selected_index]

def Create_input_tensor(game: Game):
    bitboards = board_to_bitboards(game.board)
    input_tensor = bitboards_to_tensor(bitboards).unsqueeze(0)  # (12, 8, 8) -> (1, 12, 8, 8)
    
    _, _, castling_fen, en_passant_fen = game.get_fen().split()[:4]

    player_turn = torch.tensor([[[[1] * 8] * 8]]) if game.state.player == 'b' else torch.tensor([[[[0] * 8] * 8]])  # (1, 1, 8, 8)
    en_passant = torch.tensor([[[[1] * 8] * 8]]) if en_passant_fen != '-' else torch.tensor([[[[0] * 8] * 8]])  # (1, 1, 8, 8)
    move_count = torch.full((1, 1, 8, 8), game.state.ply / 50.0, dtype=torch.float32)  # 정규화된 move count
    castling = torch.tensor([
        [[[1] * 8] * 8] if 'K' in castling_fen else [[[0] * 8] * 8],  # 백 킹사이드
        [[[1] * 8] * 8] if 'Q' in castling_fen else [[[0] * 8] * 8],  # 백 퀸사이드
        [[[1] * 8] * 8] if 'k' in castling_fen else [[[0] * 8] * 8],  # 흑 킹사이드
        [[[1] * 8] * 8] if 'q' in castling_fen else [[[0] * 8] * 8],  # 흑 퀸사이드
    ], dtype=torch.float32)  # (4, 1, 8, 8)

    special_rules = torch.cat((player_turn, en_passant, move_count, castling), dim=0).squeeze(1)  # (7, 8, 8)
    
    input_tensor = torch.cat((input_tensor.squeeze(0), special_rules), dim=0)  # (12 + 7, 8, 8)

    return input_tensor.unsqueeze(0)  # 배치 차원 추가 (1, 19, 8, 8)

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

def finalize_buffer(value_network, game_result):
    updated_buffer = []

    for state in value_network.buffer:
        updated_buffer.append((state, float(game_result)))  # y 값 추가
    value_network.buffer = updated_buffer

def train_value_network(value_network:ValueNetwork, buffer, learning_rate, epochs=25, batch_size=32):
    value_network.train() 

    optimizer = optim.Adam(value_network.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()  # 손실 함수

    for epoch in range(epochs):
        random.shuffle(buffer)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(buffer), batch_size):
            batch = buffer[i:i + batch_size]

            inputs = torch.stack([item[0] for item in batch]).to(device)  # [batch_size, 19, 8, 8]
            if len(inputs.shape) == 5:
                inputs = inputs.squeeze(1)  # [batch_size, 19, 8, 8]

            # 라벨 준비
            targets = torch.tensor([item[1] for item in batch]).to(device) 
            
            # 학습 단계
            optimizer.zero_grad()
            outputs = value_network(inputs)
            loss = loss_fn(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            # 손실 누적
            epoch_loss += loss.item()
            num_batches += 1

        # log
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    value_network.eval()

def save_model(value_network, path="value_network.pth", episode=0):
    torch.save({
        "episode": episode,
        "model_state_dict": value_network.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def apply_dynamic_quantization(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear}, 
        dtype=torch.qint8
    )
    return quantized_model

def load_model(value_network, path="value_network.pth"):
    checkpoint = torch.load(path)
    value_network.load_state_dict(checkpoint["model_state_dict"])
    starting_episode = checkpoint.get("episode", 0)
    print(f"Model loaded from {path}, starting at episode {starting_episode}")

    return starting_episode

def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def chess_bot(obs, mcts:MCTS, iteration = 15):
    move = mcts.search(iteration, obs.board)
    return move

# RL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
value_network = ValueNetwork().to(device)
value_network.apply(init_weights_xavier)
value_network.load_state_dict(torch.load("models/value_network_episode_1562.pth"), strict=False)
mcts = MCTS(value_network, exploration_weight=1.41)
buffer = []
buffer_threshold = 4096
minibatch_size = 256

total_episodes = 200000
iteration = 30
epochs = 10
learning_rate = 0.008

# Logs
game_history = [0, 0, 0]

snapshot_n = 0

snapshotManager = SnapshotManager(window=4)
snapshotManager.addSnapshot(value_network)
mcts_opponent = MCTS(snapshotManager.select(), is_opponent=True, exploration_weight=1.41)

env = make("chess", debug=True)
for episode in range(1600, total_episodes+1):
    print(f"\nEpisode: {episode}")

    env.configuration["seed"] = random.randint(0, 10000)
    result = env.run([lambda obs: chess_bot(obs, mcts, iteration), lambda obs: chess_bot(obs, mcts_opponent, iteration)])

    # Log
    print("Agent exit status/reward/time left: ")
    for agent in result[-1]:
        print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

    # Reward
    reward = 0
    if result[-1][1].reward == None or result[-1][1].reward == 0.5:
        reward = 0
        game_history[1] += 1
    elif result[-1][1].reward == 0:
        reward = 1
        game_history[0] += 1
    elif result[-1][1].reward == 1:
        reward = -1
        game_history[2] += 1
    print(f"{game_history[0]} wins {game_history[1]} draws {game_history[2]} losses")
    
    # Handle buffer
    finalize_buffer(value_network, reward)
    buffer.extend(value_network.buffer) 
    value_network.buffer.clear()
    print(f"buffer size: {len(buffer)}/{buffer_threshold}")
    
    if len(buffer) >= buffer_threshold:
        print(f"\nTraining on buffer at Episode {episode}")
        train_value_network(value_network, buffer, learning_rate, epochs, minibatch_size)
        buffer.clear()
        game_history = [0, 0, 0]

        snapshot_n += 1
        snapshotManager.addSnapshot(value_network)
        print("Snapshot generated")

        win_rate = game_history[0] / sum(game_history) if sum(game_history) > 0 else 0
        if win_rate >= 0.55:
            snapshot_n = 0
            mcts_opponent.value_network = snapshotManager.select()
            print(f"Adaptive Snapshot Replacement: Win rate {win_rate:.2f}")
            save_model(value_network, path=f"models/value_network_episode_{episode}.pth", episode=episode)
            quantized_model = apply_dynamic_quantization(value_network)
            torch.save(quantized_model.state_dict(), f"models/quantized/q_value_network_episode_{episode}.pth")

        # if snapshot_n % snapshotManager.replace_interval == 0:
        #     snapshot_n = 0
        #     mcts_opponent.value_network = snapshotManager.select()
        #     print(f"Snapshot selected")

        #     save_model(value_network, path=f"models/value_network_episode_{episode}.pth", episode=episode)
        #     quantized_model = apply_dynamic_quantization(value_network)
        #     torch.save(quantized_model.state_dict(), f"models/quantized/q_value_network_episode_{episode}.pth")

        env.render(mode='ipython', width=600, height=600)