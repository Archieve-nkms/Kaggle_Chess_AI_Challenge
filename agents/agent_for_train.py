import random
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.quantization
import time

from Chessnut import Game
from math import sqrt, log
from kaggle_environments import make

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.buffer = []  # 게임이 종료되기 전까지 state와 y_hat을 보관할 버퍼
        self.conv1 = nn.Conv2d(19, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.to(device)
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.tanh(self.fc2(x))
    
    def save_to_buffer(self, input, score):
        self.buffer.append((input, score))

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

    def get_best_child(self, exploration_weight=1.41421356237):
        parent_visits_log = log(self.visits)

        return max(
            self.children,
            key=lambda child: child.wins / child.visits + exploration_weight * sqrt(parent_visits_log / child.visits)
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
    def __init__(self, value_network):
        self.root = None
        self.value_network = value_network

    def _select(self, node):
        while node.is_fully_expanded() and not node.is_leaf():
            node = node.get_best_child()
        return node

    def _expand(self, node):
        move = node.untried_moves.pop()
        return node.expand(move)

    def _simulate(self, fen):
        input_tensor = Create_input_tensor(Game(fen))
        with torch.no_grad():
            return self.value_network.forward(input_tensor).item()

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

            # Backpropagation
            self._backpropagate(node, result)

        best_node = self.root.get_best_child(exploration_weight=0)
        input_tensor = Create_input_tensor(Game(best_node.fen))
        with torch.no_grad():
            score = self.value_network.forward(input_tensor).item()
        self.value_network.save_to_buffer(input_tensor, score)

        return best_node.move
    
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

def train_value_network(value_network, buffer, learning_rate, epochs=25, batch_size=32):
    optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
    loss_fn = nn.HuberLoss()  # 손실 함수

    for epoch in range(epochs):
        random.shuffle(buffer)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(buffer), batch_size):
            batch = buffer[i:i + batch_size]

            # 상태 텐서 준비
            inputs = torch.stack([item[0] for item in batch]).to(device)  # [batch_size, 19, 8, 8]
            if len(inputs.shape) == 5:  # 5D로 들어올 경우
                inputs = inputs.squeeze(1)  # [batch_size, 19, 8, 8]

            # 라벨 준비
            targets = torch.tensor([item[2] for item in batch]).to(device)  # [batch_size]

            # 학습 단계
            optimizer.zero_grad()
            outputs = value_network(inputs)
            loss = loss_fn(outputs.squeeze(), targets)  # 손실 계산
            loss.backward()
            optimizer.step()

            # 손실 누적
            epoch_loss += loss.item()
            num_batches += 1

        # log
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def finalize_buffer(value_network, game_result):
    updated_buffer = []

    for state, y_hat in value_network.buffer:
        updated_buffer.append((state, y_hat, float(game_result)))  # y 값 추가
    value_network.buffer = updated_buffer

def save_model(value_network, path="value_network.pth", episode=0):
    torch.save({
        "episode": episode,
        "model_state_dict": value_network.state_dict(),
    }, path)
    print(f"Model saved to {path}")

def apply_quantization(value_network, path="quantized_value_network.pth", episode=0):
    value_network.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        value_network,  # 모델
        {torch.nn.Linear},  # 양자화를 적용할 레이어 (Linear에만 적용)
        dtype=torch.qint8  # INT8 사용
    )

    torch.save({
        "episode": episode,
        "model_state_dict": quantized_model.state_dict(),
    }, path)
    print(f"Quantized model saved to {path}")

    return quantized_model

def load_model(value_network, path="value_network.pth"):
    checkpoint = torch.load(path)
    value_network.load_state_dict(checkpoint["model_state_dict"])
    starting_episode = checkpoint.get("episode", 0)
    print(f"Model loaded from {path}, starting at episode {starting_episode}")

    return starting_episode

def load_quantized_model(value_network_class, path="quantized_value_network.pth"):
    checkpoint = torch.load(path)
    quantized_model = torch.quantization.quantize_dynamic(
        value_network_class(),
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    quantized_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Quantized model loaded from {path}, starting at episode {checkpoint.get('episode', 0)}")

    return quantized_model, checkpoint.get("episode", 0)

def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def chess_bot(obs, mcts:MCTS, iteration = 10):
    start_time = time.time()

    move = mcts.search(iteration, obs.board)
    
    times.append(time.time() - start_time)
    return move

def add_snapshot():
    #TODO
    pass




# RL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
value_network = ValueNetwork().to(device)
value_network.apply(init_weights_kaiming)
mcts = MCTS(value_network)
buffer = []
buffer_threshold = 4096
model_save_interval = 1000

# Parameters
total_episodes = 200000 
iteration = 10
epochs = 15
learning_rate = 0.2

# Snapshot
snapshots = []
snapshot_index = 0
snapshot_window = 4
snapshot_replace_interval = 1000

# Logs
game_history = [0, 0, 0] 
times = []

print(f"Tensorflow {tf.__version__}")
print(f"Current Dir: {os.getcwd()}")
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")

env = make("chess", debug=True)
for episode in range(1, total_episodes+1):
    print(f"\nEpisode: {episode}")
    progress = episode / total_episodes
    if 0.2 < progress and progress <= 0.8:
        iteration = 30
        epochs = 30
        learning_rate = 0.02
    elif 0.8 < progress: 
        iteration = 100
        epochs = 20
        learning_rate = 0.002

    env.configuration["seed"] = random.randint(0, 10000)
    result = env.run([lambda obs: chess_bot(obs, mcts, iteration), lambda obs: chess_bot(obs, snapshots[snapshot_index], iteration)])

    avg_time = sum(times) / len(times)
    times.clear()

    # Log
    print(f"Avg step time: {avg_time:.2f}")
    print("Agent exit status/reward/time left: ")
    for agent in result[-1]:
        print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

    # Reward
    reward = 0
    if result[-1][1].reward == None or result[-1][1].reward == 0.5:
        reward = reward_2 = 0
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

    if len(buffer) >= buffer_threshold:
        print(f"\nTraining on buffer at Episode {episode}")
        train_value_network(value_network, buffer, learning_rate, epochs)
        buffer.clear()
        env.render(mode='ipython', width=600, height=600)
    if episode > 0 and episode % model_save_interval == 0: 
        save_model(value_network, path=f"models/value_network_episode_{episode}.pth", episode=episode)
        apply_quantization(value_network, path=f"models/q_value_network_episode_{episode}.pth", episode=episode)
    """
       if episode > 0 and episode % snapshot_replace_interval == 0:
        new_snapshot = ValueNetwork()
        new_snapshot.load_state_dict(value_network.state_dict())
        if(len(snapshots) < snapshot_window):
            snapshots.append(value_network)
        else:
            snapshots[snapshot_index] = value_network
        snapshot_index = (snapshot_index + 1) % min(len(snapshots), snapshot_window)     
    """
