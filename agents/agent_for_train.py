import random
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torch.quantization
from Chessnut import Game
from math import sqrt, log
from kaggle_environments import make

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.buffer = []  # 게임이 종료되기 전까지 state와 y_hat을 보관할 버퍼
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
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

    def get_best_child(self):
        c = 1.41421356237  # sqrt(2)
        parent_visits_log = log(self.visits)

        uct_values = [
            (child.wins / child.visits + c * sqrt(parent_visits_log / child.visits), child)
            for child in self.children
        ]
        return max(uct_values, key=lambda x: x[0])[1]

    def expand(self, move):
        game = Game(self.fen)
        game.apply_move(move)
        child_node = Node(game.get_fen(), move, self)
        self.children.append(child_node)
        return child_node

class MCTS:
    def __init__(self, fen, player, value_network):
        self.root = Node(fen, None)
        self.player = player
        self.value_network = value_network

    def _select(self, node):
        while node.is_fully_expanded() and not node.is_leaf():
            node = node.get_best_child()
        return node
    
    def _expand(self, node):
        moves = node.untried_moves
        games = [Game(node.fen) for _ in moves]

        # 각 움직임에 따른 보드 상태 생성
        for game, move in zip(games, moves):
            game.apply_move(move)

        # 모든 보드를 한 번에 네트워크에 입력
        bitboards_list = [board_to_bitboards(game.board) for game in games]
        input_tensors = torch.stack([bitboards_to_tensor(bb) for bb in bitboards_list]).to(device)

        with torch.no_grad():
            scores = self.value_network(input_tensors).squeeze()  # 차원 축소
            if scores.dim() == 0:  # 스칼라 값 처리
                scores = scores.unsqueeze(0)
            scores = scores.cpu().numpy()

        # 점수가 가장 높은 움직임 선택
        best_move = moves[scores.argmax()]
            
        return node.expand(best_move)

    def _simulate(self, fen):
        game = Game(fen)

        bitboards = board_to_bitboards(game.board)
        input_tensor = bitboards_to_tensor(bitboards).unsqueeze(0).to(device)
        with torch.no_grad():
            score = self.value_network.forward(input_tensor)
        self.value_network.save_to_buffer(input_tensor, score.item())

        return score.cpu().item()

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def search(self, iterations):
        for _ in range(iterations):
            node = self._select(self.root)
            if not node.is_fully_expanded():
                node = self._expand(node)
            result = self._simulate(node.fen)
            self._backpropagate(node, result)
        return self.root.get_best_child().move
    
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

def train_value_network(value_network, buffer, epochs=10, batch_size=32):
    optimizer = optim.Adam(value_network.parameters(), lr=0.001)
    loss_fn = nn.HuberLoss()  # 손실 함수

    for epoch in range(epochs):
        random.shuffle(buffer)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(buffer), batch_size):
            batch = buffer[i:i + batch_size]

            # 상태 텐서 준비
            inputs = torch.stack([item[0] for item in batch]).to(device)  # [batch_size, 12, 8, 8]
            if len(inputs.shape) == 5:  # 5D로 들어올 경우
                inputs = inputs.squeeze(1)  # [batch_size, 12, 8, 8]

            # 라벨 준비
            targets = torch.tensor([item[2] for item in batch]).to(device)  # [batch_size]

            # 학습 단계
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = value_network(inputs)
            loss = loss_fn(outputs.squeeze(), targets)  # 손실 계산
            loss.backward()
            optimizer.step()

            # 손실 누적
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def finalize_buffer(value_network, game_result):
    updated_buffer = []

    for state, y_hat in value_network.buffer:
        updated_buffer.append((state, y_hat, float(game_result)))  # y 값 추가
    value_network.buffer = updated_buffer

def chess_bot(obs, network:ValueNetwork):
    mcts = MCTS(obs.board, obs.mark, network) # FEN
    return mcts.search(50)

def save_model(value_network, path="value_network.pth"):
    """
    모델의 state_dict를 저장.
    """
    torch.save(value_network.state_dict(), path)
    print(f"Model saved to {path}")

def apply_quantization(value_network, path="quantized_value_network.pth"):
    # 모델을 평가 모드로 전환
    value_network.eval()

    # 동적 양자화 적용
    quantized_model = torch.quantization.quantize_dynamic(
        value_network,  # 모델
        {torch.nn.Linear},  # 양자화를 적용할 레이어 (Linear에만 적용)
        dtype=torch.qint8  # INT8 사용
    )

    # 양자화된 모델 저장
    torch.save(quantized_model.state_dict(), path)
    print(f"Quantized model saved to {path}")

    return quantized_model

def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

print(tf.__version__)
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make("chess", debug=True)
episodes = 10000
value_network = ValueNetwork().to(device)
value_network.apply(init_weights_kaiming)

value_network_2 = ValueNetwork().to(device)
value_network_2.apply(init_weights_kaiming)

buffer = []
game_history = [0, 0, 0] 

for episode in range(1, episodes+1):
    if episode % 100 == 0:  # 매 100 에피소드마다 고정된 시드로 실행
        env.configuration["seed"] = 12345
    else:
        env.configuration["seed"] = random.randint(0, 10000)
    result = env.run([lambda obs: chess_bot(obs, value_network), lambda obs: chess_bot(obs, value_network_2)])

    print("Agent exit status/reward/time left: ")
    for agent in result[-1]:
        print("\t", agent.status, "/", agent.reward, "/", agent.observation.remainingOverageTime)

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

    print(f"Episode: {episode}")
    print(f"{game_history[0]} wins {game_history[1]} draws {game_history[2]} losses")
    finalize_buffer(value_network, reward)
    finalize_buffer(value_network_2, reward * -1)
    buffer.extend(value_network.buffer) 
    buffer.extend(value_network_2.buffer)
    value_network.buffer.clear()
    value_network_2.buffer.clear()

    if episode > 0 and episode % 10 == 0:  # 매 10 에피소드마다 학습
        print(f"Training on buffer at Episode {episode}")
        train_value_network(value_network, buffer)
        value_network_2.load_state_dict(value_network.state_dict())
        buffer.clear()
    if episode > 0 and episode % 100 == 0:  # 매 100 에피소드마다 모델 저장
        apply_quantization(value_network, path=f"models/value_network_episode_{episode}.pth")
        env.render(mode='ipython', width=600, height=600)