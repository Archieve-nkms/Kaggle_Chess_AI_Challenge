from subprocess import Popen, PIPE
from Chessnut import Game

class CppBridge:
    def __init__(self, path:str):
        self.process = Popen([path], stdin=PIPE, stdout=PIPE, stderr=PIPE)

    def send(self, command:str):
        command += "\n"
        self.process.stdin.write(command.encode())
        self.process.stdin.flush()

    def receive(self):
        return self.process.stdout.readline().decode().strip()
    
    def terminate(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()
        self.process.terminate()

    # cpp에서 체스 엔진을 따로 만드는건 보류
    # 일단 파이썬과 통신해서 필요한 정보를 가져오는 방식으로 구현
    def get_best_move(self, obs):
        game = Game(obs.board)
        moves = " ".join(game.get_moves())
        self.send(f"{obs.board}?{moves}")

        while(True):
            received_request = self.receive()
            parameters = received_request.split("?")
            cmd:str = parameters[0]

            if cmd == "apply_move":
                fen = parameters[1]
                move = parameters[2]
                game = Game(fen)
                game.apply_move(move)
                moves = " ".join(game.get_moves())
                self.send(f"apply_move?{game.get_fen()}?{moves}") # output) apply_move?fen?moves
            elif cmd == "output":
                return parameters[1]

cpp_executable_path = "agents/nuee_ab/main.out"
cpp_bridge = CppBridge(cpp_executable_path)

# def chess_bot
# 1. cpp에게 board 상태들을 보내줌
# 2. cpp에게서 best_move를 받음

def chess_bot(obs):
    global cpp_bridge
    if(cpp_bridge == None):
        cpp_bridge = CppBridge(cpp_executable_path) 

    best_move = cpp_bridge.get_best_move(obs);
    return best_move