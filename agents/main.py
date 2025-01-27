from subprocess import Popen, PIPE

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


cpp_executable_path = "main.out"
cpp_bridge = CppBridge(cpp_executable_path)

def chess_bot(obs):

    cpp_bridge.send(obs)
    move = cpp_bridge.receive()

    return move

move = chess_bot("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
print(f"Move: {move}")