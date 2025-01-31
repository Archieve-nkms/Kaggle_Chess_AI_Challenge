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

    def loop(self):
        while(True):
            received_request = self.receive()
            parameters = received_request.split("?")
            cmd:str = parameters[0]
            if cmd == "output":
                return parameters[1]

cpp_executable_path = "agents/nuee_ab/main.exe"
cpp_bridge = CppBridge(cpp_executable_path)

def chess_bot(obs):
    global cpp_bridge
    if(cpp_bridge == None):
        cpp_bridge = CppBridge(cpp_executable_path) 
    
    cpp_bridge.send(f"{obs}")

    best_move = cpp_bridge.loop()
    return best_move