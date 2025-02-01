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

    def loop(self):
        while(True):
            return self.receive();

cpp_executable_path = "agents/nuee_ab/main.out"
cpp_bridge = CppBridge(cpp_executable_path)

def chess_bot(obs):
    global cpp_bridge
    if(cpp_bridge == None):
        cpp_bridge = CppBridge(cpp_executable_path) 
    
    cpp_bridge.send(f"{obs.board}")
    best_move = cpp_bridge.loop()
    return best_move