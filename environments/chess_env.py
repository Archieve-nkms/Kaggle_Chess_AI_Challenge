from Chessnut import Game


class ChessEnv:
    def __init__(self):
        self.game = Game()
        self.done = False

    def reset(self):
        """게임을 초기화하고 초기 상태 반환"""
        self.game = Game()
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        action: UCI 포맷의 움직임 (예: "e2e4")
        반환: (state, reward, done, info)
        """
        try:
            self.game.apply_move(action)
        except Exception as e:
            # 잘못된 움직임 처리
            self.done = True
            return self._get_state(), -1, self.done, {"error": str(e)}

        # 보상 계산
        reward = self._get_reward()

        # 게임 종료 여부 확인
        self.done = self.game.status in [Game.CHECKMATE, Game.STALEMATE]

        return self._get_state(), reward, self.done, {}

    def _get_reward(self):
        if self.game.status == Game.CHECKMATE:
            return 1  # 승리
        elif self.game.status == Game.STALEMATE:
            return 0  # 무승부
        elif self.game.status == Game.CHECK:
            return 0.5  # 체크
        return 0  # 일반
    
    def _get_state(self):
        """현재 체스판의 상태 반환 (FEN 문자열 사용)"""
        return self.game.get_fen()

    def legal_actions(self):
        """현재 가능한 움직임 리스트 반환"""
        return list(self.game.get_moves())