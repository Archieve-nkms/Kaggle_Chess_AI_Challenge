#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// 가상의 체스 AI 함수: 현재 상태에서 최적의 움직임 계산
std::string calculate_best_move(const std::string& board_state) {
    // TODO: 체스 AI 알고리즘 추가
    // 예시: 항상 "e2e4"를 반환
    return "e2e4";
}

int main() {
    std::string board_state;

    std::getline(std::cin, board_state);

    std::string best_move = calculate_best_move(board_state);

    std::cout << best_move << std::endl;

    return 0;
}