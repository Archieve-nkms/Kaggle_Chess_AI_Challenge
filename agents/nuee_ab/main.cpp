#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include "board.h"
#include "fen.h"
#include "move.h"
#include "genMove.h"

using namespace std;

constexpr float FLOAT_MIN = -numeric_limits<float>::max();
constexpr float FLOAT_MAX = numeric_limits<float>::max();
constexpr int DEFAULT_DEPTH = 2;

float evaluate(Board board)
{
    random_device rd; // 시드 값을 위한 random_device
    mt19937 gen(rd()); // Mersenne Twister 엔진

    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    return dis(gen);
}

float minimax(Board board, int depth, bool maximizing, float alpha = FLOAT_MIN, float beta = FLOAT_MAX)
{
    vector<Move> legal_moves = getLegalMoves(board);

    if(depth == 0 || legal_moves.size() == 0)
        return evaluate(board); 

    if(maximizing)
    {
        float max_eval = FLOAT_MIN;
        for(auto& move : legal_moves)
        {
            float eval = minimax(applyMove(board, move), depth - 1, false, alpha, beta);

            max_eval= max(max_eval, eval);
            alpha = max(alpha, eval);
            if(beta <= alpha)
                break;
        }
        return max_eval;
    }
    else
    {
        float min_eval = FLOAT_MAX;
        for(auto& move : legal_moves)
        {
            float eval = minimax(applyMove(board, move), depth - 1, true, alpha, beta);

            min_eval= min(min_eval, eval);
            alpha = min(alpha, eval);
            if(beta <= alpha)
                break;
        }

        return min_eval;
    }
    
}

string getBestMove(Board board)
{
    float best_eval = FLOAT_MIN;
    string best_move;
    vector<Move> legal_moves = getLegalMoves(board);

    for(auto& move : legal_moves)
    {
        float value = minimax(applyMove(board, move), DEFAULT_DEPTH - 1, true);
        if(value > best_eval)
        {
            best_eval = value;
            best_move = convertMoveToUci(move);
        }
    }

    return best_move;
}

int main() 
{
    initMoves();
    while(true)
    {
        string fen;
        getline(cin, fen);

        Board board;
        convertFenToBoard(&board, fen.c_str());

        string best_move = getBestMove(board);
        cout << "output?" << best_move << endl;
    }
    return 0;
}