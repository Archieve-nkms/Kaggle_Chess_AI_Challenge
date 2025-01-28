#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <random>

using namespace std;

/*
py와 통신

send: cout << << endl;
receive: getline(cin, var);
*/

// TODO:
// 1. Alpha-beta pruning
// 2. NNUE

constexpr float FLOAT_MIN = -numeric_limits<float>::max();
constexpr float FLOAT_MAX = numeric_limits<float>::max();
constexpr int DEFAULT_DEPTH = 2;

vector<string> split(string str, char delimiter = '?')
{
    vector<string> result;
    stringstream ss(str);
    string t;

    while (getline(ss, t, delimiter))
    {
        result.push_back(t);
    }
    return result;
}

pair<string, vector<string>> request(string command, string fen = "", string move = "")
{
    string str;

    if(command == "apply_move")
    {
        cout << "apply_move" << "?" << fen << "?" << move << endl;
        getline(cin, str);
        vector<string> received = split(str);
        return { received[1], split(received[2], ' ') };
    }

    return {};
}

float evaluate(string fen)
{
    random_device rd; // 시드 값을 위한 random_device
    mt19937 gen(rd()); // Mersenne Twister 엔진

    uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    return dis(gen);
}

float minimax(string fen, vector<string> possible_moves, int depth, bool maximizing, float alpha = FLOAT_MIN, float beta = FLOAT_MAX)
{
    if(depth == 0 || possible_moves.size() == 0)
        return evaluate(fen); 

    if(maximizing)
    {
        float maxEval = FLOAT_MIN;
        for (const auto& move : possible_moves)
        {
            auto received = request("apply_move", fen, move);
            float eval = minimax(received.first, received.second, depth - 1, false, alpha, beta);
            maxEval= max(maxEval, eval);
            alpha = max(alpha, eval);
            if(beta <= alpha)
                break;
        }

        return maxEval;
    }
    else
    {
        float minEval = FLOAT_MAX;
        for (const auto& move : possible_moves)
        {
            auto received = request("apply_move", fen, move);
            float eval = minimax(received.first, received.second, depth - 1, false, alpha, beta);
            minEval = min(minEval, eval);
            beta = min(beta, eval);
            if(beta <= alpha)
                break;
        }

        return minEval;
    }
    
}

string getBestMove(string fen, vector<string> possible_moves)
{
    float bestEval = FLOAT_MIN;
    string best_move = "";
    for(const auto& move : possible_moves)
    {
        auto received = request("apply_move", fen, move);
        float value = minimax(received.first, received.second, DEFAULT_DEPTH - 1, true);
        
        if(value > bestEval)
        {
            bestEval = value;
            best_move = move;
        }
    }

    return best_move;
}

int main() 
{
    while(1)
    {
        string str;
        getline(cin, str);
        vector<string> state = split(str); // fen?moves
        string best_move = getBestMove(state[0], split(state[1], ' '));
        cout << "output?" << best_move << endl;
    }
    return 0;
}