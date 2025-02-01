#pragma once
#include <cstdint>

enum Color
{
    WHITE = 0,
    BLACK
};

typedef struct Board
{
    uint8_t active = WHITE; // 0: white 1: black
    uint8_t castling = 0b0000; // 0b1111; KQkq 
    uint64_t en_passant = 0; // A capturable pawn is marked as 1
    uint64_t bitboard[12] {0}; // 0-5: white 6-11: black, [P, N, B, R, Q, K, p, n, b, r, q, k]
}Board;