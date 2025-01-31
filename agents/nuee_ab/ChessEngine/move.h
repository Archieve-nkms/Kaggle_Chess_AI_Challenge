#pragma once
#include "board.h"
#include "string"

enum Piece
{
    PAWN_WHITE = 0,
    KNIGHT_WHITE,
    BISHOP_WHITE,
    ROOK_WHITE,
    QUEEN_WHITE,
    KING_WHITE,
    PAWN_BLACK,
    KNIGHT_BLACK,
    BISHOP_BLACK,
    ROOK_BLACK,
    QUEEN_BLACK,
    KING_BLACK
};

typedef struct Move
{
    uint64_t from = 0;
    uint64_t to = 0;
    int piece = PAWN_WHITE;
    int promote_to = QUEEN_WHITE;
    Move(uint64_t f, uint64_t t, int p, int p_t = QUEEN_WHITE)
        :from(f), to(t), piece(p), promote_to(p_t)
    {

    };
}Move;

Board applyMove(Board board, Move move);
std::string convertMoveToUci(Move move);