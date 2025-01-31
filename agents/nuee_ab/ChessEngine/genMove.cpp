#include "genMove.h"
#include "bitMask.h"

uint64_t KNIGHT_MOVES[64];
uint64_t KING_MOVES[64];
uint64_t BISHOP_MOVES[64];

void initKnightMoves() 
{
    int dr[] = {-2, -2, -1, -1, 1, 1, 2, 2};
    int df[] = {-1, 1, -2, 2, -2, 2, -1, 1};

    for (int sq = 0; sq < 64; sq++) 
    {
        uint64_t move = 0ULL;
        int rank = sq / 8, file = sq % 8;

        for (int i = 0; i < 8; i++) 
        {
            int nr = rank + dr[i], nf = file + df[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                move |= (1ULL << (nr * 8 + nf));
        }

        KNIGHT_MOVES[sq] = move;
    }
}

void initBishopMoves()
{
    int dr[] = {1, 1, -1, -1}; 
    int df[] = {1, -1, -1, 1};

    for(int rank = 8; rank >= 1; rank--)
    {
        for(int file = 1; file <= 8; file++)
        {
            uint64_t move = 0ULL;

            for(int i = 0; i < 4; i++)
            {
                int nr = rank;
                int nf = file;
                while(true)
                {
                    nr += dr[i];
                    nf += df[i];
                    if(1 > nr || nr > 8 || 1 > nf || nf > 8)
                        break;
                    move |= (1ULL << ((nr - 1) * 8 + (8 - nf)));
                }
            }
            BISHOP_MOVES[(rank - 1) * 8 + (8 - file)] = move;
        }
    }
}

void initKingMoves()
{
    for (int sq = 0; sq < 64; sq++) 
    {
        uint64_t move = 0ULL;
        int rank = sq / 8, file = sq % 8;
        
        int dr[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int df[] = {-1, 0, 1, -1, 1, -1, 0, 1};

        for (int i = 0; i < 8; i++) 
        {
            int nr = rank + dr[i], nf = file + df[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                move |= (1ULL << (nr * 8 + nf));
        }

        KING_MOVES[sq] = move;
    }
}

void initMoves()
{
    initKnightMoves();
    initBishopMoves();
    initKingMoves();
}

std::vector<Move> getLegalMoves(Board board)
{
    std::vector<Move> legal_moves;

    bool isWhite = board.active == 0;
    int offset = isWhite ? 0 : 6;
    uint64_t us = 0ULL, them = 0ULL, all = 0ULL;

    for(int i = 0; i < 6; i++)
    {
        us |= board.bitboard[offset + i];
        them |= board.bitboard[(offset ^ 6) + i];
    }
    all = us | them;

    uint64_t knight_pieces = board.bitboard[offset + KNIGHT_WHITE];
    while(knight_pieces)
    {
        uint64_t from = 1ULL << __builtin_ctzll(knight_pieces);
        uint64_t moves = KNIGHT_MOVES[from] & ~us;

        while(moves)
        {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + KNIGHT_WHITE)); 
            moves &= moves | to;           
        }

        knight_pieces &= knight_pieces | from;
    }

    uint64_t bishop_pieces = board.bitboard[offset + BISHOP_WHITE];
    while (bishop_pieces) 
    {
        uint64_t from = 1ULL << __builtin_ctzll(bishop_pieces);
        uint64_t moves = BISHOP_MOVES[from] & ~us;

        // TODO: 필터링
        while (moves) {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + BISHOP_WHITE));
            moves &= moves | to;
        }

        bishop_pieces &= bishop_pieces | from;
    }

    uint64_t rook_pieces = board.bitboard[offset + ROOK_WHITE];
    while(rook_pieces)
    {
        int fromSq = __builtin_ctzll(rook_pieces);
        uint64_t from = 1ULL << fromSq;
        uint64_t mask = getRookBitmask(fromSq / 8 + 1, fromSq % 8 + 1);

        // TODO: 이동 찾기, 필터링
    }

    return legal_moves;
}