#include "genMove.h"

uint64_t KNIGHT_MOVES[64];
uint64_t BISHOP_MOVES[64];
uint64_t KING_MOVES[64];

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
    initKingMoves();
}

uint64_t getBishopMoves(uint64_t from, uint64_t occupied , uint64_t us)
{
    uint64_t moves = 0ULL;
    int sq = __builtin_ctzll(from);
    int rank = sq / 8 + 1;
    int file = 8 - (sq % 8);
    int dr[] = {1, 1, -1, -1};
    int df[] = {1, -1, -1, 1};

    for (int i = 0; i < 4; i++) 
    {
        int nr = rank;
        int nf = file;

        while (true) 
        {
            nr += dr[i];
            nf += df[i];

            if (nr < 1 || nr > 8 || nf < 1 || nf > 8) 
                break;

            uint64_t toMask = 1ULL << ((nr - 1) * 8 + (8 - nf));

            moves |= toMask;

            if (occupied & toMask) 
            {
                if (us & toMask) 
                {
                    moves &= ~toMask;
                }
                break;
            }
        }
    }

    return moves;
}

uint64_t getRookMoves(uint64_t from, uint64_t occupied , uint64_t us) 
{
    uint64_t moves = 0ULL;
    int sq = __builtin_ctzll(from);
    int rank = sq / 8 + 1;
    int file = 8 - (sq % 8);

    // 수직 & 수평 방향 이동 (↑ ↓ ← →)
    int dr[] = {1, -1, 0, 0};
    int df[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; i++) 
    {
        int nr = rank;
        int nf = file;

        while (true) 
        {
            nr += dr[i];
            nf += df[i];

            if (nr < 1 || nr > 8 || nf < 1 || nf > 8) 
                break;

            uint64_t toMask = 1ULL << ((nr - 1) * 8 + (8 - nf));

            moves |= toMask;

            if (occupied & toMask) 
            {
                if (us & toMask) 
                {
                    moves &= ~toMask;
                }
                break;
            }
        }
    }

    return moves;
}

uint64_t getEnemyAttacks(const Board &board) 
{
    uint64_t attacks = 0ULL;
    int offset = board.active == WHITE ? PAWN_BLACK : 0; // 상대편의 기물

    uint64_t us = 0ULL, them = 0ULL, all = 0ULL;
    int n = board.active == WHITE ? 0 : PAWN_BLACK;
    for(int i = 0; i < 6; i++)
    {
        us |= board.bitboard[n + i];
        them |= board.bitboard[(n ^ 6) + i];
    }
    all = us | them;

    for (int i = 0; i < 6; i++) 
    {
        uint64_t pieces = board.bitboard[offset + i];

        while (pieces) 
        {
            int fromSq = __builtin_ctzll(pieces);
            uint64_t from = 1ULL << fromSq;
            uint64_t moves = 0ULL;

            switch (i) 
            {
                case PAWN_WHITE:
                    {
                        if (board.active == BLACK) {
                            moves |= (from & ~FILE_A) << 9;  // 좌측 대각 공격 (파일 A 제외)
                            moves |= (from & ~FILE_H) << 7;  // 우측 대각 공격 (파일 H 제외)
                        } else {
                            moves |= (from & ~FILE_H) >> 9;  // 좌측 대각 공격 (파일 H 제외)
                            moves |= (from & ~FILE_A) >> 7;  // 우측 대각 공격 (파일 A 제외)
                        }
                    }
                    break;
                case KNIGHT_WHITE:
                    moves = KNIGHT_MOVES[fromSq];
                    break;
                case BISHOP_WHITE:
                    moves = getBishopMoves(from, all, them);
                    break;
                case ROOK_WHITE:
                    moves = getRookMoves(from, all, them);
                    break;
                case QUEEN_WHITE:
                    moves = getBishopMoves(from, all, them) |
                            getRookMoves(from, all, them);
                    break;
                case KING_WHITE:
                    moves = KING_MOVES[fromSq];
                    break;
            }

            attacks |= moves;
            pieces &= ~from;
        }
    }

    return attacks;
}

bool isKingInCheck(const Board &board) 
{
    uint64_t king = board.bitboard[board.active == WHITE ? KING_WHITE : KING_BLACK];
    if (!king)
    {
        return true;
    }

    uint64_t enemyAttacks = getEnemyAttacks(board);
    return (king & enemyAttacks) == king;
}

// 추후 시간 남으면 최적화
std::vector<Move> getLegalMoves(Board board)
{
    std::vector<Move> legal_moves;

    bool is_white = board.active == WHITE;
    int offset = is_white ? 0 : 6;
    uint64_t us = 0ULL, them = 0ULL, all = 0ULL;

    for(int i = 0; i < 6; i++)
    {
        us |= board.bitboard[offset + i];
        them |= board.bitboard[(offset ^ 6) + i];
    }
    all = us | them;
    
    uint64_t pawn_pieces = board.bitboard[offset + PAWN_WHITE];
    while(pawn_pieces)
    {
        uint64_t from = 1ULL << __builtin_ctzll(pawn_pieces);
        uint64_t to = is_white ? from << 8 : from >> 8;
        uint64_t promotable_rank = is_white ? RANK_7 : RANK_2;
        uint64_t double_step = is_white ? RANK_2 : RANK_7;

        if((to & all) != to)
        {
            if((from & promotable_rank) == from)
            {
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE, offset + KNIGHT_WHITE));
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE, offset + BISHOP_WHITE));
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE, offset + QUEEN_WHITE));
            }
            else if((from & double_step) == from)
            {
                to = is_white ? from << 16 : from >> 16;
                if((to & all) != to)
                    legal_moves.push_back(Move(from, to, offset + PAWN_WHITE));
            }
            else
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE));
        }

        // Capture & en passant
        if((from & FILE_A) != from)
        {
            to = is_white ? from << 9 : from >> 7;
            if((to & them) == to || to == board.en_passant)
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE));

        }
        if((from & FILE_H) != from)
        {
            to = is_white ? from << 7 : from >> 9;
            if((to & them) == to  || to == board.en_passant)
                legal_moves.push_back(Move(from, to, offset + PAWN_WHITE));
        }

        pawn_pieces &= ~from;
    }

    uint64_t knight_pieces = board.bitboard[offset + KNIGHT_WHITE];
    while(knight_pieces)
    {
        uint64_t sq = __builtin_ctzll(knight_pieces);
        uint64_t from = 1ULL << sq;
        uint64_t moves = KNIGHT_MOVES[sq] & ~us;

        while(moves)
        {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + KNIGHT_WHITE)); 
            moves &= ~to;           
        }

        knight_pieces &= ~from;
    }

    uint64_t bishop_pieces = board.bitboard[offset + BISHOP_WHITE];
    while (bishop_pieces) 
    {
        uint64_t from = 1ULL << __builtin_ctzll(bishop_pieces);
        uint64_t moves = getBishopMoves(from, all, us);

        while (moves)
        {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + BISHOP_WHITE));
            moves &= ~to;
        }

        bishop_pieces &= ~from;
    }

    uint64_t rook_pieces = board.bitboard[offset + ROOK_WHITE];
    while(rook_pieces)
    {
        int fromSq = __builtin_ctzll(rook_pieces);
        uint64_t from = 1ULL << fromSq;
        uint64_t moves = getRookMoves(from, all, us);
        while(moves)
        {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + ROOK_WHITE));
            moves &= ~to;
        }

        rook_pieces &= ~from;
    }

    uint64_t queen_pieces = board.bitboard[offset + QUEEN_WHITE];
    while(queen_pieces)
    {
        int fromSq = __builtin_ctzll(queen_pieces);
        uint64_t from = 1ULL << fromSq;
        uint64_t moves = getRookMoves(from, all, us) | getBishopMoves(from, all, us);
        while(moves)
        {
            uint64_t to = 1ULL << __builtin_ctzll(moves);
            legal_moves.push_back(Move(from, to, offset + QUEEN_WHITE));
            moves &= ~to;
        }

        queen_pieces &= ~from;
    }

    uint64_t king_piece = board.bitboard[offset + KING_WHITE];
    uint64_t sq = __builtin_ctzll(king_piece);
    uint64_t from = 1ULL << sq;
    uint64_t moves = KING_MOVES[sq] & ~us;
    while(moves)
    {
        uint64_t to = 1ULL << __builtin_ctzll(moves);
        legal_moves.push_back(Move(from, to, offset + KING_WHITE)); 
        moves &= ~to;           
    }

    // castling
    // 캐슬링 확인
    uint64_t enemyAttacks = getEnemyAttacks(board);

    // 킹 사이드 캐슬링 (0-0)
    if (board.castling & (is_white ? CASTLING_KS_WHITE : CASTLING_KS_BLACK)) {
        uint64_t path = is_white ? 0x60ULL : 0x6000000000000000ULL;  // 킹 사이드 이동 경로
        uint64_t kingEnd = is_white ? 0x40ULL : 0x4000000000000000ULL;  // 최종 킹 위치

        if (!(all & path) && !(enemyAttacks & path)) {
            uint64_t to = is_white ? (from << 2) : (from >> 2);
            legal_moves.push_back(Move(from, to, offset + KING_WHITE));
        }
    }

    // 퀸 사이드 캐슬링 (0-0-0)
    if (board.castling & (is_white ? CASTLING_QS_WHITE : CASTLING_QS_BLACK)) {
        uint64_t path = is_white ? 0xEULL : 0x7000000000000000ULL;  // 퀸 사이드 이동 경로
        uint64_t kingEnd = is_white ? 0x4ULL : 0x2000000000000000ULL;  // 최종 킹 위치

        if (!(all & path) && !(enemyAttacks & path)) 
        {
            uint64_t to = is_white ? (from >> 2) : (from << 2);
            legal_moves.push_back(Move(from, to, offset + KING_WHITE));
        }
    }

    std::vector<Move> filtered_moves;
    for (const Move &move : legal_moves) 
    {
        Board newBoard = applyMove(board, move);
        newBoard.active = !newBoard.active;
        if (!isKingInCheck(newBoard)) 
        {
            filtered_moves.push_back(move);
        }
    }

    return filtered_moves;
}