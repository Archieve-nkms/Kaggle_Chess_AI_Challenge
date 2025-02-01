#include "move.h"

Board applyMove(Board board, Move move)
{
    uint64_t from_mask = move.from;
    uint64_t to_mask = move.to;

    board.bitboard[move.piece] &= ~from_mask;
    if((move.piece == PAWN_BLACK || move.piece == PAWN_WHITE) && (to_mask & PROMOTION_RANK))
        board.bitboard[move.promote_to] |= to_mask;
    else
        board.bitboard[move.piece] |= to_mask;

    // 캐슬링
    if (move.piece == KING_WHITE || move.piece == KING_BLACK)
    {
        bool is_white = (move.piece == KING_WHITE);

        if (from_mask == getSquareBitmask(is_white ? 1 : 8, 5))
        {
            if (to_mask == getSquareBitmask(is_white ? 1 : 8, 7))
            {
                board.bitboard[is_white ? ROOK_WHITE : ROOK_BLACK] &= ~getSquareBitmask(is_white ? 1 : 8, 8);
                board.bitboard[is_white ? ROOK_WHITE : ROOK_BLACK] |= getSquareBitmask(is_white ? 1 : 8, 6);
            }
            else if (to_mask == getSquareBitmask(is_white ? 1 : 8, 3))
            {
                board.bitboard[is_white ? ROOK_WHITE : ROOK_BLACK] &= ~getSquareBitmask(is_white ? 1 : 8, 1);
                board.bitboard[is_white ? ROOK_WHITE : ROOK_BLACK] |= getSquareBitmask(is_white ? 1 : 8, 4);
            }

            board.castling &= ~(is_white ? (CASTLING_KS_WHITE | CASTLING_QS_WHITE) : (CASTLING_KS_BLACK | CASTLING_QS_BLACK));
        }
    }

    // 잡힌말 제거
    int offset = (board.active == WHITE) ? PAWN_BLACK : 0;
    for (int i = 0; i < 6; i++)
    {
        if ((to_mask & board.bitboard[offset + i]) == to_mask)
        {
            board.bitboard[offset + i] &= ~to_mask;
            break;
        }
    }

    if((move.piece == PAWN_WHITE || move.piece == PAWN_BLACK) && move.to == board.en_passant)
    {
        board.bitboard[offset + PAWN_WHITE] &= ~to_mask;
    }
    
    // 앙파상 갱신
    board.en_passant = 0ULL;
    if(move.piece == PAWN_WHITE && (move.from & RANK_2) == move.from && (move.to & RANK_4) == move.to)
    {
        board.en_passant = move.from << 8;
    }
    else if(move.piece == PAWN_BLACK && (move.from & RANK_7) == move.from && (move.to & RANK_5) == move.to)
    {
        board.en_passant = move.from >> 8;
    }

    // 룩 이동시 캐슬링 권한 제거
    if (move.piece == ROOK_WHITE)
    {
        if (move.from == getSquareBitmask(1, 1)) board.castling &= ~CASTLING_QS_WHITE;
        if (move.from == getSquareBitmask(1, 8)) board.castling &= ~CASTLING_KS_WHITE;
    }
    else if (move.piece == ROOK_BLACK)
    {
        if (move.from == getSquareBitmask(8, 1)) board.castling &= ~CASTLING_QS_BLACK;
        if (move.from == getSquareBitmask(8, 8)) board.castling &= ~CASTLING_KS_BLACK;
    }

    board.active = !board.active;
    return board;
}

std::string convertMoveToUci(Move move)
{
    std::string uci = "";

    int from_index = __builtin_ctzll(move.from);
    char from_file = 'h' - (from_index % 8);
    char from_rank = '1' + (from_index / 8);

    int to_index = __builtin_ctzll(move.to);
    char to_file = 'h' - (to_index % 8);
    char to_rank = '1' + (to_index / 8);

    uci += from_file;
    uci += from_rank;
    uci += to_file;
    uci += to_rank;

    
    if ((move.piece == PAWN_WHITE || move.piece == PAWN_BLACK) && (move.to & PROMOTION_RANK) != 0)
    {
        switch (move.promote_to)
        {
            case KNIGHT_WHITE: case KNIGHT_BLACK: uci += 'n'; break;
            case BISHOP_WHITE: case BISHOP_BLACK: uci += 'b'; break;
            case ROOK_WHITE: case ROOK_BLACK: uci += 'r'; break;
            case QUEEN_WHITE: case QUEEN_BLACK: uci += 'q'; break;
            default: break;
        }
    }

    return uci;
}
