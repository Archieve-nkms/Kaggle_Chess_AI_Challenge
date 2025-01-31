#include "move.h"
#include "bitMask.h"

Board applyMove(Board board, Move move)
{
    uint64_t from_mask = move.from;
    uint64_t to_mask = move.to;

    board.bitboard[move.piece] &= ~from_mask;
    if(to_mask & PROMOTION_RANK)
        board.bitboard[move.promote_to] |= to_mask;
    else
        board.bitboard[move.piece] |= to_mask;

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

    int offset = (move.piece < PAWN_BLACK) ? PAWN_BLACK : 0;
    for (int i = 0; i < 6; i++)
    {
        if (board.bitboard[offset + i] & to_mask)
        {
            board.bitboard[offset + i] &= ~to_mask;
            break;
        }
    }

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

    return board;
}

std::string convertMoveToUci(Move move)
{
    std::string uci = "";

    int from_index = __builtin_ctzll(move.from);
    char from_file = 'a' + (from_index % 8);
    char from_rank = '1' + (from_index / 8);

    int to_index = __builtin_ctzll(move.to);
    char to_file = 'a' + (to_index % 8);
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
