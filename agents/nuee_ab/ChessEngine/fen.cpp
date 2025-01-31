#include "fen.h"

void convertFenToBoard(Board* board,const char fen[])
{
    int idx = 0;
    char c;

    // board state;
    int file = 0;
    for(int rank = 8; rank >= 1; rank--)
    {
        file = 1;
        while(file <= 8)
        {
            int offset = 0;
            c = fen[idx++];
            if(c == '/')
                continue;
            if('0' <= c && c <= '9')
            {
                file += c - '0';
                continue;
            }
            if(c - 'Z' > 0) // if piece is black
            {
                offset = 6;
                c -= 32; // toUpper
            }

            int piece = 0;
            switch (c)
            {
                case 'P': piece = 0; break;
                case 'N': piece = 1; break;
                case 'B': piece = 2; break;
                case 'R': piece = 3; break;
                case 'Q': piece = 4; break;
                case 'K': piece = 5; break;
            }

            board->bitboard[piece + offset] |= getSquareBitmask(rank, file);
            file++;
        }
    }

    idx++; // skip space

    // Active color
    board->active = fen[idx++] == 'w' ? 0 : 1;

    idx++; // skip space

    // Castling
    uint8_t castling = 0b0000;
    while(true)
    {
        c = fen[idx++];
        if(c == 'K')
            castling |= CASTLING_KS_WHITE;
        else if(c=='Q')
            castling |= CASTLING_QS_WHITE;
        else if(c=='k')
            castling |= CASTLING_KS_BLACK;
        else if(c=='q')
            castling |= CASTLING_QS_BLACK;
        else if(c=='-')
        {
            idx++;
            break;
        }
        else
            break;
    }
    board->castling = castling;

    // en-passant
    if(fen[idx] == '-')
        idx++;
    else
    {
        file = fen[idx++] - 'a' + 1;
        int rank = fen[idx] - '1' + 1;
        board->en_passant = getSquareBitmask(rank, file);
    }
}