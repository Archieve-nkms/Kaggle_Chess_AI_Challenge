#pragma once
#include <vector>
#include <cstdint>
#include "board.h"
#include "move.h"
#include "bitMask.h"

void initMoves();
std::vector<Move> getLegalMoves(Board board); 