#pragma once
#include <vector>
#include <cstdint>
#include "board.h"
#include "move.h"

void initMoves();
std::vector<Move> getLegalMoves(Board board); 