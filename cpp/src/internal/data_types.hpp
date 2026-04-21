#pragma once
#include <vector>
using std::vector;

int WORD_SIZE = 64;

using VertexId = int;
using AdjList = vector<vector<VertexId>>;
using ColorList = vector<u_char>;
using Clique = vector<VertexId>;
using CliqueList = vector<Clique>;