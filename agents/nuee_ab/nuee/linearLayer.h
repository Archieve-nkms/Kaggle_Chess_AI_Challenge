#pragma once
#include <cstdint>

class LinearLayer
{
public:
    LinearLayer(int num_inputs, int num_outputs);
    void forward();
private:
    int8_t* weights;
    int8_t* biases;
    int num_inputs;
    int num_outputs;
};