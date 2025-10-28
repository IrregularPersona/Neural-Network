#pragma once

#include "../../Utils/Matrix.hpp"
#include "../../Utils/Activation.hpp"
#include "../../Utils/Utils.hpp"
#include <cstdlib>

#define CHECK_NULL(X) do { if(!(X)) return NULL; } while(0)

typedef struct {
  Matrix *weights;
  Matrix *bias;
  Matrix *output;
} Layer;

Layer* create_layer(size_t input_size, size_t output_size, ActivationType activations) {
  Layer* layer = (Layer*)malloc(sizeof(Layer));
  CHECK_NULL(layer);

  layer->weights = mat_create(input_size, output_size);
  if (!layer->weights) {
    free(layer);
    return NULL;
  }
  
  // scale (prolly doesnt need insane precision for now)
  // TODO: might change later
  float scale = utility::newton_sqrt(2.0f / input_size);
  for (size_t row = 0; row < input_size; row++) {
    for (size_t col = 0; col < output_size; col++) {
      float rand_val = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
      mat_set_unsafe(layer->weights, row, col, rand_val);
    }
  }

  layer->bias = mat_create_with_value(output_size, 1, 0.0f);
  if (!layer->bias) {
    mat_free(layer->weights);
    free(layer);
    return NULL;
  }

  layer->output = mat_create(output_size, 1);
  if (!layer->output) {
    mat_free(layer->weights);
    mat_free(layer->bias);
    free(layer);
    return NULL;
  }

  return layer;
}

typedef struct {
  Layer* layers;
  size_t num_layers;
  ActivationType* activations;
  float learning_rate;
} MLP;


MLP* create_mlp(Layer* layers, size_t num_layers, ActivationType* activations, double learning_rate) {
  
}
  



void layer_forward(Matrix *weight, Matrix *bias, Matrix *output) {
  
}
