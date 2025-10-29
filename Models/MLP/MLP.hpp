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

void layer_free(Layer* layer) {
  if (!layer) return;
  if (layer->weights) mat_free(layer->weights);
  if (layer->bias) mat_free(layer->bias);
  if (layer->output) mat_free(layer->output);
  free(layer);
}

int layer_forward(Matrix* input, Layer* layer, ActivationType activation_type) {
  if (!input || !layer) return -1;
  
  // Validate dimensions
  // weights is (output_size x input_size), input is (input_size x 1)
  if (layer->weights->cols != input->rows) return -1;
  if (layer->weights->rows != layer->output->rows) return -1;
  if (layer->output->rows != layer->bias->rows) return -1;
  
  float (*activation_func)(float);
  float (*derivative_func)(float);
  get_activation_function(activation_type, &activation_func, &derivative_func);
  
  // create temp matrix (could we do this in-place?)
  Matrix* temp_result = mat_create(layer->output->rows, layer->output->cols);
  if (!temp_result) return -1;
  
  // Wi * Ii -> temp
  int mul_result = mat_mul(layer->weights, input, temp_result);
  if (mul_result != 0) {
    mat_free(temp_result);
    return -1;
  }
  
  // temp -> temp + bias
  int add_result = mat_add(temp_result, layer->bias, layer->output);
  if (add_result != 0) {
    mat_free(temp_result);
    return -1;
  }
  
  mat_free(temp_result);
  
  // activation funciton used
  for (size_t i = 0; i < layer->output->rows; i++) {
    float value = mat_get(layer->output, i, 0);
    value = activation_func(value);
    mat_set_unsafe(layer->output, i, 0, value);
  }
  
  return 0;
}

typedef struct {
  Layer* layers;
  size_t num_layers;
  ActivationType* activations;
  float learning_rate;
} MLP;


MLP* create_mlp(size_t* layer_dims, size_t num_layers, ActivationType* activations, float learning_rate) {
  if (!layer_dims || num_layers < 2 || !activations) return NULL;
  
  MLP* mlp = (MLP*)malloc(sizeof(MLP));
  CHECK_NULL(mlp);
  
  mlp->num_layers = num_layers - 1;
  mlp->learning_rate = learning_rate;
  
  mlp->layers = (Layer*)malloc(sizeof(Layer) * mlp->num_layers);
  if (!mlp->layers) {
    free(mlp);
    return NULL;
  }
  
  mlp->activations = (ActivationType*)malloc(sizeof(ActivationType) * mlp->num_layers);
  if (!mlp->activations) {
    free(mlp->layers);
    free(mlp);
    return NULL;
  }
  
  for (size_t i = 0; i < mlp->num_layers; i++) {
    // Weights: (output_size x input_size) for Wx + b where x is (input_size x 1)
    mlp->layers[i].weights = mat_create(layer_dims[i+1], layer_dims[i]);
    mlp->layers[i].bias = mat_create_with_value(layer_dims[i+1], 1, 0.0f);
    mlp->layers[i].output = mat_create(layer_dims[i+1], 1);
    
    // Check for allocation failures
    if (!mlp->layers[i].weights || !mlp->layers[i].bias || !mlp->layers[i].output) {
      // Free already created matrices
      for (size_t j = 0; j <= i; j++) {
        if (mlp->layers[j].weights) mat_free(mlp->layers[j].weights);
        if (mlp->layers[j].bias) mat_free(mlp->layers[j].bias);
        if (mlp->layers[j].output) mat_free(mlp->layers[j].output);
      }
      free(mlp->layers);
      free(mlp->activations);
      free(mlp);
      return NULL;
    }
    
    // Initialize weights randomly
    float scale = utility::newton_sqrt(2.0f / layer_dims[i]);
    for (size_t row = 0; row < layer_dims[i+1]; row++) {
      for (size_t col = 0; col < layer_dims[i]; col++) {
        float rand_val = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
        mat_set_unsafe(mlp->layers[i].weights, row, col, rand_val);
      }
    }
    
    mlp->activations[i] = activations[i];
  }
  
  return mlp;
}

void mlp_free(MLP* mlp) {
  if (!mlp) return;

  // Free matrices in each layer (layers are not pointers, just structs)
  for (size_t i = 0; i < mlp->num_layers; i++) {
    if (mlp->layers[i].weights) mat_free(mlp->layers[i].weights);
    if (mlp->layers[i].bias) mat_free(mlp->layers[i].bias);
    if (mlp->layers[i].output) mat_free(mlp->layers[i].output);
  }
  
  if (mlp->layers) free(mlp->layers);
  if (mlp->activations) free(mlp->activations);
  
  free(mlp);
}

int mlp_forward(MLP* mlp, Matrix* input, Matrix* output) {
  if (!mlp || !input || !output) return -1;
  
  if (layer_forward(input, &mlp->layers[0], mlp->activations[0]) != 0) {
    return -1;
  }
  
  for (size_t i = 1; i < mlp->num_layers; i++) {
    Matrix* prev_output = mlp->layers[i-1].output;
    if (layer_forward(prev_output, &mlp->layers[i], mlp->activations[i]) != 0) {
      return -1;
    }
  }
  
  Matrix* final_output = mlp->layers[mlp->num_layers - 1].output;
  if (final_output->rows != output->rows || final_output->cols != output->cols) {
    return -1;
  }
  
  for (size_t i = 0; i < output->rows; i++) {
    for (size_t j = 0; j < output->cols; j++) {
      float value = mat_get(final_output, i, j);
      mat_set_unsafe(output, i, j, value);
    }
  }
  
  return 0;
}
