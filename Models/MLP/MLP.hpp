#pragma once

#include "../../Utils/Matrix.hpp"
#include "../../Utils/Activation.hpp"
#include "../../Utils/Loss.hpp"
#include "../../Utils/Utils.hpp"
#include <cstddef>
#include <cstdlib>

#define CHECK_NULL(X) do { if(!(X)) return NULL; } while(0)

typedef struct {
  Matrix *weights;
  Matrix *bias;
  Matrix *output;
  Matrix *input;
  Matrix *weight_grad;
  Matrix *bias_grad;
  Matrix *output_grad;
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

int layer_backward(Matrix* output_grad, Layer* layer, Matrix* input_grad, ActivationType activation_type) {
  if (!output_grad || !layer || !input_grad) return -1;

  float (*activation_deriv)(float);
  float (*activation_func)(float); // TODO: Either I need to comment this out or refactor this, since it won't be used in backwards realistically
  get_activation_function(activation_type, &activation_func, &activation_deriv);
  
  Matrix* activation_grad = mat_create(output_grad->rows, output_grad->cols);
  for (size_t i = 0; i < layer->output->rows; i++) {
    float z = mat_get(layer->output, i, 0);
    float deriv = activation_deriv(z);
    float grad = mat_get(output_grad, i, 0) * deriv;
    mat_set_unsafe(activation_grad, i, 0, grad);
  }

  // bias grad compute
  for (size_t i = 0; i < layer->bias->rows; i++) {
    mat_set_unsafe(layer->bias_grad, i, 0, mat_get(activation_grad, i, 0));
  }

  // compute input grad for next layer
  for (size_t i = 0; i < layer->weights->rows; i++) {
    for (size_t j = 0; j < layer->weights->cols; j++) {
      float grad = mat_get(activation_grad, i, 0) * mat_get(layer->input, j, 0);
      mat_set_unsafe(layer->weight_grad, i, j, grad);
    }
  }

  Matrix* weight_transpose = mat_create(layer->weights->cols, layer->weights->rows);
  mat_transpose(layer->weights, weight_transpose);
  mat_mul(weight_transpose, activation_grad, input_grad);

  mat_free(activation_grad);
  mat_free(weight_transpose);

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
    // weights: (output_size x input_size) for Wx + b where x is (input_size x 1)
    mlp->layers[i].weights = mat_create(layer_dims[i+1], layer_dims[i]);
    mlp->layers[i].bias = mat_create_with_value(layer_dims[i+1], 1, 0.0f);
    mlp->layers[i].output = mat_create(layer_dims[i+1], 1);
    // initialize gradient-related pointers (will be allocated later)
    // this is required. thats it.
    mlp->layers[i].input = NULL;
    mlp->layers[i].weight_grad = NULL;
    mlp->layers[i].bias_grad = NULL; 
    mlp->layers[i].output_grad = NULL; 
    
    if (!mlp->layers[i].weights || !mlp->layers[i].bias || !mlp->layers[i].output) {
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
    if (mlp->layers[i].input) mat_free(mlp->layers[i].input);
    if (mlp->layers[i].weight_grad) mat_free(mlp->layers[i].weight_grad);
    if (mlp->layers[i].bias_grad) mat_free(mlp->layers[i].bias_grad);
    if (mlp->layers[i].output_grad) mat_free(mlp->layers[i].output_grad);
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

int mlp_update_weights(MLP* mlp) {
  if (!mlp) return -1;

  for (size_t i = 0; i < mlp->num_layers; i++) {
    // get current layer
    Layer* layer = &mlp->layers[i];

    // get the new weights
    for (size_t row = 0; row < layer->weights->rows; row++) {
      for (size_t col = 0; col < layer->weights->cols; col++) {
        float grad = mat_get(layer->weight_grad, row, col);
        float curr_weight = mat_get(layer->weights, row, col);
        float new_weight = curr_weight - mlp->learning_rate * grad;
        mat_set_unsafe(layer->weights, row, col, new_weight);
      }
    }

    
    // get the new biases
    for (size_t row = 0; row < layer->bias->rows; row++) {
      float grad = mat_get(layer->bias_grad, row, 0);
      float curr_bias = mat_get(layer->bias, row, 0);
      float new_bias = curr_bias - mlp->learning_rate * grad;
      mat_set_unsafe(layer->bias, row, 0, new_bias);
    }
  }

  return 0;
}

float mlp_train(MLP* mlp, Matrix** inputs, Matrix** targets, size_t num_samples, size_t epochs, LossFunction loss_func) {
  if (!mlp || !inputs || !targets) return -1.0f;

  for (size_t i = 0; i < mlp->num_layers; i++) {
    if (!mlp->layers[i].input) {
      size_t input_size = (i == 0) ? mlp->layers[i].weights->cols : mlp->layers[i - 1].output->rows;
      mlp->layers[i].input = mat_create(input_size, 1);
      mlp->layers[i].weight_grad = mat_create(mlp->layers[i].weights->rows, mlp->layers[i].weights->cols);
      mlp->layers[i].bias_grad = mat_create(mlp->layers[i].bias->rows, 1);
      mlp->layers[i].output_grad = mat_create(mlp->layers[i].output->rows, 1);
    }
  }

  float avg_loss = 0.0;

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;

    for (size_t sample = 0; sample < num_samples; sample++) {
      Matrix* input = inputs[sample];
      Matrix* target = targets[sample];

      // store input for first layer 
      // TODO: do NOT index layers by input row
      for (size_t r = 0; r < input->rows; r++) {
        mat_set_unsafe(mlp->layers[0].input, r, 0, mat_get(input, r, 0));
      }

      Matrix* output = mat_create(mlp->layers[mlp->num_layers - 1].output->rows, 1);
      mlp_forward(mlp, input, output);

      for (size_t li = 1; li < mlp->num_layers; li++) {
        Matrix* prev_out = mlp->layers[li - 1].output;
        Matrix* this_in = mlp->layers[li].input;
        for (size_t r = 0; r < this_in->rows; r++) {
          mat_set_unsafe(this_in, r, 0, mat_get(prev_out, r, 0));
        }
      }

      float loss = compute_loss(loss_func, output, target);
      epoch_loss += loss;

      Matrix* output_grad = mat_create(output->rows, output->cols);
      compute_loss_derivative(loss_func, output, target, output_grad);

      Matrix* curr_grad = output_grad;
      for (int i = mlp->num_layers - 1; i >= 0; i--) {
        Matrix* input_grad = mat_create(mlp->layers[i].input->rows, 1);
        layer_backward(curr_grad, &mlp->layers[i], input_grad, mlp->activations[i]);

        if (i > 0) {
          if (curr_grad != output_grad) {
            mat_free(curr_grad);
          }
          curr_grad = input_grad;
        } else {
          mat_free(input_grad);
        }
      }
      if (curr_grad != output_grad) mat_free(curr_grad);
      mat_free(output);
      mat_free(output_grad);

      mlp_update_weights(mlp);
    }

    avg_loss = epoch_loss / num_samples;
    if (epoch % 10 == 0 || epoch == epochs - 1)
      printf("Epoch %zu/%zu = Loss: %.4f\n", epoch + 1, epochs, avg_loss);
  }
  return avg_loss;
}
