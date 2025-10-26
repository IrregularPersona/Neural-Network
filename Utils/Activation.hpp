#pragma once

#include <cmath>
#include <math.h>

typedef enum {
  ACTIVATION_SIGMOID,
  ACTIVATION_TANH,
  ACTIVATION_RELU,
}ActivationType;

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_derivative(float x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

float tanh_act(float x) {
  return tanh(x);
}

float tanh_derivative(float x) {
  double t = tanh_act(x);
  return 1.0 - (t * t);
}

float relu(float x) {
  return x > 0.0 ? x : 0.0;
}

float relu_derivative(float x) {
  return x > 0 ? 1.0 : 0.0;
}

void get_activation_function(ActivationType type, float (**activation_func)(float), float (**derivative_func)(float)) {
  switch (type) {
    case ACTIVATION_SIGMOID:
      *activation_func = sigmoid;
      *derivative_func = sigmoid_derivative;
      break;
    case ACTIVATION_TANH:
      *activation_func = tanh_act;
      *derivative_func = tanh_derivative;
      break;
    case ACTIVATION_RELU:
      *activation_func = relu;
      *derivative_func = relu_derivative;
      break;
    default:
      *activation_func = sigmoid;
      *derivative_func = sigmoid_derivative;
      break;
  }
}
