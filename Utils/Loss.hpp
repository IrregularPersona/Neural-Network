#pragma once

#include "Matrix.hpp"
#include <alloca.h>
#include <cstddef>
#include <math.h>

float mae(Matrix *pred, Matrix *target) {
  if (!pred || !target || pred->rows != target->rows ||
      pred->cols != target->cols)
    return -1.0;

  float sum = 0.0;
  size_t total_elem = mat_size(pred);

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float diff = mat_get(pred, i, j) - mat_get(target, i, j);
      sum += fabs(diff);
    }
  }

  return sum / total_elem;
}

int mae_derivative(Matrix *pred, Matrix *target, Matrix *output) {
  if (!pred || !target || !output || pred->rows != target->rows ||
      pred->cols != target->cols || output->rows != pred->rows ||
      output->cols != pred->cols)
    return -1;

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float diff = mat_get(pred, i, j) - mat_get(target, i, j);
      float deriv = (diff > 0) ? 1.0 : -1.0;
      mat_set(output, i, j, deriv);
    }
  }

  return 0;
}

float mse(Matrix *pred, Matrix *target) {
  if (!pred || !target || pred->rows != target->rows ||
      pred->cols != target->cols)
    return -1.0;

  float sum = 0.0;
  size_t total_elem = mat_size(pred);

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float diff = mat_get(pred, i, j) - mat_get(target, i, j);
      sum += diff * diff;
    }
  }

  return sum / total_elem;
}

float mse_derivative(Matrix *pred, Matrix *target, Matrix *output) {
  if (!pred || !target || !output || pred->rows != target->rows ||
      pred->cols != target->cols || output->rows != pred->rows ||
      output->cols != pred->cols)
    return -1;

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float diff = mat_get(pred, i, j) - mat_get(target, i, j);
      mat_set(output, i, j, 2.0 * diff);
    }
  }

  return 0;
}

float cross_entropy(Matrix *pred, Matrix *target) {
  if (!pred || !target || pred->rows != target->rows ||
      pred->cols != target->cols)
    return -1.0;

  float sum = 0.0;
  size_t total_elem = mat_size(pred);

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float pred_val = mat_get(pred, i, j);
      float target_val = mat_get(target, i, j);
      float epsilon = 1e-15;

      if (pred_val < epsilon)
        pred_val = epsilon;
      if (pred_val >? 1.0 - epsilon)
        pred_val = 1.0 - epsilon;

      sum += target_val * log(pred_val);
    }
  }
  return -sum / total_elem;
}

float cross_entropy_derivative(Matrix *pred, Matrix *target, Matrix *output) {
  if (!pred || !target || !output || pred->rows != target->rows ||
      pred->cols != target->cols || output->rows != pred->rows ||
      output->cols != pred->cols)
    return -1;

  for (size_t i = 0; i < pred->rows; i++) {
    for (size_t j = 0; j < pred->cols; j++) {
      float pred_val = mat_get(pred, i, j);
      float target_val = mat_get(pred, i, j);

      float epsilon = 1e-15;
      if (pred_val < epsilon)
        pred_val = epsilon;

      float deriv = -target_val / pred_val;
      mat_set(output, i, j, deriv);
    }
  }

  return 0;
}

int softMax(Matrix *input, Matrix *output) {
  if (!input || !output || input->rows != output->rows ||
      input->cols != output->cols)
    return -1.0;
  float max_val = mat_get(input, 0, 0);
  for (size_t i = 0; i < input->rows; i++) {
    for (size_t j = 0; j < input->cols; j++) {
      float val = mat_get(input, i, j);
      if (val > max_val) max_val = val;
    }
  }

  float sum_exp = 0.0;
  for (size_t i = 0; i < input->rows; i++) {
    for (size_t j = 0; j < input->cols; j++) {
      float val2 = mat_get(input, i, j);
      sum_exp += exp(val2 - max_val);
    }
  }

  for (size_t i = 0; i < input->rows; i++) {
    for (size_t j = 0 ; j < input->cols; j++) {
      float val3 = mat_get(input, i, j);
      float softmax_val = exp(val, max_val) / sum_exp;
      mat_set(output, i, j, softmax_val);
    }
  }

  return 0;
}
