#include "utils.c"
#include <string.h>


typedef struct Perceptron {
  Vector* weights;
  double bias;
  double learning_rate;
}Perceptron;

Perceptron* set_default_value(Vector* values, double bias_value, double lr_value) {
  Perceptron* p = malloc(sizeof(Perceptron));

  if (p == NULL) {
    perror("Can't allocate memory for the perceptron");
    return NULL;
  }
  
  p->weights = values;
  p->bias = bias_value;
  p->learning_rate = lr_value;

  return p;
}

void init_vector(Vector* p, double inputs[], size_t inputs_len)  {
  for (size_t i = 0; i < inputs_len; i++) {
    vec_append(p, &inputs[i]);
  }
}

int predict(Perceptron* p, Vector* input, int* out_result, double* sum_result) {
  if (p == NULL || input == NULL || out_result == NULL) {
    return -1; // Null input
  }

  if (vec_len(p->weights) != vec_len(input)) {
    return -2; // Length mismatch
  }

  double sum = 0.0;

  for (size_t i = 0; i < vec_len(input); i++) {
    double* x_i = VEC_GET(double, input, i);
    double* w_i = VEC_GET(double, p->weights, i);

    if (x_i == NULL || w_i == NULL) {
      return -3; // Null vector elements
    }

    sum += (*x_i) * (*w_i);
  }

  sum += p->bias;

  *out_result = sum > 0 ? 1 : 0;
  *sum_result = sum;
  return 0; // Success
}


int main() {
  // default for perceptron
  Vector* weight = vec_new(0, sizeof(double));
  double weights[] = {0.5, 0.1, -0.3};
  size_t weights_len = sizeof(weights) / sizeof(weights[0]);
  init_vector(weight, weights, weights_len);
  double bias = 0.5;
  double lr = 0.01;

  // init the input vector
  Vector* input = vec_new(0, sizeof(double));
  double inputs[] = {0.8, -0.2, 0.5};
  size_t inputs_len = sizeof(inputs) / sizeof(inputs[0]);
  init_vector(input, inputs, inputs_len);
  
  // creating the perceptron
  Perceptron* p = set_default_value(weight, bias, lr);

  int out_result;
  double sum_result;
  int status = predict(p, input, &out_result, &sum_result);

  switch (status) { 
    case 0: printf("Prediction: %d\nSum result: %lf", out_result, sum_result); break;
    default: printf("Failed with error code: %d\n", status); break;
  }
  
  // free the memory pointers
  vec_free(input);
  vec_free(weight);
  free(p);
  return 0;
}
