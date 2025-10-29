#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "Utils/Matrix.hpp"
#include "Utils/Activation.hpp"
#include "Utils/Loss.hpp"
#include "Models/MLP/MLP.hpp"

void test_matrix_operations() {
    printf("\n=== Testing Matrix Operations ===\n");
    
    // Create matrices
    Matrix* a = mat_create(2, 3);
    Matrix* b = mat_create(3, 2);
    Matrix* result = mat_create(2, 2);
    
    // Fill matrix a
    mat_set(a, 0, 0, 1.0f);
    mat_set(a, 0, 1, 2.0f);
    mat_set(a, 0, 2, 3.0f);
    mat_set(a, 1, 0, 4.0f);
    mat_set(a, 1, 1, 5.0f);
    mat_set(a, 1, 2, 6.0f);
    
    // Fill matrix b
    mat_set(b, 0, 0, 1.0f);
    mat_set(b, 0, 1, 4.0f);
    mat_set(b, 1, 0, 2.0f);
    mat_set(b, 1, 1, 5.0f);
    mat_set(b, 2, 0, 3.0f);
    mat_set(b, 2, 1, 6.0f);
    
    printf("\nMatrix A:");
    mat_print(a);
    printf("\nMatrix B:");
    mat_print(b);
    
    // Test multiplication
    if (mat_mul(a, b, result) == 0) {
        printf("\nMatrix A * B:");
        mat_print(result);
    }
    
    mat_free(a);
    mat_free(b);
    mat_free(result);
}

void test_activation_functions() {
    printf("\n=== Testing Activation Functions ===\n");
    
    float test_values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    
    printf("\nSigmoid:\n");
    for (int i = 0; i < 5; i++) {
        float result = sigmoid(test_values[i]);
        printf("sigmoid(%.1f) = %.4f\n", test_values[i], result);
    }
    
    printf("\nTanh:\n");
    for (int i = 0; i < 5; i++) {
        float result = tanh_act(test_values[i]);
        printf("tanh(%.1f) = %.4f\n", test_values[i], result);
    }
    
    printf("\nReLU:\n");
    for (int i = 0; i < 5; i++) {
        float result = relu(test_values[i]);
        printf("ReLU(%.1f) = %.4f\n", test_values[i], result);
    }
}

void test_loss_functions() {
    printf("\n=== Testing Loss Functions ===\n");
    
    Matrix* pred = mat_create(2, 1);
    Matrix* target = mat_create(2, 1);
    
    mat_set(pred, 0, 0, 0.7f);
    mat_set(pred, 1, 0, 0.3f);
    
    mat_set(target, 0, 0, 1.0f);
    mat_set(target, 1, 0, 0.0f);
    
    printf("\nPredicted:");
    mat_print(pred);
    printf("\nTarget:");
    mat_print(target);
    
    float mse_loss = mse(pred, target);
    printf("\nMSE Loss: %.4f\n", mse_loss);
    
    mat_free(pred);
    mat_free(target);
}

void test_mlp_forward() {
    printf("\n=== Testing MLP Forward Pass ===\n");
    
    // Create a simple network: 2 inputs → 3 hidden → 2 outputs
    size_t layer_dims[] = {2, 3, 2};
    ActivationType activations[] = {ACTIVATION_RELU, ACTIVATION_SIGMOID};
    
    MLP* mlp = create_mlp(layer_dims, 3, activations, 0.01f);
    
    if (!mlp) {
        printf("Failed to create MLP\n");
        return;
    }
    
    printf("\nNetwork created with architecture: 2 → 3 → 2\n");
    printf("Learning rate: %.3f\n", mlp->learning_rate);
    
    // Create input
    Matrix* input = mat_create(2, 1);
    mat_set(input, 0, 0, 0.5f);
    mat_set(input, 1, 0, 0.3f);
    
    printf("\nInput:");
    mat_print(input);
    
    // Forward pass
    Matrix* output = mat_create(2, 1);
    if (mlp_forward(mlp, input, output) == 0) {
        printf("\nOutput:");
        mat_print(output);
        
        // Print intermediate layer outputs
        printf("\nLayer 0 (ReLU) output:");
        mat_print(mlp->layers[0].output);
        printf("\nLayer 1 (Sigmoid) output:");
        mat_print(mlp->layers[1].output);
    } else {
        printf("Forward pass failed\n");
    }
    
    mat_free(input);
    mat_free(output);
    mlp_free(mlp);
}

void test_xor_problem() {
    printf("\n=== Testing XOR Problem ===\n");
    
    // Create network: 2 inputs → 4 hidden → 1 output
    size_t layer_dims[] = {2, 4, 1};
    ActivationType activations[] = {ACTIVATION_TANH, ACTIVATION_SIGMOID};
    
    MLP* mlp = create_mlp(layer_dims, 3, activations, 0.01f);
    
    printf("\nXOR Network: 2 → 4 → 1\n");
    printf("Activation: Tanh → Sigmoid\n");
    
    // Test XOR inputs
    float xor_inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    
    Matrix* input = mat_create(2, 1);
    Matrix* output = mat_create(1, 1);
    
    printf("\nTesting XOR (random weights, not trained yet):\n");
    for (int i = 0; i < 4; i++) {
        mat_set(input, 0, 0, xor_inputs[i][0]);
        mat_set(input, 1, 0, xor_inputs[i][1]);
        
        if (mlp_forward(mlp, input, output) == 0) {
            float result = mat_get(output, 0, 0);
            printf("XOR(%.0f, %.0f) = %.4f\n", xor_inputs[i][0], xor_inputs[i][1], result);
        }
    }
    
    mat_free(input);
    mat_free(output);
    mlp_free(mlp);
}

int main() {
    printf("=== Neural Network Library Test ===\n");
    
    // Seed random number generator
    srand(time(NULL));
    
    test_matrix_operations();
    test_activation_functions();
    test_loss_functions();
    test_mlp_forward();
    test_xor_problem();
    
    printf("\n=== All Tests Complete ===\n");
    
    return 0;
}

