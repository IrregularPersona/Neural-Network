#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "Utils/Matrix.hpp"
#include "Utils/Activation.hpp"
#include "Utils/Loss.hpp"
#include "Models/MLP/MLP.hpp"

void test_training() {
    printf("\n=== Test: Training XOR Problem ===\n");
    
    size_t layer_dims[] = {2, 4, 1};
    ActivationType activations[] = {ACTIVATION_TANH, ACTIVATION_SIGMOID};
    MLP* network = create_mlp(layer_dims, 3, activations, 0.1f);
    
    Matrix* input1 = mat_create(2, 1); mat_set(input1, 0, 0, 0.0f); mat_set(input1, 1, 0, 0.0f);
    Matrix* target1 = mat_create(1, 1); mat_set(target1, 0, 0, 0.0f);
    
    Matrix* input2 = mat_create(2, 1); mat_set(input2, 0, 0, 0.0f); mat_set(input2, 1, 0, 1.0f);
    Matrix* target2 = mat_create(1, 1); mat_set(target2, 0, 0, 1.0f);
    
    Matrix* input3 = mat_create(2, 1); mat_set(input3, 0, 0, 1.0f); mat_set(input3, 1, 0, 0.0f);
    Matrix* target3 = mat_create(1, 1); mat_set(target3, 0, 0, 1.0f);
    
    Matrix* input4 = mat_create(2, 1); mat_set(input4, 0, 0, 1.0f); mat_set(input4, 1, 0, 1.0f);
    Matrix* target4 = mat_create(1, 1); mat_set(target4, 0, 0, 0.0f);
    
    Matrix* inputs[] = {input1, input2, input3, input4};
    Matrix* targets[] = {target1, target2, target3, target4};
    
    mlp_train(network, inputs, targets, 4, 2000, LOSS_MSE);
   
    printf("\nTesting after training:\n");
    for (int i = 0; i < 4; i++) {
        Matrix* output = mat_create(1, 1);
        mlp_forward(network, inputs[i], output);
        printf("XOR(%.0f, %.0f) = %.4f\n", 
               mat_get(inputs[i], 0, 0), mat_get(inputs[i], 1, 0),
               mat_get(output, 0, 0));
        mat_free(output);
    }
    
    for (int i = 0; i < 4; i++) {
        mat_free(inputs[i]);
        mat_free(targets[i]);
    }
    mlp_free(network);

}

int main() {
    printf("=== Neural Network Backpropagation Test ===\n");
    
    srand(time(NULL));

    test_training();
    
    printf("\n=== All Tests Complete ===\n");
    
    return 0;
}

