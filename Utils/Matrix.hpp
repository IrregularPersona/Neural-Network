#pragma once

// Neural Network Matrix Implementation
// Optimized for float32 operations with SIMD support
// Linux-focused implementation using aligned_alloc

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

typedef struct Matrix {
    float *data;        // Data.
    size_t rows;        // Number of rows
    size_t cols;        // Number of columns
    size_t stride;      // Row stride 
} Matrix;

// ============================================================================
// CORE MATRIX FUNCTIONS
// ============================================================================

Matrix* mat_create(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0) {
        return NULL;
    }
    
    if (rows > SIZE_MAX / cols) {
        return NULL;
    }
    
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }
    
    size_t total_elements = rows * cols;
    
    mat->data = (float*)aligned_alloc(32, total_elements * sizeof(float));
    
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    
    mat->rows = rows;
    mat->cols = cols;
    mat->stride = cols;  
    
    memset(mat->data, 0, total_elements * sizeof(float));
    
    return mat;
}

Matrix* mat_create_with_value(size_t rows, size_t cols, float init_value) {
    Matrix* mat = mat_create(rows, cols);
    if (!mat) {
        return NULL;
    }
    
    size_t total_elements = rows * cols;
    for (size_t i = 0; i < total_elements; i++) {
        mat->data[i] = init_value;
    }
    
    return mat;
}

void mat_free(Matrix* mat) {
    if (!mat) {
        return;
    }
    
    if (mat->data) {
        free(mat->data);
    }
    
    free(mat);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Get matrix dimensions
size_t mat_rows(const Matrix* mat) {
    return mat ? mat->rows : 0;
}

size_t mat_cols(const Matrix* mat) {
    return mat ? mat->cols : 0;
}

size_t mat_size(const Matrix* mat) {
    return mat ? (mat->rows * mat->cols) : 0;
}

// Check if matrix is valid
int mat_is_valid(const Matrix* mat) {
    return (mat && mat->data && mat->rows > 0 && mat->cols > 0) ? 1 : 0;
}

// ============================================================================
// HIGH-PERFORMANCE ACCESS FUNCTIONS
// ============================================================================

// Direct data access (for performance-critical operations)
static inline float* mat_data(Matrix* mat) {
    return mat ? mat->data : NULL;
}

static inline const float* mat_data_const(const Matrix* mat) {
    return mat ? mat->data : NULL;
}

// Element access with bounds checking
static inline float mat_get(const Matrix* mat, size_t row, size_t col) {
    if (!mat || !mat->data || row >= mat->rows || col >= mat->cols) {
        return 0.0f;  // Return 0 for invalid access
    }
    return mat->data[row * mat->stride + col];
}

static inline int mat_set(Matrix* mat, size_t row, size_t col, float value) {
    if (!mat || !mat->data || row >= mat->rows || col >= mat->cols) {
        return -1;  // Error
    }
    mat->data[row * mat->stride + col] = value;
    return 0;  // Success
}

// Unsafe element access (for performance-critical loops)
static inline float mat_get_unsafe(const Matrix* mat, size_t row, size_t col) {
    return mat->data[row * mat->stride + col];
}

static inline void mat_set_unsafe(Matrix* mat, size_t row, size_t col, float value) {
    mat->data[row * mat->stride + col] = value;
}

// ============================================================================
// MATRIX OPERATIONS (Basic)
// ============================================================================

// Copy matrix
Matrix* mat_copy(const Matrix* src) {
    if (!mat_is_valid(src)) {
        return NULL;
    }
    
    Matrix* dst = mat_create(src->rows, src->cols);
    if (!dst) {
        return NULL;
    }
    
    size_t total_elements = src->rows * src->cols;
    memcpy(dst->data, src->data, total_elements * sizeof(float));
    
    return dst;
}

// Fill matrix with a value
void mat_fill(Matrix* mat, float value) {
    if (!mat_is_valid(mat)) {
        return;
    }
    
    size_t total_elements = mat->rows * mat->cols;
    for (size_t i = 0; i < total_elements; i++) {
        mat->data[i] = value;
    }
}

// Zero matrix
void mat_zero(Matrix* mat) {
    if (!mat_is_valid(mat)) {
        return;
    }
    
    size_t total_elements = mat->rows * mat->cols;
    memset(mat->data, 0, total_elements * sizeof(float));
}

// ============================================================================
// DEBUGGING AND UTILITY
// ============================================================================

// Print matrix (for debugging)
void mat_print(const Matrix* mat) {
    if (!mat_is_valid(mat)) {
        printf("Invalid matrix\n");
        return;
    }
    
    printf("Matrix [%zu x %zu]:\n", mat->rows, mat->cols);
    for (size_t i = 0; i < mat->rows; i++) {
        printf("[");
        for (size_t j = 0; j < mat->cols; j++) {
            printf("%.6f", mat_get(mat, i, j));
            if (j < mat->cols - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    }
}

// Get matrix statistics
typedef struct {
    float min, max, sum, mean;
} M_Stats;

M_Stats mat_stats(const Matrix* mat) {
    M_Stats stats = {0.0f, 0.0f, 0.0f, 0.0f};
    
    if (!mat_is_valid(mat)) {
        return stats;
    }
    
    size_t total_elements = mat->rows * mat->cols;
    if (total_elements == 0) {
        return stats;
    }
    
    stats.min = mat->data[0];
    stats.max = mat->data[0];
    
    for (size_t i = 0; i < total_elements; i++) {
        float val = mat->data[i];
        stats.sum += val;
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
    }
    
    stats.mean = stats.sum / total_elements;
    return stats;
}
