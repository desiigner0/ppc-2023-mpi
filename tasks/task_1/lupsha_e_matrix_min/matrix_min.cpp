// Copyright 2023 Lupsha Egor

#include <algorithm>
#include <random>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <task_1/lupsha_e_matrix_min/matrix_min.h>

namespace mpi = boost::mpi;

std::vector<std::vector<int>> get_random_matrix(int rows, int cols) {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = gen() % 100;
        }
    }
    return matrix;
}
int get_matrix_min_seq(const std::vector<std::vector<int>>& matrix) {
    int min = matrix[0][0];
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            if (matrix[i][j] < min) {
                min = matrix[i][j];
            }
        }
    }
    return min;
}
int get_matrix_min_prl(const std::vector<std::vector<int>>& matrix) {
    mpi::environment env;
    mpi::communicator comm;

    const int numRows = matrix.size();
    const int numCols = matrix[0].size();
    const int totalElements = numRows * numCols;
    const int localElements = totalElements / comm.size();
    const int remainingElements = totalElements % comm.size();

    int localStart = comm.rank() * localElements;
    int localEnd = localStart + localElements;
    if (comm.rank() == 0) {
        localEnd += remainingElements;
    }

    int localMin = matrix[0][0];

    for (int i = localStart; i < localEnd; ++i) {
        int row = i / numCols;
        int col = i % numCols;
        if (matrix[row][col] < localMin) {
            localMin = matrix[row][col];
        }
    }

    int globalMin;
    boost::mpi::reduce(comm, localMin, globalMin, boost::mpi::minimum<int>(), 0);

    return globalMin;
}
