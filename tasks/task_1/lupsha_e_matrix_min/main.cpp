#include <gtest/gtest.h>
#include <vector>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include "./matrix_min.h"

TEST(MatrixMinTest, RandomMatrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> matrix;

    if (world.rank() == 0) {
        matrix = get_random_matrix(5, 5);
    }

    int parallel_min = get_matrix_min_prl(matrix);

    if (world.rank() == 0) {
        int seq_min = get_matrix_min_seq(matrix);
        ASSERT_EQ(seq_min, parallel_min);
    }
}

TEST(MatrixMinTest, EmptyMatrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> matrix;

    if (world.rank() == 0) {
        matrix = {};
    }

    int parallel_min = get_matrix_min_prl(matrix);

    if (world.rank() == 0) {
        int seq_min = 0;
        ASSERT_EQ(seq_min, parallel_min);
    }
}

TEST(MatrixMinTest, SingleElementMatrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> matrix;

    if (world.rank() == 0) {
        matrix = {{42}};
    }

    int parallel_min = get_matrix_min_prl(matrix);

    if (world.rank() == 0) {
        int seq_min = 42;
        ASSERT_EQ(seq_min, parallel_min);
    }
}

TEST(MatrixMinTest, LargeMatrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> matrix;

    if (world.rank() == 0) {
        matrix = get_large_matrix();
    }

    int parallel_min = get_matrix_min_prl(matrix);

    if (world.rank() == 0) {
        int seq_min = get_matrix_min_seq(matrix);
        ASSERT_EQ(seq_min, parallel_min);
    }
}

TEST(MatrixMinTest, EqualElementsMatrix) {
    boost::mpi::communicator world;
    std::vector<std::vector<int>> matrix;

    if (world.rank() == 0) {
        matrix = {{5, 5, 5}, {5, 5, 5}, {5, 5, 5}};
    }

    int parallel_min = get_matrix_min_prl(matrix);

    if (world.rank() == 0) {
        int seq_min = 5;
        ASSERT_EQ(seq_min, parallel_min);
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    if (world.rank() != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }
    return RUN_ALL_TESTS();
}