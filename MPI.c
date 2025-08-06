#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size, n;
    double *a, *b;
    double local_sum = 0.0, total_sum;
    int local_start, local_end, i;
    double start_time, end_time, total_time = 0.0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parse the vector length from the command line argument
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s <vector_length>\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    n = atoi(argv[1]);  // The vector size specified by the user

    // Allocate memory for vectors (done once)
    a = (double*)malloc(n * sizeof(double));
    b = (double*)malloc(n * sizeof(double));

    // Generate random vectors only once on the root process (rank 0)
    if (rank == 0) {
        srand(time(NULL));  // Set the seed for randomness

        // Randomly generate values for vectors a and b
        for (i = 0; i < n; i++) {
            a[i] = (double)rand() / RAND_MAX;  // Random values between 0 and 1
            b[i] = (double)rand() / RAND_MAX;  // Random values between 0 and 1
        }
    }

    // Broadcast vector length and vectors to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(a, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Run the code 10 times
    for (int run = 0; run < 10; run++) {
        // Synchronize all processes before starting the timer
        MPI_Barrier(MPI_COMM_WORLD);

        // Start measuring time for this run
        start_time = MPI_Wtime();

        // Calculate the local range for each process
        int chunk_size = n / size;
        local_start = rank * chunk_size;
        local_end = (rank == size - 1) ? n : local_start + chunk_size;

        // Compute the local sum of squared differences
        for (i = local_start; i < local_end; i++) {
            local_sum += (a[i] - b[i]) * (a[i] - b[i]);
        }

        // Reduce all local sums to process 0 using MPI_Reduce
        MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // End measuring time for parallel work
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes are done

        // Calculate and print the result on process 0
        if (rank == 0) {
            double distance = sqrt(total_sum);
            printf("Run %d: Euclidean Distance: %f\n", run + 1, distance);

            // End measuring time and calculate the time taken for this run
            end_time = MPI_Wtime();
            double time_taken = end_time - start_time; // Time for this run
            printf("Run %d: Time taken: %f seconds\n", run + 1, time_taken);

            total_time += time_taken; // Add to total time
        }
    }

    // Calculate and print the average time after 10 runs
    if (rank == 0) {
        printf("Average Time taken: %f seconds\n", total_time / 10);
    }

    // Free memory after all runs
    free(a);
    free(b);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}

