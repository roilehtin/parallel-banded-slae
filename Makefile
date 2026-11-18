CC ?= gcc
MPICC ?= mpicc
CFLAGS ?= -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wpedantic
BIN := bin

.PHONY: all clean
all: $(BIN)/band_pthreads $(BIN)/band_openmp $(BIN)/band_mpi

$(BIN):
	mkdir -p $(BIN)

$(BIN)/band_pthreads: src/band_pthreads.c src/band_common.h | $(BIN)
	$(CC) $(CFLAGS) -pthread $< -lm -o $@

$(BIN)/band_openmp: src/band_openmp.c src/band_common.h | $(BIN)
	$(CC) $(CFLAGS) -fopenmp $< -lm -o $@

$(BIN)/band_mpi: src/band_mpi.c src/band_common.h | $(BIN)
	$(MPICC) $(CFLAGS) $< -lm -o $@

clean:
	rm -f $(BIN)/band_pthreads $(BIN)/band_openmp $(BIN)/band_mpi
