INC="inc"
NVCCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: transpose vectorSum MatrixMul

transpose: transpose.cu
	$(NVCC) $(NVCCFLAGS) transpose.cu -o transpose $(LFLAGS)

vectorSum:	vectorSum.cu
	$(NVCC)	$(NVCCFLAGS) vectorSum.cu -o vectorSum $(LFLAGS)

MatrixMul:	MatrixMul.cu
	$(NVCC)	$(NVCCFLAGS) MatrixMul.cu -o MatrixMul $(LFLAGS)

clean:
	transpose vectorSum MatrixMul
