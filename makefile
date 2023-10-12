CC=gcc
CFLAGS=-fopenmp
PROGRAM=solve_pi
SOURCES=main.cpp solve_pi.cpp solve_pi.h
OPTIMIZATION=-O0

check_sse2 := $(shell lscpu | grep -q sse && echo true)
ifeq ($(check_sse2),true)
CFLAGS += -DSSE2 -msse2
endif

check_avx := $(shell lscpu | grep -q avx && echo true)
ifeq ($(check_avx),true)
CFLAGS += -DAVX -mavx
endif

check_avx512f := $(shell lscpu | grep -q avx512f && echo true)
ifeq ($(check_avx512f),true)
CFLAGS += -DAVX512F -mavx512f
endif

.PHONY: ${PROGRAM} clean

${PROGRAM} :
	@echo "Optimization Level: ${OPTIMIZATION}"
	@${CC} ${CFLAGS} ${OPTIMIZATION} -o ${PROGRAM} ${SOURCES} && ./${PROGRAM}

clean :
	rm solve_pi
