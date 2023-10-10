CC=gcc
CFLAGS=-fopenmp -mavx -mavx512f
PROGRAM=solve_pi
SOURCES=main.cpp solve_pi.cpp solve_pi.h
OPTIMIZATION=-O0

.PHONY: ${PROGRAM} clean

${PROGRAM} :
	@echo "Optimization Level: ${OPTIMIZATION}"
	@${CC} ${CFLAGS} ${OPTIMIZATION} -o ${PROGRAM} ${SOURCES} && ./${PROGRAM}

clean :
	rm solve_pi
