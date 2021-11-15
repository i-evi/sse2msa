CC = gcc

MACHINE := $(shell uname -m)

ifeq ($(MACHINE), x86_64)
	CFLAGS += -msse4.2
else ifeq ($(MACHINE), mips)
	CFLAGS += -mmsa -I`pwd`
else ifeq ($(MACHINE), mips64)
	CFLAGS += -mmsa -I`pwd`
endif

BIN = test example

ALL = $(patsubst %, ./tests/%, $(BIN))

all: $(ALL)

%: %.c
	$(CC) $< -o $@ $(CFLAGS)

clean:
	$(RM) $(ALL)
