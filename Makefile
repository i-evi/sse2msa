CC = gcc

MACHINE := $(shell uname -m)

ifeq ($(MACHINE), x86_64)
	CFLAGS += -msse4.2
else ifeq ($(MACHINE), mips)
	CFLAGS += -mmsa
else ifeq ($(MACHINE), mips64)
	CFLAGS += -mmsa
endif

BIN = test example

ALL = $(patsubst %, ./tests/%, $(BIN))

all: $(ALL)

%: %.c
	$(CC) $< -o $@ $(CFLAGS)

clean:
	$(RM) $(ALL)
