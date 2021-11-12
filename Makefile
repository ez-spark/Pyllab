DIR:= ./src/


SRCS = $(wildcard $(DIR)*.c)

PROGS = $(patsubst %.c,%,$(SRCS))
PROGTESTS = $(patsubst %.c,%,$(SRCSTEST))

CC:= gcc

CFLAGS:= -O3 -mavx2 -g -pg -fPIC
LDLIBS:= -lm -lpthread
LABLIB:= -lllab

all: $(PROGS)

%: %.c
	$(CC) -c $< -o $@.o $(CFLAGS) $(LDLIBS)

create: $(DIR)
	ar r libllab.a $(DIR)*.o
	rm $(DIR)*.o

dynamic: $(DIR)
	gcc $(DIR)*.o -shared -o libllab.so 
	rm $(DIR)*.o
