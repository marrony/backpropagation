.PHONY: all
.DEFAULT_GOAL := all

BINDIR := bin
SOURCES := $(wildcard *.c)
PROGS := $(addprefix $(BINDIR)/, $(SOURCES:.c=))

raylib-5/libraylib.a:
	make -C raylib-5 RAYLIB_SRC_PATH=.

CFLAGS=-std=c17 -Werror -Wall -Wextra -Wno-initializer-overrides -pedantic \
    -I ./raylib-5 \
    -fsanitize=signed-integer-overflow \
    -fsanitize=unsigned-integer-overflow \
    -fsanitize=address -fassociative-math \
    -O3

LDFLAGS=-L ./raylib-5 -lraylib -framework Foundation \
    -framework CoreServices -framework CoreGraphics \
    -framework AppKit -framework IOKit

all: $(PROGS)

generated/alice.h: generated books/alice.txt
	(cat books/alice.txt && printf '\0') | xxd -i -name alice_txt > generated/alice.h

generated/vocab.h: generated bin/genvocab
	./bin/genvocab > generated/vocab.h

$(BINDIR)/%: %.c nn.h node.h game.h tokenizer.h generated/alice.h generated/vocab.h | $(BINDIR) raylib-5/libraylib.a
	cc -O3 -g $(CFLAGS) $< -o $@ $(LDFLAGS)

generated:
	mkdir -p $@

bin:
	mkdir -p $@


# $@  The target            The file name of the target of the rule.
# $<  The first dependency  The name of the first dependency (prerequisite).
# $^  All dependencies      The names of all the dependencies, with spaces between them, and without duplicates.
# $?  Newer dependencies    The names of all dependencies that are newer than the target.
# $*  The stem              The part of the target name that matched the % in a pattern rule.

