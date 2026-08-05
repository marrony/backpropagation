.PHONY: all
.DEFAULT_GOAL := all

BINDIR := bin
SOURCES := $(wildcard *.c)
PROGS := $(addprefix $(BINDIR)/, $(SOURCES:.c=))

raylib-5/libraylib.a:
	make -C raylib-5 RAYLIB_SRC_PATH=.

CFLAGS_COMMON=-std=c17 -Werror -Wall -Wextra \
		-Wno-initializer-overrides -pedantic -I ./raylib-5 \
		-DPCRE2_CODE_UNIT_WIDTH=8 $(shell pkg-config --cflags libpcre2-8) \
		-march=armv8-a

CFLAGS_DEBUG=$(CFLAGS_COMMON) -g \
    -fsanitize=signed-integer-overflow \
    -fsanitize=unsigned-integer-overflow \
    -fsanitize=address -fassociative-math

CFLAGS_RELEASE=$(CFLAGS_COMMON) -O3

CFLAGS=$(CFLAGS_RELEASE)

LDFLAGS=-L ./raylib-5 -lraylib -framework Foundation \
    -framework CoreServices -framework CoreGraphics \
    -framework AppKit -framework IOKit -framework OpenCL \
		$(shell pkg-config --libs libpcre2-8)

all: $(PROGS)

generated/alice.h: generated books/alice.txt
	(cat books/alice.txt && printf '\0') | xxd -i -name alice_txt > generated/alice.h

generated/vocab.h: generated bin/genvocab
	cp generated/vocab.h generated/vocab_old.h
	./bin/genvocab > generated/vocab.h
	diff generated/vocab_old.h generated/vocab.h || true

$(BINDIR)/node: node.c nn.h node.h transformer.h | $(BINDIR)
	cc $(CFLAGS_DEBUG) $< -o $@ $(LDFLAGS)

$(BINDIR)/genvocab: genvocab.c tokenizer.h generated/alice.h generated/vocab.h | $(BINDIR)
	cc $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/cnn-node: cnn-node.c
	echo "cnn-node.c is broken"

$(BINDIR)/%: %.c nn.h node.h game.h tokenizer.h transformer.h gpu.h generated/alice.h generated/vocab.h | $(BINDIR) raylib-5/libraylib.a
	cc $(CFLAGS) $< -o $@ $(LDFLAGS)

generated:
	mkdir -p $@

bin:
	mkdir -p $@


# $@  The target            The file name of the target of the rule.
# $<  The first dependency  The name of the first dependency (prerequisite).
# $^  All dependencies      The names of all the dependencies, with spaces between them, and without duplicates.
# $?  Newer dependencies    The names of all dependencies that are newer than the target.
# $*  The stem              The part of the target name that matched the % in a pattern rule.

