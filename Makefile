.PHONY: all

all: bin/back-propagation bin/game bin/graph

raylib-5/libraylib.a:
	make -C raylib-5 RAYLIB_SRC_PATH=.

CFLAGS=-Werror -Wall -Wextra -Wno-initializer-overrides -pedantic \
    -I ./raylib-5 \
    -fsanitize=signed-integer-overflow \
    -fsanitize=unsigned-integer-overflow \
    -fsanitize=address

LDFLAGS=-L ./raylib-5 -lraylib -framework Foundation \
    -framework CoreServices -framework CoreGraphics \
    -framework AppKit -framework IOKit

bin/back-propagation: back-propagation.c raylib-5/libraylib.a | bin
	cc -O3 -g $(CFLAGS) back-propagation.c -o bin/back-propagation $(LDFLAGS)

bin/game: game.c  raylib-5/libraylib.a | bin
	cc -O3 -g $(CFLAGS) game.c -o bin/game $(LDFLAGS)

bin/graph: graph.c  raylib-5/libraylib.a | bin
	cc -O3 -g $(CFLAGS) graph.c -o bin/graph $(LDFLAGS)

bin:
	mkdir -p $@

