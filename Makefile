.PHONY: all
.DEFAULT_GOAL := all

BINDIR := bin
SOURCES := $(wildcard *.c)
PROGS := $(addprefix $(BINDIR)/, $(SOURCES:.c=))

raylib-5/libraylib.a:
	make -C raylib-5 RAYLIB_SRC_PATH=.

CFLAGS_COMMON=-std=c23 -Werror -Wall -Wextra \
		-Wno-initializer-overrides -pedantic -I ./raylib-5 \
		-DPCRE2_CODE_UNIT_WIDTH=8 $(shell pkg-config --cflags libpcre2-8) \
		-march=armv8-a -DGPU_COMPUTATION=1

CFLAGS_DEBUG=$(CFLAGS_COMMON) -g \
    -fsanitize=signed-integer-overflow \
    -fsanitize=unsigned-integer-overflow \
    -fsanitize=address -fassociative-math

CFLAGS_RELEASE=$(CFLAGS_COMMON) -O3
    # -mcpu=native -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
    # -Rpass-analysis=loop-vectorize

CFLAGS=$(CFLAGS_RELEASE)

LDFLAGS=-L ./raylib-5 -lraylib -framework Foundation \
    -framework CoreServices -framework CoreGraphics \
    -framework AppKit -framework IOKit -framework OpenCL \
		$(shell pkg-config --libs libpcre2-8)

all: $(PROGS)

generated/alice.h: generated books/alice.txt
	(cat books/alice.txt && printf '\0') | xxd -i -name alice_txt > generated/alice.h

VOCAB_SIZE=2048
EOS_ID=256
CDATA_OUT=cdata-$(VOCAB_SIZE)
TS_CDATA_OUT=ts-cdata-$(VOCAB_SIZE)
TOKENIZER_CMD=$(BINDIR)/tokenizer-$(VOCAB_SIZE)
VOCAB_HEADER=vocabs/vocab_$(VOCAB_SIZE).h

generated/data.bin: distill/data/seqkd_gold_train.jsonl distill/data/seqkd_gold_dev.jsonl
	cd distill && uv run scripts/dump_samples.py data/seqkd_gold_*.jsonl > ../generated/data.bin

$(VOCAB_HEADER): generated bin/genvocab # generated/data.bin
	./bin/genvocab --max-vocab=$(VOCAB_SIZE) < generated/data.bin > $(VOCAB_HEADER)

$(BINDIR)/node: node.c nn.h node.h transformer.h | $(BINDIR)
	cc $(CFLAGS_DEBUG) $< -o $@ $(LDFLAGS)

$(TOKENIZER_CMD): tokenizer.c tokenizer.h | $(BINDIR)
	cc $(CFLAGS) $< -o $@ $(LDFLAGS) -DVOCAB_HEADER=$(VOCAB_HEADER) -DVOCAB_SIZE=$(VOCAB_SIZE)

$(BINDIR)/genvocab: genvocab.c tokenizer.h | $(BINDIR)
	cc $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/cnn-node: cnn-node.c
	echo "cnn-node.c is broken"

$(BINDIR)/%: %.c nn.h node.h game.h tokenizer.h transformer.h gpu.h | $(BINDIR) raylib-5/libraylib.a
	cc $(CFLAGS) $< -o $@ $(LDFLAGS) -DVOCAB_HEADER='"$(VOCAB_HEADER)"' -DVOCAB_SIZE=$(VOCAB_SIZE)

$(BINDIR)/gentext2: gentext2.c nn.h node.h game.h tokenizer.h transformer.h gpu.h gpu_kernels.c Makefile | $(BINDIR)
	cc $(CFLAGS) $< -o $@ $(LDFLAGS) -DVOCAB_HEADER='"$(VOCAB_HEADER)"' -DVOCAB_SIZE=$(VOCAB_SIZE)

$(BINDIR)/test_gpu: test_gpu.c nn.h gpu.h gpu_kernels.c | $(BINDIR)
	cc $(CFLAGS) $< -o $@ $(LDFLAGS)

generated:
	mkdir -p $@

bin:
	mkdir -p $@

corpus:
	cd distill && rm -rf data/instruct
	cd distill && uv run scripts/build_seqkd_data.py --split dev --gold
	cd distill && uv run scripts/build_seqkd_data.py --split train --gold

# teacher generations, hours + 8GB download
# 	uv run scripts/build_seqkd_data.py --split train

export: $(TOKENIZER_CMD)
	# cd distill && rm -f $(CDATA_OUT)/*
	cd distill && uv run scripts/export_seqkd_ids.py \
		--jsonl data/instruct/seqkd_gold_dev.jsonl --out $(CDATA_OUT) \
		--vocab-size $(VOCAB_SIZE) --eos-id $(EOS_ID) --max-len 3072 \
		--tokenizer-cmd '../$(TOKENIZER_CMD)'
	cd distill && uv run scripts/export_seqkd_ids.py \
		--jsonl data/instruct/seqkd_gold_train.jsonl --out $(CDATA_OUT) \
		--vocab-size $(VOCAB_SIZE) --eos-id $(EOS_ID) --max-len 3072 \
		--tokenizer-cmd '../$(TOKENIZER_CMD)'

corpus-ts:
	cd distill && rm -rf data/tinystories
	cd distill && uv run scripts/build_seqkd_data.py --config configs/tinystories.yaml --split dev --gold
	cd distill && uv run scripts/build_seqkd_data.py --config configs/tinystories.yaml --split train --gold

export-ts: $(TOKENIZER_CMD)
	# cd distill && rm -f $(TS_CDATA_OUT)/*
	cd distill && uv run scripts/export_seqkd_ids.py \
		--jsonl data/tinystories/seqkd_gold_dev.jsonl --out $(TS_CDATA_OUT) \
		--vocab-size $(VOCAB_SIZE) --eos-id $(EOS_ID) --bos-id $(EOS_ID) \
		--max-len 3072 --tokenizer-cmd '../$(TOKENIZER_CMD)'
	cd distill && uv run scripts/export_seqkd_ids.py \
		--jsonl data/tinystories/seqkd_gold_train.jsonl --out $(TS_CDATA_OUT) \
		--vocab-size $(VOCAB_SIZE) --eos-id $(EOS_ID) --bos-id $(EOS_ID) \
		--max-len 3072 --tokenizer-cmd '../$(TOKENIZER_CMD)'

# $@  The target            The file name of the target of the rule.
# $<  The first dependency  The name of the first dependency (prerequisite).
# $^  All dependencies      The names of all the dependencies, with spaces between them, and without duplicates.
# $?  Newer dependencies    The names of all dependencies that are newer than the target.
# $*  The stem              The part of the target name that matched the % in a pattern rule.

