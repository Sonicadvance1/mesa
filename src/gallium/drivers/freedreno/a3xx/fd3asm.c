/* -*- mode: C; c-file-style: "k&r"; tab-width 4; indent-tabs-mode: t; -*- */

/*
 * Copyright (C) 2014 Rob Clark <robclark@freedesktop.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <err.h>

#include "tgsi/tgsi_parse.h"
#include "tgsi/tgsi_text.h"
#include "tgsi/tgsi_dump.h"

#include "fd3_compiler.h"
#include "fd3_program.h"

#include "freedreno_lowering.h"

#include "instr-a3xx.h"
#include "ir3.h"

int fd3_compile_shader(struct fd3_shader_stateobj *so,
		const struct tgsi_token *tokens);

int fd3_compile_shader_old(struct fd3_shader_stateobj *so,
		const struct tgsi_token *tokens);

void fd3_emu_run(void *instrs, uint32_t instrs_count,
		float *consts, uint32_t consts_size,
		float *regs, uint32_t regs_size);

/* a nice hack to make a standalone compiler (assembler) for testing */


static void dump_info(struct fd3_shader_stateobj *so)
{
	struct ir3_shader_info info;
	uint32_t *bin;
	const char *type = (so->type == SHADER_VERTEX) ? "VERT" : "FRAG";

	// for debug, dump some before/after info:
	bin = ir3_shader_assemble(so->ir, &info);
	if (fd_mesa_debug & FD_DBG_DISASM) {
		unsigned i;

		debug_printf("%s: disasm:\n", type);
		disasm_a3xx(bin, info.sizedwords, 0, so->type);

		debug_printf("%s: outputs:", type);
		for (i = 0; i < so->outputs_count; i++) {
			uint8_t regid = so->outputs[i].regid;
			debug_printf(" r%d.%c", (regid >> 2), "xyzw"[regid & 0x3]);
		}
		debug_printf("\n");
		if (so->type == SHADER_VERTEX) {
			debug_printf("%s: inputs:", type);
			for (i = 0; i < so->inputs_count; i++) {
				uint8_t regid = so->inputs[i].regid;
				debug_printf(" r%d.%c", (regid >> 2), "xyzw"[regid & 0x3]);
			}
			debug_printf("\n");
		}
	}
	debug_printf("%s: %u instructions, %d half, %d full\n\n",
			type, info.instrs_count, info.max_half_reg + 1, info.max_reg + 1);
	free(bin);
}


static void
read_file(const char *filename, void **ptr, size_t *size)
{
	int fd, ret;
	struct stat st;

	*ptr = MAP_FAILED;

	fd = open(filename, O_RDONLY);
	if (fd == -1)
		errx(1, "couldn't open `%s'", filename);

	ret = fstat(fd, &st);
	if (ret)
		errx(1, "couldn't stat `%s'", filename);

	*size = st.st_size;
	*ptr = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (*ptr == MAP_FAILED)
		errx(1, "couldn't map `%s'", filename);

	close(fd);
}

static float inputs[64], consts[256];
static float outputs1[64], outputs2[64];

static void randomize(float *buf, int num)
{
	unsigned i;
	for (i = 0; i < num; i++)
		*(buf++) = drand48();
}


static void shader_run(struct fd3_shader_stateobj *so, float *outputs)
{
	struct ir3_shader_info info;
	float regs[256];
	unsigned i, j;
	uint32_t *bin;

	bin = ir3_shader_assemble(so->ir, &info);

	assert(((info.max_const + 1) * 4) < ARRAY_SIZE(consts));
	assert(((info.max_reg + 1) * 4) < ARRAY_SIZE(regs));
	// TODO half-reg support..

	/* give registers random initial values: */
	randomize(regs, ARRAY_SIZE(regs));

	/* copy input values into registers: */
	if (so->type == SHADER_VERTEX) {
		for (i = 0; i < so->inputs_count; i++) {
			uint32_t regid = so->inputs[i].regid;
			for (j = 0; j < 4; j++)
				if (so->inputs[i].compmask & (1 << j))
					regs[regid + j] = inputs[(i * 4) + j];
		}
	} else {
		regs[0] = inputs[0];
		regs[1] = inputs[1];
	}

	fd3_emu_run(bin, info.instrs_count,
			consts, (info.max_const + 1) * 4,
			regs, (info.max_reg + 1) * 4);

	/* copy output values back from registers: */
	for (i = 0; i < so->outputs_count; i++) {
		uint32_t regid = so->outputs[i].regid;
		for (j = 0; j < 4; j++)
			outputs[(i * 4) + j] = regs[regid + j];
	}

	/* dump registers.. */
	printf("----------\n");
	for (i = 0; i <= (info.max_reg + 1) * 4; i += 4) {
		printf("%02d: %08x %08x %08x %08x\n", (i / 4),
				fui(regs[i+0]), fui(regs[i+1]),
				fui(regs[i+2]), fui(regs[i+3]));
	}
	printf("----------\n");
}

static void shader_test(struct fd3_shader_stateobj *ref,
		struct fd3_shader_stateobj *test)
{
	unsigned i, j;

	randomize(inputs, ARRAY_SIZE(inputs));
	randomize(consts, ARRAY_SIZE(consts));

	shader_run(ref,  outputs1);
	shader_run(test, outputs2);

	assert(ref->outputs_count == test->outputs_count);

	for (i = 0; i < ref->outputs_count; i++) {
		for (j = 0; j < 4; j++) {
			int idx = (i * 4) + j;
			if (fui(outputs1[idx]) != fui(outputs2[idx])) {
				printf("out%d.%c: %f (%08x) vs %f (%08x)\n",
						i, "xyzw"[j],
						outputs1[idx], fui(outputs1[idx]),
						outputs2[idx], fui(outputs2[idx]));
			}
		}
	}
}

int main(int argc, char **argv)
{
	int i, ret;

	fd_mesa_debug |= FD_DBG_DISASM | FD_DBG_OPTDUMP | FD_DBG_OPTMSGS;

	for (i = 1; i < argc; i++) {
		const char *filename = argv[i];
		struct tgsi_token tokens[10000], *toks;
		struct tgsi_parse_context parse;
		struct fd3_shader_stateobj so_old, so_new;
		void *ptr;
		size_t size;

		memset(&so_old, 0, sizeof(so_old));
		memset(&so_new, 0, sizeof(so_new));

		printf("************ Reading %s\n", filename);
		read_file(filename, &ptr, &size);

		if (!tgsi_text_translate(ptr, tokens, Elements(tokens)))
			errx(1, "could not parse `%s'", filename);

		toks = fd_transform_lowering(tokens);
		if (!toks)
			toks = tokens;

		tgsi_parse_init(&parse, toks);
		switch (parse.FullHeader.Processor.Processor) {
		case TGSI_PROCESSOR_FRAGMENT:
			so_old.type = so_new.type = SHADER_FRAGMENT;
			break;
		case TGSI_PROCESSOR_VERTEX:
			so_old.type = so_new.type = SHADER_VERTEX;
			break;
		case TGSI_PROCESSOR_COMPUTE:
			so_old.type = so_new.type = SHADER_COMPUTE;
			break;
		}

		/* with old compiler: */
		ret = fd3_compile_shader_old(&so_old, toks);
		if (ret) {
			printf("old compiler failed!\n");
			return ret;
		}
		dump_info(&so_old);

		/* with new compiler: */
		ret = fd3_compile_shader(&so_new, toks);
		if (ret) {
			printf("new compiler failed!\n");
			return ret;
		}
		dump_info(&so_new);

		shader_test(&so_old, &so_new);
	}
}
