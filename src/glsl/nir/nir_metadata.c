/*
 * Copyright © 2014 Intel Corporation
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * Authors:
 *    Jason Ekstrand (jason@jlekstrand.net)
 */

#include "nir.h"

/*
 * Handles management of the metadata.
 */

void
nir_metadata_require(nir_function_impl *impl, nir_metadata required)
{
#define NEEDS_UPDATE(X) ((required & ~impl->valid_metadata) & (X))

   if (NEEDS_UPDATE(nir_metadata_block_index))
      nir_index_blocks(impl);
   if (NEEDS_UPDATE(nir_metadata_dominance))
      nir_calc_dominance_impl(impl);
   if (NEEDS_UPDATE(nir_metadata_live_variables))
      nir_live_variables_impl(impl);

#undef NEEDS_UPDATE

   impl->valid_metadata |= required;
}

void
nir_metadata_dirty(nir_function_impl *impl, nir_metadata preserved)
{
   impl->valid_metadata &= preserved;
}