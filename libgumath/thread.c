/*
* BSD 3-Clause License
*
* Copyright (c) 2017-2018, plures
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
*    contributors may be used to endorse or promote products derived from
*    this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"
#include "config.h"


#ifdef HAVE_PTHREAD_H
#include <pthread.h>

struct thread_info {
    pthread_t tid;
    int tnum;
    int nrows;
    int ncols;
    const gm_kernel_t *kernel;
    xnd_t **slices;
    int outer_dims;
    ndt_context_t ctx;
};

static void
init_static_context(ndt_context_t *ctx)
{
    static const ndt_context_t c = {
      .flags=0,
      .err=NDT_Success,
      .msg=ConstMsg,
      .ConstMsg="Success" };

    *ctx = c;
}

static void
clear_all_slices(xnd_t *slices[], int *nslices, int stop)
{
    for (int i = 0; i < stop; i++) {
        for (int k = 0; k < nslices[i]; k++) {
            ndt_del((ndt_t *)slices[i][k].type);
        }
        ndt_free(slices[i]);
    }
}

static void *
apply_thread(void *arg)
{
    struct thread_info *tinfo = arg;
    ALLOCA(xnd_t, stack, tinfo->nrows);

    for (int i = 0; i < tinfo->nrows; i++) {
        stack[i] = tinfo->slices[i][tinfo->tnum];
    }

    gm_apply(tinfo->kernel, stack, tinfo->outer_dims, &tinfo->ctx);
    return NULL;
}

int
gm_apply_thread(const gm_kernel_t *kernel, xnd_t stack[], int outer_dims,
                uint32_t flags, const int64_t nthreads, ndt_context_t *ctx)
{
    const int nrows = (int)kernel->set->sig->Function.nargs;
    ALLOCA(xnd_t *, slices, nrows);
    ALLOCA(int, nslices, nrows);
    struct thread_info *tinfo;
    int ncols, tnum;

    if (nthreads <= 1 || nrows == 0 || outer_dims == 0 ||
        !(flags & NDT_STRIDED)) {
        return gm_apply(kernel, stack, outer_dims, ctx);
    }

    for (int i = 0; i < nrows; i++) {
        int64_t ncols = nthreads;
        slices[i] = xnd_split(&stack[i], &ncols, outer_dims, ctx);
        if (ndt_err_occurred(ctx)) {
            clear_all_slices(slices, nslices, i);
            return -1;
        }
        nslices[i] = ncols;
    }

    ncols = nslices[0];
    for (int i = 1; i < nrows; i++) {
        if (nslices[i] != ncols) {
            clear_all_slices(slices, nslices, nrows);
            ndt_err_format(ctx, NDT_RuntimeError,
                "equal subdivision in threaded apply loop failed");
            return -1;
        }
    }

    tinfo = ndt_calloc(nthreads, sizeof *tinfo);
    if (tinfo == NULL) {
        clear_all_slices(slices, nslices, nrows);
        (void)ndt_memory_error(ctx);
        return -1;
    }

    for (tnum = 0; tnum < ncols; tnum++) {
        tinfo[tnum].tnum = tnum;
        tinfo[tnum].kernel = kernel;
        tinfo[tnum].nrows = nrows;
        tinfo[tnum].ncols = ncols;
        tinfo[tnum].slices = slices;
        tinfo[tnum].outer_dims = outer_dims;
        init_static_context(&tinfo[tnum].ctx);

        int ret = pthread_create(&tinfo[tnum].tid, NULL, &apply_thread,
                                 &tinfo[tnum]);
        if (ret != 0) {
            clear_all_slices(slices, nslices, nrows);
            ndt_err_format(ctx, NDT_RuntimeError, "could not create thread");
            return -1;
        }
    }

    int ret = 0;
    for (tnum = 0; tnum < ncols; tnum++) {
        ret |= pthread_join(tinfo[tnum].tid, NULL);
        if (ndt_err_occurred(&tinfo[tnum].ctx)) {
            if (!ndt_err_occurred(ctx)) {
                ndt_err_format(ctx, tinfo[tnum].ctx.err,
                               ndt_context_msg(&tinfo[tnum].ctx));
            }
            ndt_err_clear(&tinfo[tnum].ctx);
        }
    }

    if (ret != 0 && !ndt_err_occurred(ctx)) {
        ndt_err_format(ctx, NDT_RuntimeError, "error in thread");
    }
 
    clear_all_slices(slices, nslices, nrows);

    return ndt_err_occurred(ctx) ? -1 : 0;
}
#endif
