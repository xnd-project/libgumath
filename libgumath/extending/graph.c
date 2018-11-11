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
#include <math.h>
#include <complex.h>
#include <inttypes.h>
#include "ndtypes.h"
#include "xnd.h"
#include "gumath.h"


/*****************************************************************************/
/*                           Check graph invariants                          */
/*****************************************************************************/

static bool
graph_constraint(const void *graph, ndt_context_t *ctx)
{
    const xnd_t *g = (xnd_t *)graph;
    int64_t start2, step2; /* start, step of ndim2 (the graph) */
    int64_t start1, step1; /* start, step of ndim1 (an array of edges) */
    int64_t N;  /* number of nodes */


    N = ndt_var_indices(&start2, &step2, g->type, g->index, ctx);
    if (N < 0) {
        return false;
    }

    for (int32_t u = 0; u < N; u++) {
        const xnd_t edges = xnd_var_dim_next(g, start2, step2, u);
        const int64_t nedges = ndt_var_indices(&start1, &step1, edges.type,
                                               edges.index, ctx);
        if (nedges < 0) {
            return false;
        }

        for (int64_t k = 0; k < nedges; k++) {
            const xnd_t tuple = xnd_var_dim_next(&edges, start1, step1, k);
            const xnd_t target = xnd_tuple_next(&tuple, 0, ctx);
            const int32_t v = *(int32_t *)target.ptr;

            if (v < 0 || v >= N) {
                ndt_err_format(ctx, NDT_ValueError,
                    "node id must be in range [0, N-1]");
                return false;
            }
        }
    }

    return true;
}


/*****************************************************************************/
/*      Bellman-Ford single-source shortest-paths algorithm (CLRS p588)      */
/*****************************************************************************/


static const int32_t NIL = INT32_MIN;

/*
 * Allocate and initialize distance and predecessor arrays.
 * d := distance array
 * p := predecessor array
 * N := number of nodes
 * u := single source node id
 */
static int
init(double **d, int32_t **p, int64_t N, int32_t u, ndt_context_t *ctx)
{
    *d = ndt_alloc(N, sizeof **d);
    if (*d == NULL) {
        (void)ndt_memory_error(ctx);
        return -1;
    }

    *p = ndt_alloc(N, sizeof **p);
    if (*p == NULL) {
        ndt_free(*d);
        (void)ndt_memory_error(ctx);
        return -1;
    }

    for (int64_t i = 0; i < N; i++) {
        (*d)[i] = INFINITY;
        (*p)[i] = NIL;
    }

    (*d)[u] = 0;

    return 0;
}

/*
 * d := distance array
 * p := predecessor array
 * u := source node id
 * v := target node id
 * cost := cost of going from source to target
 */
static inline void
relax(double d[], int32_t p[], int32_t u, int32_t v, double cost)
{
    if (d[u] + cost < d[v]) {
        d[v] = d[u] + cost;
        p[v] = u;
    }
}

/*
 * Write the shortest path single_source ==> v to dest.
 * If dest==NULL and dsize==0, return the length of the path.
 *
 * dest := path array
 * p := predecessor array
 * single_source := single source node id
 * v := target node id
 */
static int64_t
write_path(int32_t *dest, const int32_t dsize,
           const int32_t p[], const int64_t psize,
           const int32_t s, int32_t v)
{
    int64_t n = dsize;

    while (1) {

        assert(0 <= v && v < psize);
        assert(dest == NULL || n >= 0);

        if (v == s || p[v] != NIL) {
            n--;
            if (dest != NULL) dest[n] = v;
            if (v == s) break;
        }
        else if (p[v] == NIL) {
            return 0;
        }

        v = p[v];
    }

    return dest == NULL ? -n : n;
}

static xnd_t
mk_return_array(int32_t p[], const int64_t N, const int32_t u,
                ndt_context_t *ctx)
{
    ndt_offsets_t *ndim2_offsets = NULL;
    ndt_offsets_t *ndim1_offsets = NULL;
    int32_t *ptr;
    const ndt_t *t, *type;
    int64_t sum;
    int32_t v;

    if (N+1 > INT32_MAX) {
        goto offset_overflow;
    }

    ndim2_offsets = ndt_offsets_new(2, ctx);
    if (ndim2_offsets == NULL) {
        return xnd_error;
    }
    ptr = (int32_t *)ndim2_offsets->v;
    ptr[0] = 0;
    ptr[1] = (int32_t)N;


    ndim1_offsets = ndt_offsets_new((int32_t)(N+1), ctx);
    if (ndim1_offsets == NULL) {
        ndt_decref_offsets(ndim2_offsets);
        return xnd_error;
    }

    sum = 0;
    ptr = (int32_t *)ndim1_offsets->v;
    for (v = 0; v < N; v++) {
        ptr[v] = (int32_t)sum;
        int64_t n = write_path(NULL, 0, p, N, u, v);
        sum += n;
        if (sum > INT32_MAX) {
            goto offset_overflow;
        }
    }
    ptr[v] = (int32_t)sum;


    type = ndt_from_string("node", ctx);
    if (type == NULL) {
        goto error;
    }

    t = ndt_var_dim(type, ndim1_offsets, 0, NULL, false, ctx);
    ndt_decref_offsets(ndim1_offsets);
    ndim1_offsets = NULL;
    if (t == NULL) {
        goto error;
    }

    t = ndt_var_dim(t, ndim2_offsets, 0, NULL, false, ctx);
    ndt_decref_offsets(ndim2_offsets);
    ndim2_offsets = NULL;
    if (t == NULL) {
        goto error;
    }

    xnd_master_t *x = xnd_empty_from_type(t, XND_OWN_EMBEDDED, ctx);
    if (x == NULL) {
        goto error;
    }
    xnd_t out = x->master;
    ndt_free(x);

    t = out.type->VarDim.type;
    for (v = 0; v < N; v++) {
        int32_t shape = t->Concrete.VarDim.offsets->v[v+1]-t->Concrete.VarDim.offsets->v[v];
        char *cp = out.ptr + t->Concrete.VarDim.offsets->v[v] * t->Concrete.VarDim.itemsize;
        (void)write_path((int32_t *)cp, shape, p, N, u, v);
    }

    return out;

error:
    ndt_decref_offsets(ndim2_offsets);
    ndt_decref_offsets(ndim1_offsets);
    return xnd_error;

offset_overflow:
    ndt_err_format(ctx, NDT_ValueError, "overflow in int32_t offsets");
    goto error;
}

static int
shortest_path(xnd_t stack[], ndt_context_t *ctx)
{
    const int32_t single_source = *(int32_t *)stack[1].ptr; /* start node */
    int64_t start2, step2; /* start, step of ndim2 (the graph) */
    int64_t start1, step1; /* start, step of ndim1 (an array of edges) */
    double *d;  /* distance array */
    int32_t *p; /* predecessor array */
    int64_t N;  /* number of nodes */

    /* graph in adjacency list representation */
    const xnd_t graph = xnd_nominal_next(&stack[0], ctx);
    if (graph.ptr == NULL) {
        return -1;
    }

    N = ndt_var_indices(&start2, &step2, graph.type, graph.index, ctx);
    if (N < 0) {
        return -1;
    }

    if (init(&d, &p, N, single_source, ctx) < 0) {
        return -1;
    }

    for (int64_t i = 0; i < N-1; i++) {
        for (int32_t u = 0; u < N; u++) {
            const xnd_t edges = xnd_var_dim_next(&graph, start2, step2, u);
            const int64_t nedges = ndt_var_indices(&start1, &step1, edges.type,
                                                   edges.index, ctx);
            if (nedges < 0) {
                return -1;
            }

            for (int64_t k = 0; k < nedges; k++) {
                const xnd_t tuple = xnd_var_dim_next(&edges, start1, step1, k);
                const xnd_t target = xnd_tuple_next(&tuple, 0, ctx);
                const xnd_t uvcost = xnd_tuple_next(&tuple, 1, ctx);

                const int32_t v = *(int32_t *)target.ptr;
                const double cost = *(double *)uvcost.ptr;

                relax(d, p, u, v, cost);
            }
        }
    }

    for (int32_t u = 0; u < N; u++) {
        const xnd_t edges = xnd_var_dim_next(&graph, start2, step2, u);
        const int64_t nedges = ndt_var_indices(&start1, &step1, edges.type,
                                               edges.index, ctx);
        if (nedges < 0) {
            return -1;
        }

        for (int64_t k = 0; k < nedges; k++) {
            const xnd_t tuple = xnd_var_dim_next(&edges, start1, step1, k);
            const xnd_t target = xnd_tuple_next(&tuple, 0, ctx);
            const xnd_t uvcost = xnd_tuple_next(&tuple, 1, ctx);

            const int32_t v = *(int32_t *)target.ptr;
            const double cost = *(double *)uvcost.ptr;

            if (d[u] + cost < d[v]) {
                ndt_err_format(ctx, NDT_ValueError,
                    "graph contains a negative weight cycle");
                ndt_free(d);
                ndt_free(p);
                return -1;
            }
        }
    }

    ndt_free(d);

    /* Push return value (possibly xnd_error) onto the stack. */
    stack[2] = mk_return_array(p, N, single_source, ctx);

    ndt_free(p);

    return stack[2].ptr == NULL ? -1 : 0;
}


static const ndt_methods_t graph_methods = {
  .init = NULL,
  .constraint = graph_constraint,
  .repr = NULL
};

static const gm_typedef_init_t typedefs[] = {
  { .name = "node", .type = "int32", .meth=NULL, },
  { .name = "cost", .type = "float64", .meth=NULL },
  { .name = "graph", .type = "var * var * (node, cost)", .meth=&graph_methods },
  { .name = NULL, .type = NULL, .meth=NULL }
};

static const gm_kernel_init_t kernels[] = {
  { .name = "single_source_shortest_paths",
    .sig = "graph, node -> var * var * node",
    .Xnd = shortest_path },

  { .name = NULL, .sig = NULL }
};


/****************************************************************************/
/*                       Initialize kernel table                            */
/****************************************************************************/

int
gm_init_graph_kernels(gm_tbl_t *tbl, ndt_context_t *ctx)
{
    const gm_typedef_init_t *t;
    const gm_kernel_init_t *k;

    for (t = typedefs; t->name != NULL; t++) {
        if (ndt_typedef_from_string(t->name, t->type, t->meth, ctx) < 0) {
            return -1;
        }
    }

    for (k = kernels; k->name != NULL; k++) {
        if (gm_add_kernel(tbl, k, ctx) < 0) {
            return -1;
        }
    }

    return 0;
}
