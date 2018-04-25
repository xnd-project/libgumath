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


#include <stdio.h>
#include <limits.h>
#include <stddef.h>
#include "gumath.h"


/*****************************************************************************/
/*                                 Charmap                                   */
/*****************************************************************************/

#define ALPHABET_LEN 64

static int code[UCHAR_MAX+1];
static unsigned char alpha[ALPHABET_LEN+1] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.";

static void
init_charmap(void)
{
    int i;

    for (i = 0; i < UCHAR_MAX+1; i++) {
        code[i] = UCHAR_MAX;
    }

    for (i = 0; i < ALPHABET_LEN; i++) {
        code[alpha[i]] = i;
    }
}


/*****************************************************************************/
/*                              Function tables                              */
/*****************************************************************************/

/* Function table */
struct _gm_tbl {
    gm_func_t *value;
    gm_tbl_t *next[];
};

gm_tbl_t *
gm_tbl_new(ndt_context_t *ctx)
{
    gm_tbl_t *t;
    int i;

    t = ndt_alloc_size(offsetof(gm_tbl_t, next) + ALPHABET_LEN * (sizeof t));
    if (t == NULL) {
        return ndt_memory_error(ctx);
    }

    t->value = NULL;

    for (i = 0; i < ALPHABET_LEN; i++) {
        t->next[i] = NULL;
    }

    return t;
}

void
gm_tbl_del(gm_tbl_t *t)
{
    int i;

    if (t == NULL) {
        return;
    }

    gm_func_del(t->value);

    for (i = 0; i < ALPHABET_LEN; i++) {
        gm_tbl_del(t->next[i]);
    }

    ndt_free(t);
}

int
gm_tbl_add(gm_tbl_t *tbl, const char *key, gm_func_t *value, ndt_context_t *ctx)
{
    gm_tbl_t *t = tbl;
    const unsigned char *cp;
    int i;

    for (cp = (const unsigned char *)key; *cp != '\0'; cp++) {
        i = code[*cp];
        if (i == UCHAR_MAX) {
            ndt_err_format(ctx, NDT_ValueError,
                           "invalid character in function name: '%c'", *cp);
            gm_func_del(value);
            return -1;
        }

        if (t->next[i] == NULL) {
            gm_tbl_t *u = gm_tbl_new(ctx);
            if (u == NULL) {
                gm_func_del(value);
                return -1;
            }
            t->next[i] = u;
            t = u;
        }
        else {
            t = t->next[i];
        }
    }

    if (t->value) {
        ndt_err_format(ctx, NDT_ValueError, "duplicate function name '%s'", key);
        gm_func_del(value);
        return -1;
    }

    t->value = value;
    return 0;
}

gm_func_t *
gm_tbl_find(const gm_tbl_t *tbl, const char *key, ndt_context_t *ctx)
{
    const gm_tbl_t *t = tbl;
    const unsigned char *cp;
    int i;

    for (cp = (const unsigned char *)key; *cp != '\0'; cp++) {
        i = code[*cp];
        if (i == UCHAR_MAX) {
            ndt_err_format(ctx, NDT_ValueError,
                           "invalid character in function name: '%c'", *cp);
            return NULL;
        }

        if (t->next[i] == NULL) {
            ndt_err_format(ctx, NDT_ValueError,
                           "cannot find function '%s'", key);
            return NULL;
        }
        t = t->next[i];
    }

    if (t->value == NULL) {
        ndt_err_format(ctx, NDT_RuntimeError,
                       "cannot find function '%s'", key);
        return NULL;
    }

    return t->value;
}

int
gm_tbl_map(const gm_tbl_t *tbl, int (*f)(const gm_func_t *, void *), void *state)
{
    const gm_tbl_t *t = tbl;
    int i;

    if (t->value) {
        if (f(t->value, state) < 0) {
            return -1;
        }
    }

    for (i = 0; i < ALPHABET_LEN; i++) {
        if (t->next[i] && gm_tbl_map(t->next[i], f, state) < 0) {
            return -1;
        }
    }

    return 0;
}


/*****************************************************************************/
/*                           Initialize global values                        */
/*****************************************************************************/

void
gm_init(void)
{
    static bool initialized = false;

    if (!initialized) {
        init_charmap();
    }
    else {
        fprintf(stderr, "gm_init: warning: ignoring attempt to initialize "
                        "libgumath a second time\n");
    }
}
