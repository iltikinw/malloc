/**
 * @file mymalloc.c
 * @brief A 64-bit struct-based segregated free list dynamic memory allocator implementation
 *
 * A dynamic memory allocator maintains the heap as a collection of blocks
 * that are either allocated or free, and have variable size.
 *
 * My implementation represents the heap as an implicit list of blocks.
 * Structure of each block:
 *
 *     Each block contains:
 *
 *     - A header with metadata: size, alloc, prev_alloc.
 *         size:       size of the block in memory, 16-byte aligned.
 *         alloc:      allocation status of the block.
 *         prev_alloc: allocation status of the previous block. Used for
 *                     coalescing.
 *
 *     - A union with data based on allocation status:
 *       If block is free: Two pointers to the previous and next blocks in
 *                         the corresponding segregated free list.
 *                         A footer with the same metadata as the header.
 *       If block is allocated: A block payload with allocated data.
 *
 * Along with the implicit list, I keep track of free blocks in a segregated
 * free list. An array of 16 indices manages 16 segregated free lists of
 * increasing size classes.
 *
 * The set of size classes/flist_ind is:
 *     {[0, 32], [33, 64], [65, 128], ... , [262145, MAX]}
 *
 * Segregated free lists are doubly-linked and blocks are inserted based on
 * the LIFO principle.
 *
 * Descriptions of individual functions, data structures, and global variables
 * are provided in their respective leading comments.
 *
 * @author Iltikin Wayet
 */

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "memlib.h"
#include "mm.h"

#ifdef DRIVER
/* create aliases for driver tests */
#define malloc mm_malloc
#define free mm_free
#define realloc mm_realloc
#define calloc mm_calloc
#define memset mem_memset
#define memcpy mem_memcpy
#endif /* def DRIVER */

/*
 *****************************************************************************
 * If DEBUG is defined (such as when running mdriver-dbg), these macros      *
 * are enabled. You can use them to print debugging output and to check      *
 * contracts only in debug mode.                                             *
 *                                                                           *
 * Only debugging macros with names beginning "dbg_" are allowed.            *
 * You may not define any other macros having arguments.                     *
 *****************************************************************************
 */
#ifdef DEBUG
/* When DEBUG is defined, these form aliases to useful functions */
#define dbg_requires(expr) assert(expr)
#define dbg_assert(expr) assert(expr)
#define dbg_ensures(expr) assert(expr)
#define dbg_printf(...) ((void)printf(__VA_ARGS__))
#define dbg_printheap(...) print_heap(__VA_ARGS__)
#else
/* When DEBUG is not defined, these should emit no code whatsoever,
 * not even from evaluation of argument expressions.  However,
 * argument expressions should still be syntax-checked and should
 * count as uses of any variables involved.  This used to use a
 * straightforward hack involving sizeof(), but that can sometimes
 * provoke warnings about misuse of sizeof().  I _hope_ that this
 * newer, less straightforward hack will be more robust.
 * Hat tip to Stack Overflow poster chqrlie (see
 * https://stackoverflow.com/questions/72647780).
 */
#define dbg_discard_expr_(...) ((void)((0) && printf(__VA_ARGS__)))
#define dbg_requires(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_assert(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_ensures(expr) dbg_discard_expr_("%d", !(expr))
#define dbg_printf(...) dbg_discard_expr_(__VA_ARGS__)
#define dbg_printheap(...) ((void)((0) && print_heap(__VA_ARGS__)))
#endif

/* Basic constants */

typedef uint64_t word_t;

/** @brief Word and header size (bytes) */
static const size_t wsize = sizeof(word_t);

/** @brief Double word size (bytes) */
static const size_t dsize = 2 * wsize;

/** @brief Minimum block size (bytes) */
static const size_t min_block_size = dsize;

/**
 * @brief Default minimum size by which we expand the heap.
 * (Must be divisible by dsize)
 */
static const size_t chunksize = (1 << 10);

/**
 * @brief: Bit mask for block allocation status.
 */
static const word_t alloc_mask = 0x1;

/**
 * @brief: Bit mask for previous block allocation status.
 *     1 if previous block allocated, 0 if not.
 */
static const word_t prev_alloc_mask = 0x2;

/**
 * @brief: Bit mask for previous block size.
 *     1 if previous block smallest size possible, 0 if not.
 */
static const word_t prev_small_mask = 0x4;

/**
 * @brief: Bit mask to extract only size bits of block.
 *     Since 16-byte aligned, LS 4 bits don't matter for size.
 */
static const word_t size_mask = ~(word_t)0xF;

/**
 * @brief Basic unit of heap memory.
 *     Represents the header value and a union with either both free list
 *     pointers or the payload of the block. Footer implicitly included.
 *
 * Two potential states based on allocation status:
 *     Free:      no payload, rather two explicit list pointers.
 *                has both header and footer.
 *     Allocated: payload instead of explicit list pointers.
 *                only has header, no footer.
 */
typedef struct block {
    /**
     * @brief Header contains size + allocation flag
     *     Same format as footer. Not included due to variable payload size.
     */
    word_t header;
    union {
        /** @brief Pointers to previous and next free list items. */
        struct {
            struct block *next;
            struct block *prev;
        };
        /** @brief Pointer to an allocated block payload. */
        char payload[0];
    } info;
} block_t;

/* Global variables */

/** @brief Pointer to first block in the heap. */
static block_t *heap_start = NULL;

/** @brief Number of segregated free lists. */
static const int flist_len = 15;

/** @brief Pointer to first block in the free list. */
static block_t *flist[15];

/*
 * ---------------------------------------------------------------------------
 *                        BEGIN SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/**
 * @brief Returns the maximum of two integers.
 * @param[in] x
 * @param[in] y
 * @return `x` if `x > y`, and `y` otherwise.
 */
static size_t max(size_t x, size_t y) {
    return (x > y) ? x : y;
}

/**
 * @brief Rounds `size` up to next multiple of n
 * @param[in] size
 * @param[in] n
 * @return The size after rounding up
 */
static size_t round_up(size_t size, size_t n) {
    return n * ((size + (n - 1)) / n);
}

/**
 * @brief Packs the `size` and `alloc` of a block into a word suitable for
 *        use as a packed value.
 *
 * Packed values are used for both headers and footers.
 *
 * The allocation status is packed into the lowest bit of the word.
 *
 * @param[in] size The size of the block being represented
 * @param[in] alloc True if the block is allocated
 * @return The packed value
 */
static word_t pack(size_t size, bool alloc, bool prev_alloc, bool prev_small) {
    word_t word = size;
    if (alloc) {
        word |= alloc_mask;
    }
    if (prev_alloc) {
        word |= prev_alloc_mask;
    }
    if (prev_small) {
        word |= prev_small_mask;
    }
    return word;
}

/**
 * @brief Extracts the size represented in a packed word.
 *
 * This function simply clears the lowest 4 bits of the word, as the heap
 * is 16-byte aligned.
 *
 * @param[in] word
 * @return The size of the block represented by the word
 */
static size_t extract_size(word_t word) {
    return (word & size_mask);
}

/**
 * @brief Extracts the size of a block from its header.
 * @param[in] block
 * @return The size of the block
 */
static size_t get_size(block_t *block) {
    return extract_size(block->header);
}

/**
 * @brief Given a payload pointer, returns a pointer to the corresponding
 *        block.
 * @param[in] bp A pointer to a block's payload
 * @return The corresponding block
 */
static block_t *payload_to_header(void *bp) {
    return (block_t *)((char *)bp - offsetof(block_t, info.payload));
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        payload.
 * @param[in] block
 * @return A pointer to the block's payload
 * @pre The block must be a valid block, not a boundary tag.
 */
static void *header_to_payload(block_t *block) {
    dbg_requires(get_size(block) != 0);
    return (void *)(block->info.payload);
}

/**
 * @brief Given a block pointer, returns a pointer to the corresponding
 *        footer.
 * @param[in] block
 * @return A pointer to the block's footer
 * @pre The block must be a valid block, not a boundary tag.
 */
static word_t *header_to_footer(block_t *block) {
    dbg_requires(get_size(block) != 0 &&
                 "Called header_to_footer on the epilogue block");
    return (word_t *)(block->info.payload + get_size(block) - dsize);
}

/**
 * @brief Given a block footer, returns a pointer to the corresponding
 *        header.
 * @param[in] footer A pointer to the block's footer
 * @return A pointer to the start of the block
 * @pre The footer must be the footer of a valid block, not a boundary tag.
 */
static block_t *footer_to_header(word_t *footer) {
    size_t size = extract_size(*footer);
    dbg_assert(size != 0 && "Called footer_to_header on the prologue block");
    return (block_t *)((char *)footer + wsize - size);
}

/**
 * @brief Returns the payload size of a given block.
 *
 * The payload size is equal to the entire block size minus the sizes of the
 * block's header and footer.
 *
 * @param[in] block
 * @return The size of the block's payload
 */
static size_t get_payload_size(block_t *block) {
    size_t asize = get_size(block);
    return asize - wsize;
}

/**
 * @brief Returns the allocation status of a given header value.
 *
 * This is based on the lowest bit of the header value.
 *
 * @param[in] word
 * @return The allocation status correpsonding to the word
 */
static bool extract_alloc(word_t word) {
    return (bool)(word & alloc_mask);
}

/**
 * @brief Returns the allocation status of a block, based on its header.
 * @param[in] block
 * @return The allocation status of the block
 */
static bool get_alloc(block_t *block) {
    return extract_alloc(block->header);
}

/**
 * @brief Returns the allocation status of the previous block, based
 *        on parameter block header.
 * @param[in] block
 * @return The allocation status of the previous block
 */
static bool get_prev_alloc(block_t *block) {
    word_t header = block->header;
    return (bool)(header & prev_alloc_mask);
}

/**
 * @brief Returns whether the previous block is the smallest size
 *        possible.
 * @param[in] block
 * @return true of previous block is smallest size, false if not.
 */
static bool get_prev_small(block_t *block) {
    word_t header = block->header;
    return (bool)(header & prev_small_mask);
}

/**
 * @brief Writes an epilogue header at the given address.
 *
 * The epilogue header has size 0, and is marked as allocated.
 *
 * @param[out] block The location to write the epilogue header
 */
static void write_epilogue(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires((char *)block == (char *)mem_heap_hi() - 7);
    block->header = 0x1;
}

/**
 * @brief Finds the next consecutive block on the heap.
 *
 * This function accesses the next block in the "implicit list" of the heap
 * by adding the size of the block.
 *
 * @param[in] block A block in the heap
 * @return The next consecutive block on the heap
 * @pre The block is not the epilogue
 */
static block_t *find_next(block_t *block) {
    dbg_requires(block != NULL);
    dbg_requires(get_size(block) != 0 &&
                 "Called find_next on the last block in the heap");
    return (block_t *)((char *)block + get_size(block));
}

/**
 * @brief Finds the footer of the previous block on the heap.
 * @param[in] block A block in the heap
 * @return The location of the previous block's footer
 */
static word_t *find_prev_footer(block_t *block) {
    // Compute previous footer position as one word before the header
    return &(block->header) - 1;
}

/**
 * @brief Finds the previous consecutive block on the heap.
 *
 * This is the previous block in the "implicit list" of the heap.
 *
 * If the function is called on the first block in the heap, NULL will be
 * returned, since the first block in the heap has no previous block!
 *
 * The position of the previous block is found by reading the previous
 * block's footer to determine its size, then calculating the start of the
 * previous block based on its size.
 *
 * @param[in] block A block in the heap
 * @return The previous consecutive block in the heap.
 */
static block_t *find_prev(block_t *block) {
    dbg_requires(block != NULL);
    word_t *footerp = find_prev_footer(block);
    return footer_to_header(footerp);
}

/**
 * @brief Returns the previous adjacent smallest block.
 *        Block is size 16 bytes.
 * @param[in] block
 * @return The previous adjacent block with size 16
 */
static block_t *find_prev_small(block_t *block) {
    return (block_t *)((char *)block - dsize);
}

/**
 * @brief Writes a block starting at the given address.
 *
 * This function writes the header only.
 *
 * @param[out] block : The location to begin writing the block header
 * @param[in] size   : The size of the new block
 * @param[in] alloc  : The allocation status of the new block
 * Precondition: block must not be NULL.
 */
static void write_block(block_t *block, size_t size, bool alloc,
                        bool prev_alloc, bool prev_small) {
    dbg_requires(block != NULL);
    block->header = pack(size, alloc, prev_alloc, prev_small);
}

/**
 * @brief Writes the footer of a given block.
 *     Separate function has better throughput.
 *
 * @param[out] block : The location to begin writing the block header
 * @param[in] size   : The size of the new block
 * @param[in] alloc  : The allocation status of the new block
 * Precondition: block must not be NULL.
 */
static void write_block_footer(block_t *block, size_t size, bool alloc,
                               bool prev_alloc, bool prev_small) {
    dbg_requires(block != NULL);

    word_t *footerp = header_to_footer(block);
    *footerp = pack(size, alloc, prev_alloc, prev_small);
}

/**
 * @brief Writes the prev_alloc flag of the next block.
 *
 * @param[in] block      : write the header of <block>'s next block.
 * @param[in] prev_alloc : value to set the prev_alloc bit. True = 1, False = 0
 * Precondition: block must not be NULL.
 */
static void write_next_block(block_t *block, bool prev_alloc, bool prev_small) {
    dbg_requires(block != NULL);
    block_t *next = find_next(block);
    next->header = prev_alloc ? next->header | prev_alloc_mask
                              : next->header & ~(prev_alloc_mask);
    next->header = prev_small ? next->header | prev_small_mask
                              : next->header & ~(prev_small_mask);
}

/*
 * ---------------------------------------------------------------------------
 *                        END SHORT HELPER FUNCTIONS
 * ---------------------------------------------------------------------------
 */

/******** The remaining content below are helper and debug routines ********/

/**
 * @brief Returns index in the array of free lists.
 *     Index is based on the log_2 of <size> minus a constant.
 *     Index corresponds to a segregated free list of given size class.
 * @param[in] size : size of the relevant block.
 * @return index of the block in the array of free lists.
 */
static int get_flist_ind(size_t size) {
    if (size < 17)
        return 0;
    if (size < 33)
        return 1;
    if (size < 65)
        return 2;
    if (size < 129)
        return 3;
    if (size < 257)
        return 4;
    if (size < 513)
        return 5;
    if (size < 1025)
        return 6;
    if (size < 2049)
        return 7;
    if (size < 4097)
        return 8;
    if (size < 8193)
        return 9;
    if (size < 16385)
        return 10;
    if (size < 32769)
        return 11;
    if (size < 65537)
        return 12;
    if (size < 131073)
        return 13;
    else
        return 14;
}

/**
 * @brief Adds the block to its corresponding segregated free list.
 *     First finds the corresponding segregated free list (array index)
 *     Then, updates pointers in the list to insert block.
 *     Block inserted based on LIFO principle.
 *
 * @param[in] block : block to be inserted in segregated free list.
 * Precondition: block must not be NULL.
 */
static void add_freeblock(block_t *block) {
    dbg_requires(block != NULL);

    int flist_ind = get_flist_ind(get_size(block));
    size_t size = get_size(block);

    // If free list has no blocks
    if (flist[flist_ind] == NULL) {
        flist[flist_ind] = block;
        // Update below iff size > smallest possible size.
        if (size != min_block_size) {
            block->info.prev = NULL;
        }
        block->info.next = NULL;
    }
    // If free list has at least one valid block
    else {
        // Update below iff size > smallest possible size.
        if (size != min_block_size) {
            flist[flist_ind]->info.prev = block;
            block->info.prev = NULL;
        }
        block->info.next = flist[flist_ind];
        flist[flist_ind] = block;
    }
}

/**
 * @brief Removes the block from its corresponding segregated free list.
 *     First finds the corresponding segregated free list (array index)
 *     Then, updates pointers in the list to remove block.
 *     Block removed based on LIFO principle and 4 cases outlined below.
 *
 * @param[in] block : block to be removed from segregated free list.
 * Precondition: block must not be NULL.
 */
static void rem_freeblock(block_t *block) {
    dbg_requires(block != NULL);

    int flist_ind = get_flist_ind(get_size(block));

    block_t *prev = NULL;
    block_t *next = block->info.next;

    if (get_size(block) != min_block_size) {
        prev = block->info.prev;
        // Block is first in the list.
        if (next != NULL && prev == NULL) {
            next->info.prev = NULL;
            flist[flist_ind] = next;
        }
        // Block is in the middle of the list.
        else if (next != NULL && prev != NULL) {
            next->info.prev = prev;
            prev->info.next = next;
        }
        // Block is the only node in the list.
        else if (next == NULL && prev == NULL) {
            flist[flist_ind] = next;
        }
        // Block is last in the list.
        else if (next == NULL && prev != NULL) {
            prev->info.next = next;
        }
        block->info.prev = NULL;
        block->info.next = NULL;
    } else {
        // Finding previous item in free list.
        block_t *curr = flist[flist_ind];
        while (curr != NULL) {
            if (curr->info.next == block)
                prev = curr;
            curr = curr->info.next;
        }
        if (prev == NULL)
            flist[flist_ind] = next;
        else
            prev->info.next = next;
    }
}

/**
 * @brief Coalesces block with adjacent free blocks.
 *     If no adjacent free blocks, adds block to segregated free list.
 *     Coalesces based on four cases outlined in code below.
 *     With each coalesce, updates next block's header to accurate prev_alloc
 * value.
 *
 * @param[in] block : block to be coalesced with adjacent free blocks.
 * Precondition: block must not be equal to NULL.
 *
 * @return pointer to newly-coalesced block
 */
static block_t *coalesce_block(block_t *block) {
    dbg_requires(block != NULL);

    size_t size = get_size(block);

    block_t *prev = NULL;
    block_t *next = find_next(block);

    bool prev_alloc = get_prev_alloc(block);
    bool prev_small = get_prev_small(block);
    bool next_alloc = get_alloc(next);

    // Case 2: Next block also free
    if (prev_alloc && !next_alloc) {
        rem_freeblock(next);
        size += get_size(next);

        write_block(block, size, false, true, prev_small);
        write_block_footer(block, size, false, true, prev_small);
    }
    // Case 3: Previous block also free
    else if (!prev_alloc && next_alloc) {
        prev = prev_small ? find_prev_small(block) : find_prev(block);
        rem_freeblock(prev);
        size += get_size(prev);

        write_block(prev, size, false, true, prev_small);
        write_block_footer(prev, size, false, true, prev_small);
        block = prev;
    }
    // Case 4: Both next and previous blocks also free
    else if (!prev_alloc && !next_alloc) {
        prev = prev_small ? find_prev_small(block) : find_prev(block);
        rem_freeblock(prev);
        rem_freeblock(next);
        size += (get_size(prev) + get_size(next));

        write_block(prev, size, false, true, prev_small);
        write_block_footer(prev, size, false, true, prev_small);
        block = prev;
    }
    // Case 1: Default case.
    bool curr_small = size == min_block_size;
    write_next_block(block, false, curr_small);
    add_freeblock(block);

    return block;
}

/**
 * @brief Extends size of the heap in memory.
 *     Steps of extension outlined in code below.
 *     Coalesces block at the end.
 *
 * @param[in] size : size by which we extend the size of the heap.
 *     Number rounded up to follow 16-byte alignment convention.
 *     If size < chunksize, we extend by chunksize = 4096 byte instead.
 *
 * @return pointer to free block inserted at the end of the heap.
 */
static block_t *extend_heap(size_t size) {
    void *bp;

    // Allocate an even number of words to maintain alignment
    size = round_up(size, dsize);
    if ((bp = mem_sbrk((intptr_t)size)) == (void *)-1) {
        return NULL;
    }

    // Initialize free block header/footer
    block_t *block = payload_to_header(bp);
    bool prev_alloc = get_prev_alloc(block);
    bool prev_small = get_prev_small(block);

    write_block(block, size, false, prev_alloc, prev_small);
    write_block_footer(block, size, false, prev_alloc, prev_small);

    // Create new epilogue header
    block_t *block_next = find_next(block);
    write_epilogue(block_next);

    // Coalesce in case the previous block was free
    block = coalesce_block(block);
    return block;
}

/**
 * @brief Writes a block in the heap, splitting if too large
 *     If asize is less than block_size - min_block_size, split.
 *     If asize exactly fills the space, don't split.
 *     Splitting entails first writing the block, then adding the
 *     next adjacent free block to the segregated free list.
 *
 * @param[in] block : block in memory to which we write.
 * @param[in] asize : size of the write to memory.
 * Precondition: <block> must not be allocated.
 * Precondition: size of <block> must be greater than or equal to <asize>.
 */
static void split_block(block_t *block, size_t asize) {
    dbg_requires(!get_alloc(block));
    dbg_requires(get_size(block) >= asize);

    rem_freeblock(block);
    bool prev_small = get_prev_small(block);

    size_t block_size = get_size(block);
    if ((block_size - asize) >= min_block_size) {

        bool block_small = asize == min_block_size;
        write_block(block, asize, true, true, prev_small);

        block_t *next = find_next(block);
        size_t next_size = block_size - asize;
        bool next_small = next_size == min_block_size;

        write_block(next, next_size, false, true, block_small);
        write_block_footer(next, next_size, false, true, block_small);
        write_next_block(next, false, next_small);

        coalesce_block(next);
    } else {
        write_block(block, block_size, true, true, prev_small);
        block_t *next = find_next(block);
        write_next_block(block, true, get_prev_small(next));
    }

    dbg_ensures(get_alloc(block));
}

/**
 * @brief Searches through the next 10 free blocks for a better fit.
 *     Starts at the first block with a potential fit.
 *
 * @param[in] curr_ind : initial index within the segregated free list array.
 * @param[in] curr     : block with initial best fit.
 *                          also use this as an iterating value.
 * @param[in] asize    : size we want to allocate.
 * @param[in] target   : size of the initial block with best fit.
 *
 * @return block with better fit or initial block.
 */
static block_t *find_better_fit(int curr_ind, block_t *curr, size_t asize,
                                size_t target) {
    block_t *ret = curr;
    size_t curr_size;
    for (int i = 0; curr_ind < flist_len - 1 && i < 21; i++) {
        // If curr = NULL, then go to the next size class.
        if (curr == NULL) {
            curr_ind++;
            curr = flist[curr_ind];
        }
        // If cur != NULL, test if block is better fit.
        // Keeps going regardless of better fit in case even better fit coming.
        if (curr != NULL) {
            curr_size = get_size(curr);
            if (asize <= curr_size && curr_size < target) {
                target = curr_size;
                ret = curr;
            }
            curr = curr->info.next;
        }
    }
    return ret;
}

/**
 * @brief Searches through segregated free lists for free block to allocate.
 *     Starts at corresponding segregated free list to asize.
 *     Searches subsequent segregated free lists of greater size class
 *     if no match.
 *     Simple first fit search, but with segregated free lists, approximates
 *     to mix of best fit search.
 *
 * @param[in] asize : size which we want to allocate.
 *     Looking for free block >= this size.
 *
 * @return non-NULL : free block found and pointer returned.
 *         NULL     : free block not found and NULL returned.
 */
static block_t *find_fit(size_t asize) {
    int flist_ind = get_flist_ind(asize);
    for (int i = flist_ind; i < flist_len; i++) {
        block_t *curr = flist[i];
        while (curr != NULL) {
            size_t target = get_size(curr);
            if (asize <= target) {
                return find_better_fit(i, curr, asize, target);
            }
            curr = curr->info.next;
        }
    }
    // No space found, return NULL.
    return NULL;
}

/**
 * @brief Checks block invariants:
 *     1. Within heap boundaries
 *     2. Block size >= min_block_size
 *     3. Header and footers consistent
 *     4. Header aligned
 *     5. No two adjacent free blocks present
 *     Specific checks highlighted in comments below.
 *
 * @param[in] block : block to check invariants of
 * @param[in] line  : line at which mm_cheackheap was called.
 * Precondition: block must not be NULL.
 *
 * @return true if invariants hold, false if not.
 */
static bool check_block(block_t *block, int line) {
    dbg_requires(block != NULL);
    // Check block within heap boundaries
    if (!(mem_heap_lo() <= (void *)block && (void *)block <= mem_heap_hi())) {
        fprintf(stderr, "Block %p not within heap bounds! Line: %d\n",
                (void *)&block, line);
        return false;
    }
    // Check block size greater than minimum
    size_t size = get_size(block);
    if (size < min_block_size) {
        fprintf(stderr, "Block %p of size %ld less than %ld size! Line: %d\n",
                (void *)&block, size, min_block_size, line);
        return false;
    }
    // Only check below two if size > min_block_size.
    if (size != min_block_size) {
        // Check block header and footer sizes match
        word_t *footer = header_to_footer(block);
        if (!get_alloc(block) && get_size(block) != extract_size(*footer)) {
            fprintf(stderr, "Block %p header, footer size no match! Line: %d\n",
                    (void *)&block, line);
            return false;
        }
        // Check block header and footer alloc bits match
        if (!get_alloc(block) && get_alloc(block) != extract_alloc(*footer)) {
            fprintf(stderr,
                    "Block %p header, footer alloc no match! Line: %d\n",
                    (void *)&block, line);
            return false;
        }
    }
    // Check block header alignment
    size_t align_actual = (size_t)header_to_payload(block);
    size_t align_expect = round_up(align_actual, (size_t)16);
    if (align_actual != align_expect) {
        fprintf(stderr, "Block %p header not aligned! Line: %d\n",
                (void *)&block, line);
        return false;
    }
    // Check no two adjacent free blocks
    if (!get_alloc(block) && !get_alloc(find_next(block))) {
        fprintf(stderr, "Two adjacent blocks (%p, %p) found! Line: %d\n",
                (void *)block, (void *)find_next(block), line);
        return false;
    }
    return true;
}

/**
 * @brief Checks block invariant:
 *     1. Free list pointers are consistent.
 *     Doubly-linked pointers must be consistent.
 *
 * @param[in] block : block to check invariant of
 * @param[in] line  : line at which mm_checkheap was called.
 * Precondition: block must not be NULL.
 *
 * @return true if invariants hold, false if not.
 */
static bool check_freeblock(block_t *block, int line) {
    dbg_requires(block != NULL);
    block_t *next = block->info.next;
    block_t *prev = block->info.prev;
    if ((next != NULL && next->info.prev != NULL && next->info.prev != block) ||
        (prev != NULL && prev->info.next != NULL && prev->info.next != block)) {
        fprintf(stderr, "Free block next/prev pointer neq! Line: %d\n", line);
        return false;
    }
    return true;
}

/**
 * @brief Verifies heap invariants.
 *     Checks performed:
 *         1. Heap start neq NULL
 *         2. Prologue block exists and aligned
 *         3. Implicit list invariants hold
 *            Iterates through implicit list, checking each block.
 *         4. Epilogue block existence.
 *         5. Segregated free list invariants hold
 *
 * @param[in] line : line at which mm_checkheap called.
 *     Passed to helper functions called.
 *
 * @return true if invariants hold, false if not.
 */
bool mm_checkheap(int line) {
    // Implicit/Explicit List Check
    // Check heap_start
    if (heap_start == NULL) {
        fprintf(stderr, "heap_start equals NULL! Line: %d\n", line);
        return false;
    }
    // Check prologue block existence and alignment
    if ((block_t *)(&heap_start->header - 1) != (block_t *)mem_heap_lo()) {
        fprintf(stderr, "Invalid prologue block! Line: %d\n", line);
        return false;
    }
    // Iterate through implicit list and check each block
    block_t *curr = heap_start;
    while (curr != NULL && get_size(curr) != 0) {
        if (!check_block(curr, line))
            return false;
        curr = find_next(curr);
    }
    // Check epilogue
    if (curr == NULL || !(get_size(curr) == 0 && get_alloc(curr))) {
        fprintf(stderr, "No epilogue block found! Line: %d\n", line);
        return false;
    }
    // Free list Check
    for (int i = 0; i < flist_len; i++) {
        curr = flist[i];
        while (curr != NULL) {
            if (!check_freeblock(curr, line))
                return false;
            curr = curr->info.next;
        }
    }
    return true;
}

/**
 * @brief Initializes all data structures and empty heap.
 * @return true if sucessful, false if memory errors.
 */
bool mm_init(void) {
    // Create the initial empty heap
    word_t *start = (word_t *)(mem_sbrk(2 * wsize));

    if (start == (void *)-1) {
        return false;
    }

    start[0] = pack(0, true, true, false); // Heap prologue (block footer)
    start[1] = pack(0, true, true, false); // Heap epilogue (block header)

    // Heap starts with first "block header", currently the epilogue
    heap_start = (block_t *)&(start[1]);

    // Initialize free list
    for (int i = 0; i < flist_len; i++) {
        flist[i] = NULL;
    }

    // Extend the empty heap with a free block of chunksize bytes
    if (extend_heap(chunksize) == NULL) {
        return false;
    }

    return true;
}

/**
 * @brief Allocate <size> bytes on the heap.
 *     Specific steps outlined in comments in code below.
 *
 * @param[in] size : amount of bytes to be allocated on the heap
 * Precondition: heap invariants must hold.
 * Postcondition: heap invariants still hold.
 *
 * @return pointer to allocated block or NULL if memory errors.
 */
void *malloc(size_t size) {
    dbg_requires(mm_checkheap(__LINE__));

    size_t asize;      // Adjusted block size
    size_t extendsize; // Amount to extend heap if no fit is found
    block_t *block;
    void *bp = NULL;

    // Initialize heap if it isn't initialized
    if (heap_start == NULL) {
        if (!(mm_init())) {
            dbg_printf("Problem initializing heap. Likely due to sbrk\n");
            return NULL;
        }
    }

    // Ignore spurious request
    if (size == 0) {
        dbg_ensures(mm_checkheap(__LINE__));
        return bp;
    }

    // Adjust block size to include overhead and to meet alignment requirements
    asize = round_up(size + wsize, dsize);

    // Search the free list for a fit
    block = find_fit(asize);

    // If no fit is found, request more memory, and then and place the block
    if (block == NULL) {
        // Always request at least chunksize
        extendsize = max(asize, chunksize);
        block = extend_heap(extendsize);
        // extend_heap returns an error
        if (block == NULL) {
            return bp;
        }
    }

    // The block should be marked as free
    dbg_assert(!get_alloc(block));

    // Try to split the block if too large
    split_block(block, asize);

    bp = header_to_payload(block);

    dbg_ensures(mm_checkheap(__LINE__));
    return bp;
}

/**
 * @brief Frees pointer <bp> from the heap previously allocated by
 *     mm_malloc(). If <bp> already freed, does nothing.
 *
 * @param[in] bp : pointer to block of memory to be freed.
 * Precondition: heap invariants must hold.
 * Postcondition: heap invariants still hold.
 */
void free(void *bp) {
    dbg_requires(mm_checkheap(__LINE__));

    if (bp == NULL) {
        return;
    }

    block_t *block = payload_to_header(bp);
    size_t size = get_size(block);
    bool prev_alloc = get_prev_alloc(block);
    bool prev_small = get_prev_small(block);
    bool curr_small = size == min_block_size;

    // The block should be marked as allocated
    dbg_assert(get_alloc(block));

    // Mark the block as free
    write_block(block, size, false, prev_alloc, prev_small);
    if (size != min_block_size)
        write_block_footer(block, size, false, prev_alloc, prev_small);
    write_next_block(block, false, curr_small);

    // Try to coalesce the block with its neighbors
    coalesce_block(block);

    dbg_ensures(mm_checkheap(__LINE__));
}

/**
 * @brief Reallocates pointer <ptr> to new size <size>.
 *     Allocates new memory of <size>, copies data over.
 *     Returns pointer to reallocated block.
 *
 * @param[in] ptr : pointer to be reallocated.
 * @param[in] size : amount of bytes to be reallocated.
 *
 * @return pointer to reallocated block or NULL if memory errors.
 */
void *realloc(void *ptr, size_t size) {
    block_t *block = payload_to_header(ptr);
    size_t copysize;
    void *newptr;

    // If size == 0, then free block and return NULL
    if (size == 0) {
        free(ptr);
        return NULL;
    }

    // If ptr is NULL, then equivalent to malloc
    if (ptr == NULL) {
        return malloc(size);
    }

    // Otherwise, proceed with reallocation
    newptr = malloc(size);

    // If malloc fails, the original block is left untouched
    if (newptr == NULL) {
        return NULL;
    }

    // Copy the old data
    copysize = get_payload_size(block); // gets size of old payload
    if (size < copysize) {
        copysize = size;
    }
    memcpy(newptr, ptr, copysize);

    // Free the old block
    free(ptr);

    return newptr;
}

/**
 * @brief Allocates <elements> * <size> bytes in memory.
 *     Initializes all allocated values to zero.
 *     Returns pointer to allocated block.
 *
 * @param[in] elements : number of elements to be allocated.
 * @param[in] size : size of each element.
 *
 * @return pointer to allocated block or NULL if memory errors.
 */
void *calloc(size_t elements, size_t size) {
    void *bp;
    size_t asize = elements * size;

    if (elements == 0) {
        return NULL;
    }
    if (asize / elements != size) {
        // Multiplication overflowed
        return NULL;
    }

    bp = malloc(asize);
    if (bp == NULL) {
        return NULL;
    }

    // Initialize all bits to 0
    memset(bp, 0, asize);

    return bp;
}

/*
 *****************************************************************************
 * Do not delete the following super-secret(tm) lines!                       *
 *                                                                           *
 * 53 6f 20 79 6f 75 27 72 65 20 74 72 79 69 6e 67 20 74 6f 20               *
 *                                                                           *
 * 66 69 67 75 72 65 20 6f 75 74 20 77 68 61 74 20 74 68 65 20               *
 * 68 65 78 61 64 65 63 69 6d 61 6c 20 64 69 67 69 74 73 20 64               *
 * 6f 2e 2e 2e 20 68 61 68 61 68 61 21 20 41 53 43 49 49 20 69               *
 *                                                                           *
 * 73 6e 27 74 20 74 68 65 20 72 69 67 68 74 20 65 6e 63 6f 64               *
 * 69 6e 67 21 20 4e 69 63 65 20 74 72 79 2c 20 74 68 6f 75 67               *
 * 68 21 20 2d 44 72 2e 20 45 76 69 6c 0a c5 7c fc 80 6e 57 0a               *
 *                                                                           *
 *****************************************************************************
 */
