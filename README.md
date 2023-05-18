# mymalloc.c - A segregated free list dynamic memory allocator implementation written in C.
Written as part of CMU's Computer Systems Course. Received a 100% grade.
## Info on dynamic memory allocators
A dynamic memory allocator maintains the heap in memory as a collection of blocks that are either allocated or free, and have variable size.
## How my implementation works
My implementation represents the heap as an implicit list of blocks. The heap is an implicit list in that there are no explicit pointers between blocks within the heap; rather, the size of the block is stored and used to find the next block.
### Each block contains
A header with metadata: ```size```, ```alloc```, ```prev_alloc```.
* ```size```: size of the block in memory, 16-byte aligned.
* ```alloc```: allocation status of the block.
* ```prev_alloc```: allocation status of the previous block. Used for coalescing.

A union with data based on allocation status:
* If block is free: 
    * Two pointers to the previous and next blocks in the corresponding segregated free list.
    * A footer with the same metadata as the header.
* If block is allocated: A block payload with allocated data.
### Segregated free lists
Along with the implicit list, I keep track of free blocks in a segregated free list. An array of 16 indices manages 16 segregated free lists of increasing size classes. The list of size classes/flist_ind is: $[0, 32], (32, 64], (62, 128], \cdots , (262144, \text{INT MAX}]$.

Segregated free lists are doubly-linked and blocks are inserted based on the LIFO principle.
## Demos
The version publicly available in this repository does not work on its own. For demos, please contact me at iltikinw@gmail.com, and I'd love to connect!