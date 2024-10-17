# Architecture of `ruddy`

This document outlines the inner workings of `ruddy` and should be mostly relevant for developers. 

## Standalone vs. Pooled BDDs

In `ruddy`, we differentiate between two "types" of decision diagrams based on how they are stored in memory.
Internally, these are manipulated by the same algorithms, but they provide different trade-offs in terms of 
performance and ease of use.

#### Standalone BDD 

An immutable BDD that fully owns its nodes and is stored in a single continuous array. When 
operating on standalone BDDs, each operation creates a new standalone BDD that is independent of the operands.
One can still access and reference the individual nodes of a standalone BDD, but they cannot be modified. 

Advantages:
 * Minimizes memory usage of a single BDD, since data structures like task cache and node uniqueness table can be
   freed once the BDD is computed. The BDD nodes can be easily dumped to disk, compressed, etc.
 * BDDs are immutable, hence they can be safely copied or even shared across threads. A single BDD can be used as
   an argument in multiple parallel operations.
 * We can optimize the memory layout of frequently used BDDs. We can optimize the variable ordering of BDDs that are
   not used in the same operations.

Disadvantages:
 * Multiple BDDs that share the same sub-graph need to store a copy of it. 
 * Some simple operations that could be implemented in constant time. like negation, require linear time because
   a copy of the BDD must be created.
 * By default, supporting data structures (like task cache or node uniqueness table) need to be re-allocated for
   each operation.

#### Pooled BDD

A BDD whose nodes are stored in a "pool" that is shared by multiple BDDs. When operating on pooled BDDs, 
only BDDs from the same pool can be considered (using arbitrary BDDs is possible, but requires extra effort and
has worse performance).

Advantages:
 * Minimizes memory usage of multiple BDDs that share the same sub-graph. Furthermore, supporting data structures
   (task cache and node uniqueness table) can be reused between operations.
 * Some simple modifications can be implemented as constant-time operations.
 * In some situations, it may actually be cleaner to have one "memory pool" object and pointers into this pool,
   instead of moving memory around with each BDD.

Disadvantages:
 * Requires garbage collection to remove BDD nodes from the pool that are no longer used.
 * Any support for parallelism or thread sharing has to be built into the pool object directly, which is often not 
   easy to do.
 * If the BDDs *do not* share enough nodes, it can actually lead to worse performance, as disconnected BDDs can
   still cause hash collisions in the supporting data structures.

### Summary

Overall, standalone BDDs are usually a good "first choice" for any symbolic algorithm, because they are relatively
simple and don't require any special "management". If the algorithm uses a lot of BDDs that share the same 
sub-structure, it may be beneficial to switch parts of the algorithm to pooled BDDs, but this should usually be a 
conscious decision based on performance testing.

**Warning:** Our experiments suggest that for complex BDDs, meaningful node sharing between BDDs is actually 
quite rare. In such case, a pooled BDD approach may in fact hurt performance instead of improving it, since
all BDDs have to be always considered *together*, instead of separating them into independent data structures.
Further possible source of a slowdown is the garbage collection: with standalone BDDs, this step is almost
free (de-allocation of a single array), but for pooled BDDs, the whole node storage needs to be traversed and marked
(or extra reference counters need to be maintained).

Where pooled BDDs are typically most useful are algorithms that directly build a collection of functions with the
same shared element, e.g. `a | f`, `b | f`, `c | f`, `d | f`. Here, we have four functions that are different, but 
each contains `f` as a "sub-element". However, even this pattern does not *guarantee* meaningful node sharing.

### Design considerations

Standalone BDDs and pooled BDDs can be built using the same underlying data structures. The main difference is in how
to access these structures. Pooled BDDs are more complex because the node storage has to support node deletion, but
this does not necessarily impact how the data is stored in memory, it only adds an extra algorithm that needs to be
implemented on top of the existing data structure.

## Pointer compression

Somewhat counterintuitively, pointer compression does not actually involve any sophisticated compression algorithms.
The whole point is that for data structures that don't need the full address space, we can use smaller pointers
(or array indices) to save memory (and memory bandwidth). Since BDDs mostly consist of pointers (or array indices),
pointer compression is highly effective here.

For example, a BDD node triple `(var, low, high)` that uses 16-bit indices is stored using 6 bytes, 32-bit indices
require 12 bytes, and 64-bit indices require 24 bytes. Furthermore, one could argue that supporting 2^64 variables
is unnecessary and 2^32 is sufficient. Similarly, 2^64 BDD nodes is unrealistic with current hardware for the 
foreseeable future, meaning we could limit the indices to 48 bits (281 "tera-nodes"). With this optimization, one node
only requires 16 bytes (2/3 of the "full" 64-bit version). However, do note that 48-bit data types are not natively 
supported by CPUs, so some overhead may be introduced when actually loading/storing such data in RAM (we always
have to read either 32, 64, or 128 bits, and then we need to split/stitch those values into three variables, having 
32, 48, and 48 bits respectively).

Overall, pointer compression works as follows:

 * We implement the same BDD algorithms multiple times using different pointer widths. Exact widths are up for 
   discussion, but 16, 32 and 48 seem like a reasonable choice. With rust's const-generics, some of this could be
   perhaps also just automated, but not sure how useful that will be (see next point).
 * These algorithms can be the same, but we could also introduce some extra optimizations depending on pointer width.
   For example, we can be fairly certain that any 16-bit BDD will fit into the cache of any relatively modern CPU,
   meaning we don't have to optimize for cache friendliness if we don't want to. Similarly, we know that 48-bit
   BDDs will only be necessary for very large datasets and their performance will depend heavily on memory latency, 
   so we can try to be more aggressive when optimizing memory accesses.
 * Whenever a BDD operation exceeds the address space of the current pointer width, we restart the operation using
   the implementation for larger pointer widths. Whenever a BDD is significantly smaller than it's pointer width 
   allows, we convert it to smaller pointer width (similar to how dynamic arrays are expanded/shrunk based on the
   available capacity).
 * In theory, when the limit is reached, we could just "stop" the operation, extend all data structures "in place" and
   then continue using the code for different pointer width. However, this is a bit more involved, since we need to
   implement the conversions.
 * Also, this is a bit more complicated for standalone BDDs, since we need to consider every combination of pointer
   widths, while for a pooled BDD, we generally assume the BDDs come from the same pool, and we only need to 
   grow/shrink the whole pool.

## SIMD

Most CPUs nowadays support instructions that allow working with more than 64 bits at a time (128, 256 and 512 are 
generally supported, although 512 support is still relatively rare outside of servers). This is probably not super
relevant for the main logic of the BDD apply algorithm, but it could be a useful fact for working with hash tables.
In reality, the CPU is always pulling 64B (512 bits) of data into cache at a time, even if only one byte is needed.
Using the remaining bits that are "already there" for something could be interesting. Another aspect to consider
is that even if the CPU has to wait for the data, grouping it together into fewer load instructions could be 
beneficial if the bottleneck is the size of the load queue.

A possible way to test this without using SIMD is to focus on the 16-bit variant of BDDs when using pointer 
compression. In such case, a single BDD node (and other datastructures related to the hash tables) easily fits 
into a 64-bit integer. So we could try to test a version of this implementation where certain operations are performed
directly on the "bitvector" representation of the data.

A disadvantage of this approach is that the CPU needs to have these instructions. For x86 CPUs, 256-bit instructions
are present on virtually everything that was sold in the last ~10 years. But on ARM/Apple Silicon, this isn't really
the case. For now, I would say we should focus on the x86 implementation and for 16-bit and 32-bit BDDs, and we'll see
if it actually helps or not.

## Tuning of the cache skip condition

Currently, tasks skip the cache if both nodes have only one parent. This covers ~half of the cases where the cache 
could actually be skipped. We would want to test some heuristics that could give a higher percentage on average while 
not increasing the worst case runtime too much (skipping the cache in the wrong situation can slow down the
algorithm because the partial result is not available once it is needed again). It's not necessary that the condition
is 100% accurate (such condition should not exist), but some more randomized approach with better average success 
would be cool.

## Optimizing node ordering

For the BDD apply algorithm, BDD node ordering can be an important factor. BDD nodes are generated in DFS-post-order, 
but visited in DFS-pre-order. Of course, sorting a BDD is a non-trivial operation, and we probably don't want to do
it for every BDD. But maybe for very larger BDDs, or BDDs that are used very often, this could be a viable strategy
to improve performance. 

## Parallelism

In the future, we could consider making the BDD apply algorithm parallel. It should be relatively easy to make the core
data structures lock-free using atomics, we just have to figure out the synchronization logic between threads.
