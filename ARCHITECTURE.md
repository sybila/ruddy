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

