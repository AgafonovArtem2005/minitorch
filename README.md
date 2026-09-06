# minitorch
Implemented Modules 0-3. Added jupeter notebook to run tests with gpu on Google Colab.

# Task 1.5: Training
### Simple
Epoch  10  loss  2.2454095061862045 correct 50 \
Epoch  20  loss  1.7582743454068683 correct 50 \
Epoch  30  loss  1.5054874378982193 correct 50 \
Epoch  40  loss  1.3111739989882958 correct 50 \
Epoch  50  loss  1.1935882176382473 correct 50 \
Average epoch time: 9.974323 seconds \
CPU times: user 914 ms, sys: 136 ms, total: 1.05 s \
Wall time: 8min 19s

### Split
Epoch  10  loss  262.35482967398156 correct 31 \
Epoch  20  loss  260.28426825164405 correct 31 \
Epoch  30  loss  58.938720650680665 correct 26 \
Epoch  40  loss  52.8927714793367 correct 24 \
Epoch  50  loss  154.64510309566512 correct 31 \
Average epoch time: 9.718389 seconds \
CPU times: user 875 ms, sys: 132 ms, total: 1.01 s \
Wall time: 8min 6s

### Xor
Epoch  10  loss  308.3779733208323 correct 25 \
Epoch  20  loss  37.889154941496564 correct 37 \
Epoch  30  loss  99.5455661795283 correct 33 \
Epoch  40  loss  82.00403384500638 correct 32 \
Epoch  50  loss  3.134994922165941 correct 50 \
Average epoch time: 9.693106 seconds \
CPU times: user 883 ms, sys: 119 ms, total: 1 s \
Wall time: 8min 5s

# Task 2.5: Training
### Simple
Epoch  10  loss  6.340203198495972 correct 48 \
Epoch  20  loss  2.5045299437124973 correct 50 \
Epoch  30  loss  2.080864041763703 correct 50 \
Epoch  40  loss  1.9357366034044197 correct 50 \
Epoch  50  loss  1.8367295508791 correct 50 \
Average epoch time: 137.995708 seconds \
CPU times: user 14.6 s, sys: 1.74 s, total: 16.3 s \
Wall time: 1h 55min 3s

### Split
Epoch  10  loss  317.0517359363456 correct 27 \
Epoch  20  loss  176.71513936526293 correct 23 \
Epoch  30  loss  47.58012209410987 correct 35 \
Epoch  40  loss  144.51784174913783 correct 27 \
Epoch  50  loss  48.69949931378496 correct 32 \
Average epoch time: 121.131763 seconds \
CPU times: user 12.8 s, sys: 1.93 s, total: 14.8 s \
Wall time: 1h 40min 58s

### Xor
Epoch  10  loss  192.6083739031757 correct 29 \
Epoch  20  loss  276.1490689258452 correct 30 \
Epoch  30  loss  169.9106798680696 correct 31 \
Epoch  40  loss  276.16197269057557 correct 30 \
Epoch  50  loss  133.8561177944526 correct 37 \
Average epoch time: 120.925255 seconds \
CPU times: user 12.9 s, sys: 1.97 s, total: 14.8 s \
Wall time: 1h 40min 48s


# Task 3.2: Matrix Multiplication

```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, /Users/art
emagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (154)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/artemagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (154) 
----------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                               | 
        out: Storage,                                                                                                       | 
        out_shape: Shape,                                                                                                   | 
        out_strides: Strides,                                                                                               | 
        in_storage: Storage,                                                                                                | 
        in_shape: Shape,                                                                                                    | 
        in_strides: Strides,                                                                                                | 
    ) -> None:                                                                                                              | 
        if len(out_strides) == len(in_strides) and (out_strides == in_strides).all() and out.size == in_storage.size:-------| #1
            for i in prange(out.size):--------------------------------------------------------------------------------------| #8
                out[i] = fn(in_storage[i])                                                                                  | 
        else:                                                                                                               | 
            for out_ordinal in prange(out.size):----------------------------------------------------------------------------| #11
                out_index = np.empty_like(out_shape)                                                                        | 
                to_index(out_ordinal, out_shape, out_index)                                                                 | 
                in_index = np.empty_like(in_shape)                                                                          | 
                broadcast_index(out_index, out_shape, in_shape, in_index)                                                   | 
                out[index_to_position(out_index, out_strides)] = fn(in_storage[index_to_position(in_index, in_strides)])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--3 has the following loops fused into it:
   +--4 (fused)
   +--0 (fused)
+--6 has the following loops fused into it:
   +--10 (fused)
+--7 has the following loops fused into it:
   +--9 (fused)
Following the attempted fusion of parallel for-loops there are 8 parallel for-
loop(s) (originating from loops labelled: #1, #8, #11, #2, #3, #5, #6, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--11 is a parallel loop
   +--2 --> rewritten as a serial loop
   +--3 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
   +--7 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--7 (parallel)
+--9 (parallel)


Parallel region 1:
+--3 (parallel)
+--0 (parallel)
+--4 (parallel)


Parallel region 2:
+--11 (parallel)
   +--2 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--0 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--10 (parallel)
   +--7 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--7 (parallel, fused with loop(s): 9)


Parallel region 1:
+--3 (parallel, fused with loop(s): 0, 4)


Parallel region 2:
+--11 (parallel)
   +--2 (serial)
   +--3 (serial, fused with loop(s): 0, 4)
   +--5 (serial)
   +--6 (serial, fused with loop(s): 10)
   +--7 (serial, fused with loop(s): 9)


 
Parallel region 0 (loop #7) had 1 loop(s) fused.
 
Parallel region 1 (loop #3) had 2 loop(s) fused.
 
Parallel region 2 (loop #11) had 4 loop(s) fused and 5 loop(s) serialized as 
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (61) is hoisted out 
of the parallel loop labelled #11 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[:] = np.mod(ordinal // 
np.cumprod(np.concatenate((np.ones(1), shape[:0:-1])))[::-1], shape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (83) is hoisted out 
of the parallel loop labelled #11 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[...] = big_index[len(big_shape) - len(shape):] % shape
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, /Users/art
emagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (198)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/artemagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (198) 
------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                           | 
        out: Storage,                                                                                                   | 
        out_shape: Shape,                                                                                               | 
        out_strides: Strides,                                                                                           | 
        a_storage: Storage,                                                                                             | 
        a_shape: Shape,                                                                                                 | 
        a_strides: Strides,                                                                                             | 
        b_storage: Storage,                                                                                             | 
        b_shape: Shape,                                                                                                 | 
        b_strides: Strides,                                                                                             | 
    ) -> None:                                                                                                          | 
        if len(out_strides) == len(a_strides) and len(out_strides) == len(b_strides) and \                              | 
            (out_strides == a_strides).all() and (out_strides == b_strides).all() and \---------------------------------| #13, 14
                out.size == a_storage.size and out.size == b_storage.size:                                              | 
            for i in prange(out.size):----------------------------------------------------------------------------------| #23
                out[i] = fn(a_storage[i], b_storage[i])                                                                 | 
        else:                                                                                                           | 
            for out_ordinal in prange(out.size):------------------------------------------------------------------------| #27
                out_index = np.empty_like(out_shape)                                                                    | 
                to_index(out_ordinal, out_shape, out_index)                                                             | 
                a_index = np.empty_like(a_shape)                                                                        | 
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                 | 
                b_index = np.empty_like(b_shape)                                                                        | 
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                 | 
                a_ordinal = index_to_position(a_index, a_strides)                                                       | 
                b_ordinal = index_to_position(b_index, b_strides)                                                       | 
                out[index_to_position(out_index, out_strides)] = fn(a_storage[a_ordinal], b_storage[b_ordinal])         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--16 has the following loops fused into it:
   +--17 (fused)
   +--12 (fused)
+--20 has the following loops fused into it:
   +--24 (fused)
+--21 has the following loops fused into it:
   +--25 (fused)
+--22 has the following loops fused into it:
   +--26 (fused)
Following the attempted fusion of parallel for-loops there are 11 parallel for-
loop(s) (originating from loops labelled: #13, #14, #23, #27, #15, #16, #18, 
#19, #20, #21, #22).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--27 is a parallel loop
   +--15 --> rewritten as a serial loop
   +--16 --> rewritten as a serial loop
   +--18 --> rewritten as a serial loop
   +--19 --> rewritten as a serial loop
   +--20 --> rewritten as a serial loop
   +--21 --> rewritten as a serial loop
   +--22 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--22 (parallel)
+--26 (parallel)


Parallel region 1:
+--16 (parallel)
+--12 (parallel)
+--17 (parallel)


Parallel region 2:
+--27 (parallel)
   +--15 (parallel)
   +--16 (parallel)
   +--17 (parallel)
   +--12 (parallel)
   +--18 (parallel)
   +--19 (parallel)
   +--20 (parallel)
   +--24 (parallel)
   +--21 (parallel)
   +--25 (parallel)
   +--22 (parallel)
   +--26 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--22 (parallel, fused with loop(s): 26)


Parallel region 1:
+--16 (parallel, fused with loop(s): 12, 17)


Parallel region 2:
+--27 (parallel)
   +--15 (serial)
   +--16 (serial, fused with loop(s): 12, 17)
   +--18 (serial)
   +--19 (serial)
   +--20 (serial, fused with loop(s): 24)
   +--21 (serial, fused with loop(s): 25)
   +--22 (serial, fused with loop(s): 26)


 
Parallel region 0 (loop #22) had 1 loop(s) fused.
 
Parallel region 1 (loop #16) had 2 loop(s) fused.
 
Parallel region 2 (loop #27) had 5 loop(s) fused and 7 loop(s) serialized as 
part of the larger parallel loop (#27).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (61) is hoisted out 
of the parallel loop labelled #27 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[:] = np.mod(ordinal // 
np.cumprod(np.concatenate((np.ones(1), shape[:0:-1])))[::-1], shape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (83) is hoisted out 
of the parallel loop labelled #27 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[...] = big_index[len(big_shape) - len(shape):] % shape
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (83) is hoisted out 
of the parallel loop labelled #27 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[...] = big_index[len(big_shape) - len(shape):] % shape
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, /Use
rs/artemagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py 
(248)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/artemagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (248) 
-----------------------------------------------------------------------|loop #ID
    def _reduce(                                                       | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        reduce_dim: int,                                               | 
    ) -> None:                                                         | 
        for out_ordinal in prange(out.size):---------------------------| #36
            out_index = np.empty_like(out_shape)                       | 
            to_index(out_ordinal, out_shape, out_index)                | 
            a_ordinal = index_to_position(out_index, a_strides)        | 
            result = a_storage[a_ordinal]                              | 
            for _ in range(1, a_shape[reduce_dim]):                    | 
                a_ordinal += a_strides[reduce_dim]                     | 
                result = fn(result, a_storage[a_ordinal])              | 
            out[index_to_position(out_index, out_strides)] = result    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #36, #29, #30, #31, #28, #32, #34).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--36 is a parallel loop
   +--32 --> rewritten as a serial loop
   +--33 --> rewritten as a serial loop
   +--34 --> rewritten as a serial loop
   +--35 --> rewritten as a serial loop
   +--28 --> rewritten as a serial loop
   +--29 --> rewritten as a serial loop
   +--30 --> rewritten as a serial loop
   +--31 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--36 (parallel)
   +--32 (parallel)
   +--33 (parallel)
   +--34 (parallel)
   +--35 (parallel)
   +--28 (parallel)
   +--29 (parallel)
   +--30 (parallel)
   +--31 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--36 (parallel)
   +--32 (serial)
   +--33 (serial)
   +--34 (serial)
   +--35 (serial)
   +--28 (serial)
   +--29 (serial)
   +--30 (serial)
   +--31 (serial)


 
Parallel region 0 (loop #36) had 0 loop(s) fused and 8 loop(s) serialized as 
part of the larger parallel loop (#36).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (45) is hoisted out 
of the parallel loop labelled #36 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: return np.sum(index * strides)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (61) is hoisted out 
of the parallel loop labelled #36 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[:] = np.mod(ordinal // 
np.cumprod(np.concatenate((np.ones(1), shape[:0:-1])))[::-1], shape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (61) is hoisted out 
of the parallel loop labelled #36 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[:] = np.mod(ordinal // 
np.cumprod(np.concatenate((np.ones(1), shape[:0:-1])))[::-1], shape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (61) is hoisted out 
of the parallel loop labelled #36 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index[:] = np.mod(ordinal // 
np.cumprod(np.concatenate((np.ones(1), shape[:0:-1])))[::-1], shape)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /Users/artemagafonov/Docum
ents/deep_learning_2/hw1/minitorch/minitorch/tensor_data.py (45) is hoisted out 
of the parallel loop labelled #36 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: return np.sum(index * strides)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, /Users/arte
magafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (270)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/artemagafonov/Documents/deep_learning_2/hw1/minitorch/minitorch/fast_ops.py (270) 
------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                              | 
    out: Storage,                                                                                                                         | 
    out_shape: Shape,                                                                                                                     | 
    out_strides: Strides,                                                                                                                 | 
    a_storage: Storage,                                                                                                                   | 
    a_shape: Shape,                                                                                                                       | 
    a_strides: Strides,                                                                                                                   | 
    b_storage: Storage,                                                                                                                   | 
    b_shape: Shape,                                                                                                                       | 
    b_strides: Strides,                                                                                                                   | 
) -> None:                                                                                                                                | 
    """                                                                                                                                   | 
    NUMBA tensor matrix multiply function.                                                                                                | 
                                                                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                                                                           | 
                                                                                                                                          | 
    ```                                                                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                                                                     | 
    ```                                                                                                                                   | 
                                                                                                                                          | 
    Optimizations:                                                                                                                        | 
                                                                                                                                          | 
    * Outer loop in parallel                                                                                                              | 
    * No index buffers or function calls                                                                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                                                                | 
                                                                                                                                          | 
                                                                                                                                          | 
    Args:                                                                                                                                 | 
        out (Storage): storage for `out` tensor                                                                                           | 
        out_shape (Shape): shape for `out` tensor                                                                                         | 
        out_strides (Strides): strides for `out` tensor                                                                                   | 
        a_storage (Storage): storage for `a` tensor                                                                                       | 
        a_shape (Shape): shape for `a` tensor                                                                                             | 
        a_strides (Strides): strides for `a` tensor                                                                                       | 
        b_storage (Storage): storage for `b` tensor                                                                                       | 
        b_shape (Shape): shape for `b` tensor                                                                                             | 
        b_strides (Strides): strides for `b` tensor                                                                                       | 
                                                                                                                                          | 
    Returns:                                                                                                                              | 
        None : Fills in `out`                                                                                                             | 
    """                                                                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                                | 
                                                                                                                                          | 
    for out_ordinal in prange(out.size):--------------------------------------------------------------------------------------------------| #37
        n, i, j = out_ordinal // (out_shape[1] * out_shape[2]), out_ordinal // out_shape[2] % out_shape[1], out_ordinal % out_shape[2]    | 
        a_ordinal = n * a_batch_stride + i * a_strides[1]                                                                                 | 
        b_ordinal = n * b_batch_stride + j * b_strides[2]                                                                                 | 
        result = 0.0                                                                                                                      | 
        for _ in range(a_shape[2]):                                                                                                       | 
            result += a_storage[a_ordinal] * b_storage[b_ordinal]                                                                         | 
            a_ordinal += a_strides[2]                                                                                                     | 
            b_ordinal += b_strides[1]                                                                                                     | 
        out[n * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = result                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #37).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Task 3.5: Training
## CPU
### Simple
Epoch  10  loss  4.294131351529167 correct 47 \
Epoch  20  loss  1.435285677559112 correct 49 \
Epoch  30  loss  1.677683616214031 correct 49 \
Epoch  40  loss  1.5983026154090172 correct 48 \
Epoch  50  loss  0.4879018478583375 correct 48 \
Average epoch time: 1.363375 seconds \
CPU times: user 175 ms, sys: 28 ms, total: 203 ms \
Wall time: 1min 25s

### Split
Epoch  10  loss  5.8678667507516264 correct 32 \
Epoch  20  loss  5.110819444914359 correct 41 \
Epoch  30  loss  4.07376433702538 correct 47 \
Epoch  40  loss  4.023892865370894 correct 48 \
Epoch  50  loss  3.390791752233194 correct 49 \
Average epoch time: 1.350757 seconds \
CPU times: user 173 ms, sys: 27.6 ms, total: 201 ms \
Wall time: 1min 22s

### Xor
Epoch  10  loss  5.712422261472123 correct 36 \
Epoch  20  loss  6.171774532713402 correct 34 \
Epoch  30  loss  3.7808769117132237 correct 45 \
Epoch  40  loss  4.8634245090903025 correct 44 \
Epoch  50  loss  3.968042600091156 correct 46 \
Average epoch time: 1.378270 seconds \
CPU times: user 173 ms, sys: 29.7 ms, total: 203 ms \
Wall time: 1min 23s

## GPU
### Simple
Epoch  10  loss  2.387255555833306 correct 48 \
Epoch  20  loss  2.5094731132893404 correct 49 \
Epoch  30  loss  1.6991721011678895 correct 49 \
Epoch  40  loss  0.4988444825054913 correct 49 \
Epoch  50  loss  0.5217051304753264 correct 49 \
Average epoch time: 4.211172 seconds \
CPU times: user 430 ms, sys: 59.7 ms, total: 490 ms \
Wall time: 3min 35s

### Split
Epoch  10  loss  5.7173833377411025 correct 41 \
Epoch  20  loss  4.2841901824169994 correct 43 \
Epoch  30  loss  6.1351959527273525 correct 47 \
Epoch  40  loss  3.915415439011754 correct 41 \
Epoch  50  loss  3.430651859704271 correct 46 \
Average epoch time: 4.120596 seconds \
CPU times: user 412 ms, sys: 61.1 ms, total: 473 ms \
Wall time: 3min 29s

### Xor
Epoch  10  loss  3.9957597652121235 correct 39 \
Epoch  20  loss  2.4388540211834684 correct 47 \
Epoch  30  loss  2.5612751010066432 correct 47 \
Epoch  40  loss  1.1885829107957069 correct 49 \
Epoch  50  loss  2.769510717300415 correct 49 \
Average epoch time: 3.022956 seconds \
CPU times: user 322 ms, sys: 51 ms, total: 373 ms \
Wall time: 2min 36s
