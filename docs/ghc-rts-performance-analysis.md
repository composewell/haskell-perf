# GHC RTS Stats

GHC RTS has a builtin stats reporting mechanism which can be accessed
using `+RTS -s` flag to print the stats when the program has terminated
or if compiled with `-T` flag then we can access it programmatically via
the `getRTSStats` API. It reports the CPU time used by the program and
allocations among other things.

RTS Stats give entire OS process level (not OS thread level) cpu time
and not Haskell thread-wise cpu time. When multiple OS threads are
used, the cpu time reported is the cpu time of all the Haskell threads
combined, and all the OS threads combined.

<!--
Also, the way kernel accounts this time it could be off by a little
(microseconds) because each thread's cpu time is recorded at the last
kernel accounting event. Allocations are recorded by the GHC RTS only at
the GC boundary, so the allocations reported are from the point when the
last GC happened. So we need to be careful when using or interpreting
these stats.
-->

In a multithreaded program using RTS stats we can only tell time how
much total CPU time (and allocations) the entire Haskell process (all
threads) spent between two points, but we cannot tell which Haskell
thread spent how much time or how much time was actually spent by the
instructions between those two points.

## getRTSStats

The getRTSStats call gives us the CPU time (essentially get_clocktime
or getrusage under the hood to get the CPU time) of the process and
allocation count at the last GC. Can we use getRTSStats at point A and then
getRTSStats at point B and diff them?

There are two problems with this. (1) the allocation count is recorded
from the last GC which does not correspond to point A or point B unless
we force a GC at both the points which is not practical and is going to
change the performance characteristics of the program drastically. (2)
the CPU time that we get is for the entire process which includes all
the OS threads, if the program is built with `-threaded` option then
this is always going to be inaccurate; even if we get OS thread level
CPU time we cannot attribute it correctly to Haskell threads that run on
top of OS threads; even if we build the program without `-threaded` we
may have multiple Haskell threads running and if between our measurement
points the threads are switched we will attribute the other thread's
time incorrectly to our current control flow. Even if we have a single
Haskell thread if there is a threadDelay or an IO call in the middle of
the two points the thread will switch out and even the time when it was
not scheduled will be counted in our measurement. So the only case left
is when we build without `-threaded`, we have a single Haskell thread
and we ensure it will not yield in the middle.

This is used in this way in micro-benchmarking programs but it is very
restrictive and useless in practice.

For small programs though the `+RTS -s` options is very useful to assess the
performance characteristics. I often take out the small piece that I want to
measure and run it with `+RTS -s`.

## Interpreting the RTS Stats

We divide the stats in two categories.  The first category is the non-gc
stats, these stats are accurate up to the time of the `getRTSStats`
call. These stats include:

* `cpu_ns`: the total accumulated `user` (note that it DOES NOT INCLUDE the
  system time) process CPU time till now starting after RTS init. So it
  includes the following:
  * Haskell thread CPU time
  * the RTS scheduler overhead
  * GC CPU time
* `mutator_cpu_ns` = `cpu_ns` - GC CPU time
* `elapsed_ns`: the total wall-clock time since RTS init.
* `mutator_elapsed_ns` = `elapsed_ns` - wall-clock time elapsed in GCs.

The second category is the GC stats, these stats are updated at the end
of a minor or major GC. They remain unchanged between GCs. All other
stats returned by `getRTSStats`, other than the ones listed above fall
in this category.

Note that the GC cpu time should be computed by adding the `gc_cpu_ns`
and `nonmoving_gc_cpu_ns` when the non-moving gc is enabled.

## Using getRTSStats with haskell-perf

In the `haskell-perf` library we do have a convenient way to wrap a
function around to use `getRTSStats` before and after it and print the
resulting stats. If the function passes the criterion mentioned above
then we can get a decent measurement, not very accurate but workable. We
automatically perform a GC before and after.

## Summary:

* `+RTS -s` option is very useful in assessing the behavior of the entire
  program.
* `getRTSStats` can be used to measure the timing of a piece of code if we are
  single threaded, the thread does not yield during the measurement, we are
  forcing GCs for roughly correct allocation counts. GCs happening in the
  middle of the measurement can add to the noise.
