<!--
# GHC RTS Stats

RTS Stats give entire OS process level (not OS thread level) cpu time
and not Haskell thread cpu time. When multiple OS threads are used, the
cpu time recorded is the cpu time of all the threads combined. Also, the
way kernel accounts this time it could be off by a little (microseconds)
because each thread's cpu time is recorded at the last accounting
event. Allocations are recorded by the GHC RTS only at the GC boundary,
so the allocations reported are from the point when the last GC
happened. So we need to be careful when using or interpreting these
stats.

If we built the program without -threaded and we are using a single
Haskell thread then we can get cpu time between any two points in the
program accurately. Accurate accounting of allocations will require a GC
to be forced which is not usually practical.

In a multithreaded program using RTS stats we can only tell time how
much total CPU time (and allocations) the entire Haskell process (all
threads) spent between two points, but we cannot tell which Haskell
thread spent how much time.
-->

# Components of a Haskell Process

* An OS level process
* Multiple OS level threads in the OS process
* Multiple Haskell green threads that are scheduled on the OS threads. Haskell
  threads can run on any of the available OS threads every time it is ready to
  run.

# A prototypical program

A simple yet comprehensive program to understand different components of
performance analysis and stats. You can play with this to understand how things
work, how the stats add up and what they mean.

See examples/console-loop-multi-thread.hs .

# How many OS threads do we have?

To see how may OS threads a haskell process is using on Linux.
Run examples/console-loop-multi-thread.hs , note its pid printed in the output.
All of its OS threads are:
```
ls /proc/<PID>/task
```

Even when compiled without the `-threaded` option we might see two threads
because the RTS still uses a separate thread for GC and for forking async
cleanup threads via GC.

One of the tasks will have the same pid as the process pid, this is the main OS
thread. You can try changing the number of capabilities using +RTS -N and see
the effect.

GHC may also use independent OS threads for ffi, for GC, for IO manager,
however it will guarantee that only as many user threads can run at a time as
specified with the -N rts option.

Usually we see 3 threads plus 2 threads per capability when compiled
with `-threaded` option.

# GHC RTS stats

The getRTSStats call gives us the CPU time (essentially get_clocktime
or getrusage under the hood to get the CPU time) of the process and
allocation count at the last GC. Can we use getRTSStats at point A and then
getRTSStats at point B and diff them?

There are two problems with this. (1) the allocation count is recorded
from the last GC which does not correspond to point A or point B unless
we force a GC at both the points which is not practical and is going to
change the performance characterstics of the program drastically. (2)
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
performance characterstics. I often take out the small piece that I want to
measure and run it with `+RTS -s`.

# Using it

In the haskell-perf library we do have a convenient way to wrap a function
around and use getRTSStats before and after it. If the function passes the
criterion mentioned above then we can get a decent measurement, not very
accurate but workable. We automatically perform a GC before and after.

# Interpreting the RTS Stats

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

# Variability of Measurements

Performance measurement is tricky and there are many factors to take care of if
you want to get reliable results:

* cannot use wall-clock time, need to use process cpu time
* disable CPU frequency scaling
* Memory contention can affect the measurement, do not run other things on the
  same machine.
* cache effect due to context switching can affect it, do not run other things
  on the same machine.
* Different CPUs running at different frequencies can make the results
  unpredictable.
* The clocks of different CPUs may not be in sync.

To counter the last two factors we should use instruction count or allocation
count rather than time as a reliable measure.
