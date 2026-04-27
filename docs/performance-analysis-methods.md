# Haskell Performance Analysis

## GHC RTS Stats

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

## GHC Event logging

Eventlog based Haskell thread aware time and allocation analysis is
possible with stock GHC but there are some limitations and drawbacks
which are fixed in the RTS patch described below. The patch basically
adds accurate information and more information, and we then use a custom
event log analysis program to provide an accurate and comprehensive
picture of the entire program.

TBD: document the exact limitations and differences.

## threadCPUTime# prim op

Available in the
[GHC 9.2.8 RTS patch](https://github.com/composewell/ghc/releases/tag/ghc-9.2.8-perf-counters-1-rc1).

Install the patched GHC using:

```
ghcup install -u https://github.com/composewell/ghc/releases/download/ghc-9.2.8-perf-counters-1-rc1/ghc-9.2.8.20231130-x86_64-unknown-linux.tar.xz ghc
```

This is a very simple and easy to use mechanism. The RTS is modified
such that we record the accurate time and allocation information in a
Haskell thread control block at the points when the thread is scheduled
and descheduled. Thus for each Haskell thread we can always get how much
time the thread spent on CPU and how much allocations it did.

An RTS API is provided to fetch the current thread's accumulated cpu
time and allocation stats.  We can collect these stats between point A
and B in a program, diff will tell us the time spent and allocations
between the two points.

We have to ensure that we are diffing the data for the same thread id at
both the points. See [this example program](./threadCPUTime.hs).

The API has some measurement overhead but it is not very high.  If we
are nesting measurements be aware that outer measurement will measure
the measurement overhead of the inner one. If you are measuring a
relatively small amount of time then reduce the overhead (approx 2
microseconds and 300 byte allocations, measure the exact value using an
empty code block).

This is very useful in micro-measurements and analysis of the CPU cost
different segments of code in a particular Haskell thread without worrying
about the preemption points of the thread.

By measuring the wall clock time as well at the two points we can find
the idle time for the thread. However, the idle time includes the queue
time and the IO time - it may not be very useful unless we know the
breakup. For that we need to add the facility to measure either the
queue time or the IO time.

Limitations: this allows only thread specific measurements, we cannot
tell what other threads and everything else in the system is doing
between the two points of measurements. It can be added to the patch
though. For accurate synchronization (if needed) of all threads at the
given points we can stop-the-world, can be useful in testing but not a
good idea in production though. Also, managing windows with possible
nesting can complicate the RTS code.

## Eventlog based perf counters

Available in GHC 8.10.7 RTS patch. Can be ported to later GHCs.

This gives you a more comprehensive picture of the entire program
between any two specified points, it gives a detailed report about all
the threads in the system not just the current thread.

See the [README](../README.md) for more details on this.
