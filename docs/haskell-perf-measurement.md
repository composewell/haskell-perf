## Components of a Haskell Process

* An OS level process
* Multiple OS level threads within the OS process
* Multiple Haskell green threads that are scheduled on the OS threads. Haskell
  threads can run on any of the available OS threads every time it is ready to
  run.

## How many OS threads do we have?

To see how many OS threads a Haskell process is using on Linux.  Run
[this example](test/snippets/console-loop-multi-thread.hs), note its pid
printed in the output.  All of its OS threads can be printed by:
```
ls /proc/<PID>/task
```

## OS threads without -threaded option

Even when compiled without the `-threaded` option we can see two OS
threads running (ghc 9.10.3 on Linux) because the RTS uses one more
thread for some other async tasks.  But only one OS thread is used for
running the user program.

Not using the -threaded option may be more CPU efficient but it can
lead to higher latencies because of FFI calls blocking the thread. When
compiled with -threaded we can run with a single capability using -N1
and take adavantage of offloading FFI calls to other threads.

One of the tasks will have the same pid as the process pid, this is the
main OS thread used for executing Haskell threads.

## OS threads with -threaded build and -N1 RTS option

When compiled with -threaded option, GHC may also use multiple independent OS
threads for ffi, for GC, for IO manager.

With -N1 even though only one capability is used for running the Haskell
code we may see more threads being used by the RTS.  Using ghc 9.10.3 on
Linux we see five OS threads running with -N1.

## OS threads with -threaded build

You can try changing the number of capabilities using +RTS -N and see
the effect.  Usually we can see 3 threads plus 2 threads per capability
when compiled with `-threaded` option.  However, GHC guarantees that
only as many user threads can run at a time as specified with the -N rts
option.

## Process Level View

If we consider the entire Haskell process, the elapsed wall-clock time
of each OS thread consists of:

* CPU execution time,
* time spent runnable but waiting on the OS scheduler,
* time spent blocked waiting for I/O or synchronization events.

When a process uses multiple OS threads, `ProcessCPUTime` may exceed elapsed
wall-clock time because multiple threads can execute simultaneously on
different CPUs.

As a result, `ProcessCPUTime` is generally not suitable as a delta metric for
measuring the CPU consumption of a specific piece of user code in a
multithreaded program. In such cases, `ThreadCPUTime` is usually more
appropriate, since it measures CPU usage attributable to a single OS thread.
In a single-threaded process, however, `ProcessCPUTime` can safely be used for
this purpose.

`ProcessCPUTime` is nevertheless useful as an aggregate CPU utilization
metric. Its value can approach elapsed wall-clock time multiplied by the
degree of CPU parallelism available to the process. Lower values indicate
that the process spent more time not executing, for example waiting for
scheduling, synchronization, I/O, or other runtime stalls.

Useful reporting metric:

* Total Elapsed time
* Entire process CPU Time
    * User time (getrusage)
    * System time (getrusage)
* For each OS thread - ThreadCPUTime
* Overall rusage stats for the process
* OS Sched run-queue wait time (using sched_wakeup, sched_switch trace events
  via  perf, libperf, libtraceevent)

The total elapsed time for any OS thread can be decomposed as:
```
elapsed =
    on_cpu
  + runnable_wait
  + off_cpu_wait(reason)
```

```
reason in {
    futex,
    disk_io,
    network_io,
    epoll,
    sleep,
    paging,
    pipe_wait,
    signal_wait,
    ...
}
```

## Useful Metrics

Let's consider the following cases.

### Single Capability, Single Haskell Thread

This is the simplest execution model for performance analysis. In this
configuration, a single OS thread executes the measured code. If there
is also only one Haskell thread, then when that thread yields there is
no other Haskell thread that can run on the same capability. Likewise,
no parallel runtime worker executes on that capability. Aside from brief
RTS scheduler bookkeeping, the only substantial additional work that may
execute on that capability is garbage collection or foreign-function
(FFI) code.

In this model, the OS `ThreadCPUTime` delta can be used to measure the
total CPU time consumed by the execution thread between two points. This
measurement includes both user-code execution and RTS activity such as
garbage collection and scheduler overhead. To estimate the CPU time
spent in user code alone, the CPU time attributable to RTS activity must
be subtracted. Since there is no other Haskell thread, this entire CPU
time can be attributed to the single Haskell thread.

The CPU time between two points on the OS thread (GHC capability) can be
decomposed as follows:

* User-code execution time
* CPU time spent in FFI code
* Haskell GC CPU time
* RTS scheduler and bookkeeping CPU time

### Single Capability, Multiple Haskell Threads

In this configuration, `ThreadCPUTime` for the OS thread can no longer be
used to directly measure the CPU time consumed by a particular Haskell
thread. A Haskell thread may yield execution between the measurement
points, allowing another Haskell thread to run on the same capability and
consume CPU time on the same OS thread.

To measure CPU usage attributable to an individual Haskell thread, the
measurement must instead be performed at the RTS level. This can be done
using RTS eventlog tracing or by recording timestamps when the Haskell
thread is scheduled onto and descheduled from a capability.

The mutator CPU time reported by GHC is the cumulative CPU time spent
outside garbage collection. It therefore includes both Haskell thread
execution time and RTS overhead incurred during mutator execution, such
as scheduler bookkeeping and allocation management.

If we separately measure the cumulative CPU time attributable to
individual Haskell threads, then the difference between the total
mutator CPU time and the cumulative Haskell thread CPU time provides an
estimate of RTS overhead excluding garbage collection.

### Multiple capabilities

Same as above applies in this case as well.

## Variability of Measurements

Performance measurement is tricky and there are many factors to take care of if
you want to get reliable results:

* Disable CPU frequency scaling, can cause run-to-run or variability in the
  same run.
* Do not run other things on the same machine. interrupts, kernel
  activity, background daemons can also affect:
  * Memory contention can affect the measurement.
  * cache effects due to context switching can affect it.
* Discard first runs, first runs are usually outliers because of warm up effects,
  instruction cache cold, data cache cold, page faults, branch predictor
  not trained.
* Use thread affinity. Thread migration to another CPU: causes cache
  invalidation, different core state, timing noise.
* Use larger measurements. In smaller one measurement overhead and
  variance may dominate: timing calls (clock_gettime), counters, RTS
  stats.
* Different CPUs running at different frequencies can make the results
  unpredictable.
* The clocks of different CPUs may not be perfectly in sync.

To counter the last two factors we should use instruction count or
allocation count rather than time as a more reliable measure. Even
the instruction count might vary because of measurement overhead adds
instruction count, which can vary depending on how many times the thread
is context switched.

## Haskell specific variability

* Lazy evaluation, may defer work which might get evaluated later in the
  context of some other measurement window.

