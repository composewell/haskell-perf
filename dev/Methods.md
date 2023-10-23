# Methods

There are 2 methods of measuring accurate haskell performance.

## Eventlog method

__Pros__:

- Enables us to perform extensive analisys.
  - We can understand the state of the system completely at any given time.
  - We can make more elaborate windows.
  - We can see all the other threads in a given window.
  - The windows are not confined to a single thread.
- Relatievely less invasive change in the RTS.

__Cons__:

- Overhead of measurement is more.
- Collecting metrics while the app is running in production is not straight.
  forward and requires more moving parts.

## ThreadCPUTime primop method

__Pros__:

- Low measurement overhead.
- Collecting metrics in production is straight forward.

__Cons__:

- Relatively more invasive.
- Very simple analisys. Relative analisys isn't possible to understand the system.

__Implementation detail__:

- We need to hang the perfomance counters block from the TSO.
- On every tield we can update the perf counters.
- To get the updated value of the counter we need to yield.
