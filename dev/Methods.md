# Methods

There are 2 methods of measuring accurate Haskell performance.

## Eventlog method

__Pros__:

- Enables us to perform extensive analysis.
  - We can understand the state of the system completely at any given time.
  - We can make more elaborate windows.
  - We can see all the other threads in a given window.
  - The windows are not confined to a single thread.
- Relatively less invasive change in the RTS.

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
- Very simple analysis. Relative analysis isn't possible to understand the system.

__Implementation detail__:

- We need to hang the performance counters block from the TSO.
- On every yield we can update the perf counters.
- To get the updated value of the counter we need to yield.
