# Haskell Performance Analysis

## Analysis Methods:

There are three different ways of investigating the performance:

* GHC RTS Stats (available in stock GHC)
* threadCPUTime# RTS primitive (available in patched GHC)
* GHC Event logging (available in patched GHC)

## Why this package?

* GHC RTS does not provide thread-wise stats, so it is not useful in
  multi-threaded programs. Not useful in production systems.
* It does not give accurate point-to-point allocation stats, only gc-to-gc
  stats which are not very useful in production systems.

threadCPUTime# primtive added vis a RTS patch allows us to find accurate CPU
time, allocation, cache misses and many other CPU performace counters on a per
thread basis between point A to point B in a program.

The event logging patch is the most powerful method and records what the entire
system is doing at any point of time via events logged to a file. The file is
later analyzed to find out the CPU time, allocations and other stats on a per
thread basis or on defined window basis.

The RTS patch added additional events required to record the cpu time,
allocations and other counters accurately.  This package provides a
program to analyze the generated event log file and build correct
performance picture of the program.

## What it does

With these tools you can find:

* Analyze the entire program to:
  * where is the program spending time
  * How much time and allocations each thread is doing
  * account for each and every microsecond spent and bytes allocated
* Zoom into smaller parts of the program to find why it is expensive:
  * analyze smaller windows in more detail
  * report micro level hardware counters e.g. cache misses
* It can run in production systems without much overhead

## What it provides?

* A library for easy data collection and reporting in a running a program
* An executable to analyze the eventlog offline

## Detailed documents

For more details on each of the performance analysis methods see the following
documents:
* GHC RTS Stats
* threadCPUTime# RTS primitive
* GHC Event logging
* GHC patches details
