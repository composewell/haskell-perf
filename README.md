# haskell-perf

GHC Patch: https://github.com/composewell/ghc/tree/ghc-8.10.7-eventlog-enhancements

## Enable perf counters

Enable unrestricted use of perf counters:

```
# echo -1 > /proc/sys/kernel/perf_event_paranoid
```

## Disable CPU scaling

Set the scaling governer of all your cpus to `performance`:

```
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
...
...
echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
```

## Generating the eventlog

To generate the event log, we need to compile the program with the eventlog enabled
and run the program setting the `-l` rts option.

There are multiple ways of doing this.

__Using plain GHC__:

```
ghc Main.hs -rtsopts -eventlog
./Main +RTS -l -RTS
```

__Using Cabal__:

The `.cabal` file should contain the following ghc options
```
ghc-options: -eventlog "-with-rtsopts=-l"
```

If the `-threaded` option is used while compiling. You may want to use the `-N1`
rts option.

## Creating windows

Helper function to create windows:

```
{-# LANGUAGE BangPatterns #-}

import Control.Monad.IO.Class (MonadIO(..))
import Debug.Trace (traceEventIO)

{-# INLINE withTracingFlow #-}
withTracingFlow :: MonadIO m => String -> m a -> m a
withTracingFlow tag action = do
    liftIO $ traceEventIO ("START:" ++ tag)
    !res <- action
    liftIO $ traceEventIO ("END:" ++ tag)
    pure res
```

We can wrap parts of the flow we want to analyze with `withTracingFlow` using a
tag to help us identify it.

## Labelling Threads

We should label our threads to identify the thread to scrutinize while reading
the stats.

For example,

To scrutinize the main thread:

```
import GHC.Conc (myThreadId, labelThread)

main :: IO ()
main = do
    tid <- myThreadId
    labelThread tid "main-thread"
    withTracingFlow "main" $ do
       ...
```

To scrutinize the server thread in warp we can use the following middleware:

```
eventlogMiddleware :: Application -> Application
eventlogMiddleware app request respond = do
    tid <- myThreadId
    labelThread tid "server"
    traceEventIO ("START:server")
    app request respond1

    where

    respond1 r = do
        res <- respond r
        traceEventIO ("END:server")
        return res

```

We can use `eventlogMiddleware` as the outermost layer.

## Reading the results

We get a lot of output currently. We are in the process of simplifying the
statistics and making the details controllable via options.

Currently, the program prints a lot of information. It's essential to understand
what to ignore given the use case.

The use-case we assume is: __Understand the window CPU time and Thread allocated__.

Consider the following program:

```
{-# LANGUAGE BangPatterns #-}

import Control.Monad (unless)
import Control.Monad.IO.Class (MonadIO(..))
import Debug.Trace (traceEventIO)
import GHC.Conc (myThreadId, labelThread)

{-# INLINE withTracingFlow #-}
withTracingFlow :: MonadIO m => String -> m a -> m a
withTracingFlow tag action = do
    liftIO $ traceEventIO ("START:" ++ tag)
    !res <- action
    liftIO $ traceEventIO ("END:" ++ tag)
    pure res

{-# INLINE printSumLoop #-}
printSumLoop :: Int -> Int -> Int -> IO ()
printSumLoop _ _ 0 = print "All Done!"
printSumLoop chunksOf from times = do
    withTracingFlow "sum" $ print $ sum [from..(from + chunksOf)]
    printSumLoop chunksOf (from + chunksOf) (times - 1)

main :: IO ()
main = do
    tid <- myThreadId
    labelThread tid "main-thread"
    withTracingFlow "main" $ do
         printSumLoop 10000 1 100
```

The statics gleaned from the eventlog of the above program will look like the
following:

```
--------------------------------------------------
Summary Stats
--------------------------------------------------

Global thread wise stat summary
tid       label samples ThreadCPUTime ThreadAllocated
--- ----------- ------- ------------- ---------------
  1 main-thread       2       967,479         434,384
  2           -       1         5,854          17,664

  -           -       3       973,333         452,048


Window [1:main] thread wise stat summary
ProcessCPUTime: 1,174,455
ProcessUserCPUTime: 0
ProcessSystemCPUTime: 1,175,000

ThreadCPUTime:934,898
GcCPUTime:0
RtsCPUTime:239,557
tid       label samples ThreadCPUTime ThreadAllocated
--- ----------- ------- ------------- ---------------
  1 main-thread       1       934,898         429,952

  -           -       1       934,898         429,952


Window [1:sum] thread wise stat summary
ProcessCPUTime: 953,862
ProcessUserCPUTime: 0
ProcessSystemCPUTime: 949,000

ThreadCPUTime:833,991
GcCPUTime:0
RtsCPUTime:119,871
tid       label samples ThreadCPUTime ThreadAllocated
--- ----------- ------- ------------- ---------------
  1 main-thread     100       833,991         328,224

  -           -     100       833,991         328,224


--------------------------------------------------
Detailed Stats
--------------------------------------------------

Window [1:main] thread wise stats for [ThreadCPUTime]
tid       label   total count     avg minimum maximum stddev
--- ----------- ------- ----- ------- ------- ------- ------
  1 main-thread 934,898     1 934,898 934,898 934,898      0


Grand total: 934,898

Window [1:main] thread wise stats for [ThreadAllocated]
tid       label   total count     avg minimum maximum stddev
--- ----------- ------- ----- ------- ------- ------- ------
  1 main-thread 429,952     1 429,952 429,952 429,952      0


Grand total: 429,952

Window [1:sum] thread wise stats for [ThreadCPUTime]
tid       label   total count   avg minimum maximum stddev
--- ----------- ------- ----- ----- ------- ------- ------
  1 main-thread 833,991   100 8,340   5,533  63,493  5,714


Grand total: 833,991

Window [1:sum] thread wise stats for [ThreadAllocated]
tid       label   total count   avg minimum maximum stddev
--- ----------- ------- ----- ----- ------- ------- ------
  1 main-thread 328,224   100 3,282   2,960  31,584  2,844


Grand total: 328,224

Global thread wise stats for [ThreadCPUTime]
tid       label   total count     avg minimum maximum  stddev
--- ----------- ------- ----- ------- ------- ------- -------
  1 main-thread 967,479     2 483,740  33,519 933,960 450,220
  2           -   5,854     1   5,854   5,854   5,854       0


Grand total: 973,333

Global thread wise stats for [ThreadAllocated]
tid       label   total count     avg minimum maximum  stddev
--- ----------- ------- ----- ------- ------- ------- -------
  1 main-thread 434,384     2 217,192   4,920 429,464 212,272
  2           -  17,664     1  17,664  17,664  17,664       0


Grand total: 452,048
```

From the __Global thread wise stat summary__ under __Summary Stats__ figure out
the thread id we want to scrutinize. In this case, we care about the
`main-thread`. The thread id is `1`.

We can skip to the __Detailed Stats__ section.

We want to look at all the windows we want to scrutinize that run in the
`main-thread`. The windows in the above program are `main` and `sum`.  The
thread id is prepended to the windows. So we want to look at sections
corresponding to `[1:main]` and `[1:sum]`.

That is,
```
Window [1:main] thread wise stats for [ThreadCPUTime]
tid       label   total count     avg minimum maximum stddev
--- ----------- ------- ----- ------- ------- ------- ------
  1 main-thread 934,898     1 934,898 934,898 934,898      0


Grand total: 934,898

Window [1:main] thread wise stats for [ThreadAllocated]
tid       label   total count     avg minimum maximum stddev
--- ----------- ------- ----- ------- ------- ------- ------
  1 main-thread 429,952     1 429,952 429,952 429,952      0


Grand total: 429,952

Window [1:sum] thread wise stats for [ThreadCPUTime]
tid       label   total count   avg minimum maximum stddev
--- ----------- ------- ----- ----- ------- ------- ------
  1 main-thread 833,991   100 8,340   5,533  63,493  5,714


Grand total: 833,991

Window [1:sum] thread wise stats for [ThreadAllocated]
tid       label   total count   avg minimum maximum stddev
--- ----------- ------- ----- ----- ------- ------- ------
  1 main-thread 328,224   100 3,282   2,960  31,584  2,844
```

Consider one specific section,

```
Window [1:sum] thread wise stats for [ThreadCPUTime]
tid       label   total count   avg minimum maximum stddev
--- ----------- ------- ----- ----- ------- ------- ------
  1 main-thread 833,991   100 8,340   5,533  63,493  5,714
```

This section is a table. It has 8 columns. It can have multiple rows.  We should
only scrutinize the row where the `tid` matches `main-thread`. ie. `tid == 1`.

The granularity of `ThreadCPUTime` is in nanoseconds and `ThreadAllocated` is
in bytes.

Columns:

- `tid`: The thread id
- `label`: The thread label
- `total`: The total accumulated sum of all the samples
- `count`: Number of samples or the times this window is seen
- `avg`: The average size of the samples
- `minimum`: The minimum of all the samples
- `maximum`: The maximum of all the samples
- `stddev`: The standard deviation of the samples

__NOTE__: It is important to look at `stddev`. If `stddev` is more than 30% of
the average and if the difference between the `minimum` and `maximum` is too
much, the `average` might have unecessary outliers. In the future we would like
to remove outliers automatically.
