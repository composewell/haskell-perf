module Streamly.Metrics.Perf
    (
      PerfMetrics(..)
    , benchWith
    , bench
    , preRun
    , postRun
    )
where

import Control.Monad (unless)
import Data.Maybe (catMaybes)
import GHC.Stats (getRTSStats, getRTSStatsEnabled, RTSStats(..))
import Streamly.Internal.Data.Time.Units (NanoSecond64, fromAbsTime)
import Streamly.Metrics.Measure (measureWith)
import Streamly.Metrics.Perf.Type (PerfMetrics(..), checkMonotony)
import Streamly.Metrics.Perf.RUsage (getRuMetrics, pattern RUsageSelf)
import Text.Show.Pretty (ppShow)

import qualified Streamly.Internal.Data.Time.Clock as Clock

{-# INLINE getProcMetrics #-}
getProcMetrics :: IO [PerfMetrics]
getProcMetrics = do
    time <- Clock.getTime Clock.Monotonic
    tcpu <- Clock.getTime Clock.ThreadCPUTime
    pcpu <- Clock.getTime Clock.ProcessCPUTime

    let tcpuSec = fromIntegral (fromAbsTime tcpu :: NanoSecond64) * 1e-9
    let pcpuSec = fromIntegral (fromAbsTime pcpu :: NanoSecond64) * 1e-9
    let timeSec = fromIntegral (fromAbsTime time :: NanoSecond64) * 1e-9
    return
        [ MonotonicTime timeSec
        , ProcessCPUTime pcpuSec
        , ThreadCPUTime tcpuSec
        ]

-- Compatible with GHC 8.2 (base 4.10) onwards
{-# INLINE getGcMetrics #-}
getGcMetrics :: IO [PerfMetrics]
getGcMetrics = do
    res <- getRTSStatsEnabled
    if res
    then do
        stats <- getRTSStats
        pure
            [ GcAllocatedBytes (fromIntegral (allocated_bytes stats))
            , GcCopiedBytes (fromIntegral (copied_bytes stats))
            , GcMaxLiveBytes (fromIntegral (max_live_bytes stats))
            , GcMaxLargeObjectBytes (fromIntegral (max_large_objects_bytes stats))
            , GcMaxCompactBytes (fromIntegral (max_compact_bytes stats))
            , GcMaxMemInUse (fromIntegral (max_mem_in_use_bytes stats))
            , GcMutatorCpuTime (fromIntegral (mutator_cpu_ns stats) / 1e9)
            , GcMutatorElapsedTime
                (fromIntegral (mutator_elapsed_ns stats) / 1e9)
            , GcGcCpuTime (fromIntegral (gc_cpu_ns stats) / 1e9)
            , GcGcElapsedTime (fromIntegral (gc_elapsed_ns stats) / 1e9)
            , GcCpuTime (fromIntegral (cpu_ns stats) / 1e9)
            , GcElapsedTime (fromIntegral (elapsed_ns stats) / 1e9)
            ]
    else pure []

{-# INLINE getPerfMetrics #-}
getPerfMetrics :: IO [PerfMetrics]
getPerfMetrics = do
    procMetrics <- getProcMetrics
    gcMetrics <- getGcMetrics
    ruMetrics <- getRuMetrics RUsageSelf
    return $ concat [procMetrics, gcMetrics, ruMetrics]

{-# INLINE preRun #-}
preRun :: IO [PerfMetrics]
preRun = do
      -- XXX If we have nested perf measurement calls then it is a bad idea to
      -- perform GC.
      -- performGC
      getPerfMetrics

{-# INLINE postRun #-}
postRun :: [PerfMetrics] -> IO [PerfMetrics]
postRun stats = do
      -- We should not leave garbage behind. Any GC work is part of the
      -- function being benchmarked. We start with no garbage before we start
      -- measuring and we leave no garbage behind. However, this also adds a
      -- constant overhead of GC which would otherwise be lesser if we do not
      -- perform GC too often.  So this may show inflated cpu times but the
      -- numbers would be consistent across iterations.
      --
      -- XXX We can have this as an option.
      -- XXX The allocations and cycles consumed by the measuring functions
      -- also add to the benchmark stats. We could run a dummy function and
      -- measure the overhead and deduct that overhead. However, this is pretty
      -- small and should not matter when benchmarking large functions.
      -- performGC
      stats1 <- getPerfMetrics
      let diff = zipWith (-) stats1 stats
          r =
                catMaybes
                    $ zipWith (\old new ->
                        if checkMonotony old new
                        then Nothing
                        else Just (new, old)) stats stats1
      unless (null r) $ do
          putStrLn "Warning: perf stat diff validation failed. "
          putStrLn "(New, old) ="
          putStrLn $ ppShow r
          putStrLn "New ="
          putStrLn $ ppShow stats1
          putStrLn "Old ="
          putStrLn $ ppShow stats
      return diff

-- | Benchmark a function application returning the function output and the
-- performance metrics.
benchWith :: (a -> IO b) -> a -> IO (b, [PerfMetrics])
benchWith = measureWith preRun postRun

-- | Like 'benchWith' but benchmark an action instead.
bench :: IO a -> IO (a, [PerfMetrics])
bench action = benchWith (const action) ()
