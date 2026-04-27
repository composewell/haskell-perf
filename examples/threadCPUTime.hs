{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE MagicHash #-}

import Control.Concurrent (threadDelay)
import GHC.Exts (threadCPUTime#, ThreadId#)
import GHC.Int(Int64(..), Int32(..))
import GHC.IO(IO(..))

-- returns (sec, nsec, words, slices)
getThreadStats :: IO (Int64, Int64, Int32, Int32)
getThreadStats = IO $ \s ->
   case threadCPUTime# s of
    (# s', sec, nsec, allocs, slices #) ->
        (# s', (I64# sec, I64# nsec, I32# allocs, I32# slices) #)

-- returns (nsec, bytes, slices)
stat :: IO (Int64, Int32, Int32)
stat = do
    (sec, nsec, words, slices) <- getThreadStats
    let tenPow9 = 1000000000
        threadCPUTime = sec * tenPow9 + nsec
        threadAllocs = words * 8
        threadSlices = slices
    pure (threadCPUTime, threadAllocs, threadSlices)

main :: IO ()
main = do
    (nsec1, bytes1, slices1) <- stat
    -- threadDelay 1
    (nsec2, bytes2, slices2) <- stat
    putStrLn $ "cpu time nsec: " ++ show (nsec2 - nsec1)
    putStrLn $ "allocated bytes: " ++ show (bytes2 - bytes1)
    putStrLn $ "sched-out count: " ++ show (slices2 - slices1)
