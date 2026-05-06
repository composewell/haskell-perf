import Control.Concurrent(threadDelay)
import Streamly.Metrics.Channel
    (Channel, newChannel, forkChannelPrinter, benchOnWith)
-- import Streamly.Metrics.Channel (printChannel)
import Streamly.Metrics.Perf.Type (PerfMetrics)

import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Stream as Stream

runWithStats :: Channel PerfMetrics -> String -> (a -> IO b) -> a -> IO ()
runWithStats chan label f arg = do
    _ <- benchOnWith chan label f arg
    return ()

-- A simple operation that does nothing. When we measure this operation the cpu
-- time that is spent is just the overhead of the measuring code.
noOp :: b -> IO ()
noOp = (const (return ()))

sumOp :: Int -> IO Int
sumOp =
      Stream.fold Fold.sum
    . Stream.enumerateFromTo (1::Int)

{-
-- Pure code example
listSum :: Int -> Int
listSum =
      sum
    . enumFromTo (1::Int)
-}

timeout :: Int
timeout = 1

initStats :: IO (Channel PerfMetrics)
initStats = do
    chan <- newChannel
    -- The channel will collect 100 samples per label, as soon as it receives
    -- 100 it will print the stats and start collecting the next batch.
    -- If no sample comes in "timeout" seconds then print the batch anyway.
    -- @forkChannelPrinter channel timeout batch-size@.
    _ <- forkChannelPrinter chan (fromIntegral timeout) 100
    return chan

main :: IO ()
main = do
    -- Initialize a channel to send the stats to
    chan <- initStats

    let withStats = runWithStats chan

    -- One shot measurement, just one call
    withStats "noOpOne" noOp (1000000 :: Int)
    withStats "sumOpOne" sumOp (1000000 :: Int)

    -- Run many iterations and print the stats for batches of 100
    let iterations n = Stream.fold Fold.drain . Stream.replicateM n
        withStatsMany label f arg = iterations 1000 $ runWithStats chan label f arg

    -- Run the "noOp" and "sumOp" 1000 times, passing 1000000 as argument and
    -- sending the stats to "chan". The stats collected for noOp and sumOp will
    -- be sent to the channel and printed by it on console.
    withStatsMany "noOpMany" noOp (1000000 :: Int)
    withStatsMany "sumOpMany" sumOp (1000000 :: Int)

    -- Wait for the channel to drain
    threadDelay ((timeout + 2) * 1000000)
