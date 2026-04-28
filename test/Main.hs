import Control.Concurrent(threadDelay)
import Streamly.Metrics.Channel
    (Channel, newChannel, forkChannelPrinter, benchOnWith)
-- import Streamly.Metrics.Channel (printChannel)
import Streamly.Metrics.Perf.Type (PerfMetrics)

import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Stream as Stream
import Prelude hiding (sum)

noop :: Channel PerfMetrics -> IO ()
noop chan = do
    benchOnWith chan "noop" (const (return ())) (1000000 :: Int)

sum :: Channel PerfMetrics -> IO ()
sum chan = do
    _ <- benchOnWith
        chan "sum" (Stream.fold Fold.sum . Stream.enumerateFromTo (1::Int)) 1000000
    return ()

main :: IO ()
main = do
    chan <- newChannel
    _ <- forkChannelPrinter chan 10 100
    Stream.fold Fold.drain (Stream.replicateM 1000 (noop chan))
    Stream.fold Fold.drain (Stream.replicateM 1000 (sum chan))
    threadDelay 1000000
    {-
    Stream.drain
        ((Stream.replicateM 1000 (noop chan) <> Stream.replicateM 1000 (sum chan))
            `Stream.parallelFst` Stream.fromEffect (printChannel chan 1 10))
    -}
