module Streamly.Metrics.Channel.Unbounded
    (
      Channel
    , newChannel
    , send
    , printChannel
    , forkChannelPrinter
    , benchOn
    , benchOnWith
    )
where

import Control.Concurrent (forkIO, ThreadId, yield)
import Control.Concurrent.Chan
import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Function ((&))
import Streamly.Internal.Data.Time.Clock (getTime, Clock (Monotonic))
import Streamly.Internal.Data.Time.Units (AbsTime)
import Streamly.Metrics.Perf.Type (PerfMetrics(..))
import Streamly.Metrics.Perf (benchWith)
import Streamly.Metrics.Type (Indexable)
import Streamly.Data.Stream (Stream)
import Streamly.Data.Stream.Prelude (MonadAsync)

import qualified Streamly.Data.Stream as Stream

import Prelude hiding (showList)
import Streamly.Metrics.Channel.Common

-------------------------------------------------------------------------------
-- Event processing
-------------------------------------------------------------------------------

-- | An unbounded metrics channel. TBChannel should be preferred but this can
-- be used when STM cannot be used e.g. if TBChannel would create nested STM
-- transactions.
newtype Channel a = Channel (Chan (AbsTime, ([Char], [a])))

-- | Create a new metrics channel.
newChannel :: IO (Channel a)
newChannel = Channel <$> newChan

-- | Send a list of metrics to a metrics channel.
-- @send channel description metrics@
send :: MonadIO m => Channel a -> String -> [a] -> m ()
send (Channel chan) desc metrics = do
    -- XXX should use asyncClock
    now <- liftIO $ getTime Monotonic
    liftIO $ writeChan chan (now, (desc, metrics))
    liftIO yield

fromChan :: MonadAsync m => Chan a -> Stream m a
fromChan = Stream.repeatM . (liftIO . readChan)

-- | Forever print the metrics on a channel to the console periodically after
-- aggregating the metrics collected till now.
printChannel :: (MonadAsync m, Show a, Fractional a, Indexable a) =>
    Channel a -> Double -> Int -> m b
printChannel (Channel chan) timeout batchSize =
      fromChan chan
    & aggregateListBy timeout batchSize
    & printKV

forkChannelPrinter :: (MonadAsync m, Show a, Fractional a, Indexable a) =>
    Channel a -> Double -> Int -> m ThreadId
forkChannelPrinter chan timeout = liftIO . forkIO . printChannel chan timeout

-- | Benchmark a function application and the send the results to the specified
-- metrics channel.
benchOnWith :: Channel PerfMetrics -> String -> (a -> IO b) -> a -> IO b
benchOnWith chan desc f arg = do
    (r, xs) <- benchWith f arg
    send chan desc (Count 1 : xs)
    return r

-- | Like 'benchOnWith' but benchmark an action instead of function
-- application.
benchOn :: Channel PerfMetrics -> String -> IO b -> IO b
benchOn chan desc f = benchOnWith chan desc (const f) ()
