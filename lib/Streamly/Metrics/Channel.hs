module Streamly.Metrics.Channel
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

import Control.Concurrent (forkIO, ThreadId)
import Control.Concurrent.STM (atomically)
import Control.Concurrent.STM.TBQueue
    (TBQueue, newTBQueue, readTBQueue, writeTBQueue)
import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Function ((&))
import Streamly.Data.Stream (Stream)
import Streamly.Internal.Data.Time.Clock (getTime, Clock (Monotonic))
import Streamly.Internal.Data.Time.Units (AbsTime)
import Streamly.Metrics.Channel.Common (aggregateListBy, printKV)
import Streamly.Metrics.Perf.Type (PerfMetrics(..))
import Streamly.Metrics.Perf (benchWith)
import Streamly.Metrics.Type (Indexable)
import Streamly.Data.Stream.Prelude (MonadAsync)

import qualified Streamly.Data.Stream as Stream

-------------------------------------------------------------------------------
-- Event processing
-------------------------------------------------------------------------------

-- XXX Use streamly SVar instead so that we do not need STM and we can use just
-- one channel type.

-- | A metrics channel.
newtype Channel a = Channel (TBQueue (AbsTime, ([Char], [a])))

-- | Create a new metrics channel.
newChannel :: IO (Channel a)
newChannel = atomically $ do
    tbq <- newTBQueue 1
    return $ Channel tbq

-- | Send a list of metrics to a metrics channel.
-- @send channel description metrics@
send :: MonadIO m => Channel a -> String -> [a] -> m ()
send (Channel chan) desc metrics = do
    -- XXX should use asyncClock
    now <- liftIO $ getTime Monotonic
    liftIO $ atomically $ writeTBQueue chan (now, (desc, metrics))

fromChan :: MonadAsync m => TBQueue a -> Stream m a
fromChan = Stream.repeatM . (liftIO . atomically . readTBQueue)

-- XXX Print actual batch size and also scale the results per event.

-- | Forever print the metrics on a channel to the console periodically after
-- aggregating the metrics collected till now.
printChannel :: (MonadAsync m, Show a, Fractional a, Indexable a) =>
    Channel a -> Double -> Int -> m b
printChannel (Channel chan) timeout batchSize =
      fromChan chan
    & aggregateListBy timeout batchSize
    & printKV

-- | Start an async thread to print the stats received on the supplied channel
-- and print the stats on console.
--
-- Usage: @forkChannelPrinter channel timeout batch-size@.
--
-- Stats are printed when either as many stat samples as the batch size have
-- been received or we have not received a stat in "timeout" seconds.
forkChannelPrinter :: (MonadAsync m, Show a, Fractional a, Indexable a) =>
    Channel a -> Double -> Int -> m ThreadId
forkChannelPrinter chan timeout = liftIO . forkIO . printChannel chan timeout

-- | Benchmark a function application and send the results to the specified
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
