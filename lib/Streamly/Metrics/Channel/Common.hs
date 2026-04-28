module Streamly.Metrics.Channel.Common
    (
      aggregateListBy
    , printKV
    )
where

import Control.Monad.IO.Class (liftIO, MonadIO)
import Data.Bifunctor (second)
import Data.Maybe (fromJust, isJust)
import Streamly.Internal.Data.Time.Units (AbsTime)
import Streamly.Metrics.Type (showList, Indexable)
import Streamly.Data.Stream (Stream)
import Streamly.Data.Stream.Prelude (MonadAsync)

import qualified Streamly.Internal.Data.Fold as Fold
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.Internal.Data.Stream.Prelude as Stream

import Prelude hiding (showList)

-------------------------------------------------------------------------------
-- Event processing
-------------------------------------------------------------------------------

aggregateListBy :: (MonadAsync m, Ord k, Fractional a) =>
    Double -> Int -> Stream m (AbsTime, (k, [a])) -> Stream m (k, [a])
aggregateListBy timeout batchsize stream =
    fmap (second fromJust)
        $ Stream.filter (isJust . snd)
        $ Stream.classifySessionsBy
            0.1 False (return . (> 1000)) timeout f stream

    where

    scale Nothing _ = Nothing
    scale (Just xs) count = Just $ map (/ count) xs

    f =
        Fold.teeWithFst
            scale
            (Fold.take batchsize (Fold.foldl1' (zipWith (+))))
            (Fold.lmap (const 1) Fold.sum)

printKV :: (MonadIO m, Show k, Show a, Indexable a) => Stream m (k, [a]) -> m b
printKV stream =
    let f (k, xs) = liftIO $ putStrLn $ show k ++ ":\n" ++ showList xs
     in Stream.fold (Fold.drainMapM f) stream >> error "printChannel: Metrics channel closed"
