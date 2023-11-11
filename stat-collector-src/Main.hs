{-# LANGUAGE QuasiQuotes #-}

module Main (main) where

--------------------------------------------------------------------------------
-- Imports
--------------------------------------------------------------------------------

-- import Control.Concurrent (threadDelay)
import Control.Monad.IO.Class (MonadIO(..))
import Data.Function ((&))
import Data.List (foldl')
import Data.Map (Map)
import Data.Word (Word8)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import Foreign.Storable (Storable, peek)
import Numeric (showFFloat)
import Streamly.Data.Array (Array)
import Streamly.Data.Fold (Fold)
import Streamly.Data.Stream (Stream)
import Streamly.Internal.Data.Fold (Fold(..), Step(..))
import Streamly.Internal.Data.Ring (slidingWindow)
import Streamly.Internal.Data.Tuple.Strict (Tuple3Fused' (Tuple3Fused'))
import Streamly.Unicode.String (str)
import System.IO (hFlush, stdout, stdin)
import Text.Read (readMaybe)

import qualified Data.Map as Map
import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.FileSystem.Handle as Handle
import qualified Streamly.Internal.Data.Fold as Fold
import qualified Streamly.Internal.Data.Ring as Ring
import qualified Streamly.Unicode.Stream as Unicode
import qualified System.Console.ANSI as ANSI

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

data Counter
    = CpuTime
    | Allocated
    | SchedOut
    deriving (Read, Show, Ord, Eq)

type Tag = String
type Value = Double

type EventId = (Tag, Counter)

data Event
    = Event EventId Value

getEventId :: Event -> EventId
getEventId (Event evId _) = evId

getEventVal :: Event -> Value
getEventVal (Event _ evVal) = evVal

type StatResult = (Double, (Maybe (Double, Double)))

--------------------------------------------------------------------------------
-- Windowed Folds
--------------------------------------------------------------------------------

{-# INLINE range #-}
range :: (MonadIO m, Storable a, Ord a, Show a) => Int -> Fold m a (Maybe (a, a))
range n = Fold step initial extract

    where

    -- XXX Use Ring unfold and then fold for composing maximum and minimum to
    -- get the range.

    initial =
        if n <= 0
        then error "range: window size must be > 0"
        else
            let f (a, b) = Partial $ Tuple3Fused' a b (0 :: Int)
             in fmap f $ liftIO $ Ring.new n

    step (Tuple3Fused' rb rh i) a = do
        rh1 <- liftIO $ Ring.unsafeInsert rb rh a
        return $ Partial $ Tuple3Fused' rb rh1 (i + 1)

    -- XXX We need better Ring array APIs so that we can unfold the ring to a
    -- stream and fold the stream using a fold of our choice.
    --
    -- We could just scan the stream to get a stream of ring buffers and then
    -- map required folds over those, but we need to be careful that all those
    -- rings refer to the same mutable ring, therefore, downstream needs to
    -- process those strictly before it can change.
    foldFunc i
        | i < n = Ring.unsafeFoldRingM
        | otherwise = Ring.unsafeFoldRingFullM

    extract (Tuple3Fused' rb rh i) =
        if i == 0
        then return Nothing
        else do
            x <- liftIO $ peek (unsafeForeignPtrToPtr (Ring.ringStart rb))
            let accum (mn, mx) a = do
                  return (min mn a, max mx a)
            fmap Just $ foldFunc i rh accum (x, x) rb

--------------------------------------------------------------------------------
-- Parsing Input
--------------------------------------------------------------------------------

-- Event format:
-- STAT/<counterName>/<tag>/<value>

errorString :: String -> String -> String
errorString line reason = [str|Error:
Line: #{line}
Reason: #{reason}
|]

parseLineToEvent :: Monad m => String -> m (Either String Event)
parseLineToEvent line = do
    res <-
        Stream.fromList line
            & Stream.foldMany (Fold.takeEndBy_ (== '/') Fold.toList)
            & Stream.toList
    case res of
        ["STAT", counter, tag, val] ->
            case readMaybe counter :: Maybe Counter of
                Just x ->
                    case readMaybe val :: Maybe Double of
                        Just y -> pure $ Right $ Event (tag, x) y
                        Nothing ->
                            pure $ Left $ errorString line "Not a valid value"
                Nothing -> pure $ Left $ errorString line "Not a valid counter"
        _ -> pure $ Left $ errorString line "Chunks /= 4"

parseInputToEventStream :: MonadIO m => Stream m (Array Word8) -> Stream m Event
parseInputToEventStream inp =
    Unicode.decodeUtf8Chunks inp
        & Stream.foldMany
              (Fold.takeEndBy_
                   (== '\n')
                   (Fold.rmapM parseLineToEvent Fold.toList))
        & Stream.catRights

--------------------------------------------------------------------------------
-- Processing stats
--------------------------------------------------------------------------------

statCollector :: MonadIO m => Int -> Fold m Double StatResult
statCollector winSize =
    slidingWindow
        winSize
        (Fold.tee Fold.windowMean (Fold.lmap fst (range winSize)))

eventCollector :: MonadIO m => Int -> Fold m Event (Map EventId StatResult)
eventCollector winSize =
    Fold.toMap getEventId (Fold.lmap getEventVal (statCollector winSize))

scanStats :: MonadIO m => Stream m Event -> Stream m (Map EventId StatResult)
scanStats = Stream.postscan (eventCollector 100)

--------------------------------------------------------------------------------
-- Printing stats
--------------------------------------------------------------------------------

fill :: Int -> String  -> String
fill i x =
    let len = length x
     in replicate (i - len) ' ' ++ x

printTable :: [[String]] -> IO ()
printTable rows = do
    case map (unwords . fillRow) rows of
        [] -> putStrLn "printTable: empty rows"
        (header:rest) -> putStrLn $ unlines $ header:unwords separatorRow:rest

    where

    rowLengths = map (map length) rows -- [[Int]]
    maxLengths = foldl' (zipWith max) (head rowLengths) rowLengths
    separatorRow = map (\n -> replicate n '-') maxLengths
    fillRow r = zipWith (\n x -> fill n x) maxLengths r

statsToTable :: Map EventId StatResult -> [[String]]
statsToTable mp =
    ["Tag", "Counter", "Mean", "Min", "Max"]
        : map
              (\((t, c), (me, rg)) ->
                   [ t
                   , show c
                   , showFFloat (Just 2) me ""
                   , showMaybe (fmap fst rg)
                   , showMaybe (fmap snd rg)
                   ])
              (Map.toList mp)

    where

    showMaybe Nothing = "-"
    showMaybe (Just x) = showFFloat (Just 2) x ""

printSlidingStats :: Stream IO (Map EventId StatResult) -> IO ()
printSlidingStats strm =
    Stream.fold
        (Fold.drainMapM
            (\mp -> do
                 ANSI.clearScreen
                 printTable (statsToTable mp)
                 hFlush stdout
                 -- threadDelay 1000000
            ))
        strm

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
    Stream.unfold Handle.chunkReader stdin
        & parseInputToEventStream
        & scanStats
        & printSlidingStats
