{-# LANGUAGE QuasiQuotes #-}

module Main (main) where

--------------------------------------------------------------------------------
-- Imports
--------------------------------------------------------------------------------

-- import Control.Concurrent (threadDelay)
import Data.Int (Int32, Int64)
import System.Environment (getArgs)
import Control.Monad.IO.Class (MonadIO(..))
import Data.Function ((&))
import Data.List (foldl', findIndex, sortBy, find)
import Data.Map (Map)
import Data.Maybe (fromMaybe, fromJust)
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
import System.Posix.Signals (installHandler, Handler(Catch), sigINT, sigTERM)
import Data.Text.Format.Numbers (prettyI)

import qualified Data.Text as Text
import qualified Data.Map as Map
import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Stream.Prelude as Stream
import qualified Streamly.FileSystem.Handle as Handle
import qualified Streamly.Internal.Data.Fold as Fold
import qualified Streamly.Internal.Data.Ring as Ring
import qualified Streamly.Unicode.Stream as Unicode
import qualified System.Console.ANSI as ANSI

--------------------------------------------------------------------------------
-- Utils
--------------------------------------------------------------------------------

double :: Int -> Double
double = fromIntegral

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

data Boundary a b
    = Start a
    | End a (Maybe b)
    deriving (Read, Show, Ord, Eq)

getWindowId :: Boundary WindowId Label -> WindowId
getWindowId (Start a) = a
getWindowId (End a _) = a

data Counter
    = ThreadCpuTime
    | ProcessCpuTime
    | WallClockTime
    | Allocated
    | SchedOut
    deriving (Read, Show, Ord, Eq)

type WindowId = String
type Label = String
type ThreadId = Int32
type Tag = String
type Value = Int64

data EventId =
    EventId
        { evTid :: ThreadId
        , evCounter :: Counter
        , evTag :: Tag
        }
    deriving (Eq, Ord, Show)

data UnboundedEvent
    = UEvent (Boundary WindowId Label) ThreadId Counter Value
    deriving (Show)

data Event
    = Event EventId Value
    deriving (Show)

getEventId :: Event -> EventId
getEventId (Event evId _) = evId

getEventVal :: Event -> Value
getEventVal (Event _ evVal) = evVal

--------------------------------------------------------------------------------
-- Folds
--------------------------------------------------------------------------------

statsLayout :: [String]
statsLayout =
    [ "latest", "total", "count", "avg", "minimum", "maximum", "stddev"]

{-# INLINE stats #-}
stats :: Fold IO Int64 [(String, Int)]
stats =
      Fold.lmap (fromIntegral :: Int64 -> Int)
    $ Fold.distribute
        [ fmap (\x -> ("latest", fromJust x)) Fold.latest
        , fmap (\x -> ("total", x)) Fold.sum
        , fmap (\x -> ("count", x)) Fold.length
        , fmap (\x -> ("avg", round x)) (Fold.lmap double Fold.mean)
        , fmap (\x -> ("minimum", fromJust x)) Fold.minimum
        , fmap (\x -> ("maximum", fromJust x)) Fold.maximum
        , fmap (\x -> ("stddev", round x)) (Fold.lmap double Fold.stdDev)
        ]

--------------------------------------------------------------------------------
-- Parsing Input
--------------------------------------------------------------------------------

-- Event format:
-- Start/<window-id>/<label>/<tid>/<counterName>/<value>
-- End/<window-id>/<label>/<tid>/<counterName/<value>

errorString :: String -> String -> String
errorString line reason = [str|Error:
Line: #{line}
Reason: #{reason}
|]

parseLineToEvent :: Monad m => String -> m (Either String UnboundedEvent)
parseLineToEvent line = do
    res <-
        Stream.fromList line
            & Stream.foldMany (Fold.takeEndBy_ (== '/') Fold.toList)
            & Stream.toList
    case res of
        ["Start", windowId, tid, counter, val] ->
            case withParsed (UEvent (Start windowId)) tid counter val of
                Just val -> pure $ Right val
                Nothing -> pure $ Left $ errorString line "Not valid"
        ["End", windowId, tid, counter, val] ->
            case withParsed (UEvent (End windowId Nothing)) tid counter val of
                Just val -> pure $ Right val
                Nothing -> pure $ Left $ errorString line "Not valid"
        ["End", windowId, label, tid, counter, val] ->
            case withParsed (UEvent (End windowId (Just label))) tid counter val of
                Just val -> pure $ Right val
                Nothing -> pure $ Left $ errorString line "Not valid"
        _ -> pure $ Left $ errorString line "Chunks /= 4"

    where

    withParsed
        :: (ThreadId -> Counter -> Value -> UnboundedEvent)
        -> String
        -> String
        -> String
        -> Maybe UnboundedEvent
    withParsed func tid counter val =
        func <$> readMaybe tid <*> readMaybe counter <*> readMaybe val

parseInputToEventStream
    :: MonadIO m => Stream m (Array Word8) -> Stream m UnboundedEvent
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

boundEvents :: Monad m => Fold m UnboundedEvent (Maybe Event)
boundEvents = Fold step initial extract extract
    where
    initial = pure $ Partial (Nothing, Map.empty)

    alterFunc :: UnboundedEvent -> Maybe Value -> (Maybe Event, Maybe Value)
    alterFunc (UEvent (Start _) _ _ val) Nothing = (Nothing, Just val)
    alterFunc (UEvent (Start _) _ _ val) (Just _) = (Nothing, Just val)
    alterFunc (UEvent (End w Nothing) tid counter val) (Just prevVal) =
        ( Just (Event (EventId tid counter w) (val - prevVal))
        , Nothing
        )
    alterFunc (UEvent (End w (Just tag)) tid counter val) (Just prevVal) =
        ( Just (Event (EventId tid counter (w ++ ":" ++ tag)) (val - prevVal))
        , Just prevVal
        )
    alterFunc _ Nothing = (Nothing, Nothing)

    step (_, mp) uev@(UEvent b tid counter _) =
        pure $ Partial
             $ Map.alterF (alterFunc uev) (getWindowId b, tid, counter) mp

    extract (ev, _) = pure ev

statCollector :: Fold IO Event (Map EventId [(String, Int)])
statCollector =
    Fold.demuxToMap getEventId deriveFold

    where

    deriveFold ev = pure (Fold.lmap getEventVal stats)

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

printStatsMap
    :: (Show a, Show b, Show c, Ord a, Ord b, Ord c)
    => (EventId -> a)
    -> (EventId -> b)
    -> (EventId -> c)
    -> Map EventId [(String, Int)]
    -> IO ()
printStatsMap index1 index2 index3 mp =
    mapM_ printOneTable $ Map.toList $ anchorOnTidAndCounter mp

    where

    alterFunction v Nothing = Just [v]
    alterFunction v (Just v0) = Just (v:v0)

    foldingFunction mp ev v =
        Map.alter (alterFunction (index3 ev, v)) (index1 ev, index2 ev) mp

    anchorOnTidAndCounter mp =
        Map.foldlWithKey' foldingFunction Map.empty mp

    printOneTable ((i1, i2), rows) = do
        let i1Str = show i1
            i2Str = show i2
            headingL1 = [str|Index1: #{i1Str}|]
            headingL2 = [str|Index2: #{i2Str}|]
            divider = replicate (max (length headingL2) (length headingL1)) '-'
        putStrLn divider
        putStrLn headingL1
        putStrLn headingL2
        putStrLn divider
        putStrLn ""
        printTable
            $ (:) tableHeader
            $ map (\(i3, v) -> show i3 : map (pShowInt . snd) v) rows
        putStrLn ""

    pShowInt = Text.unpack . prettyI (Just ',')

    tableHeader = "Index3":statsLayout

--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

main :: IO ()
main = do
    statsMap <-
        Stream.unfold Handle.chunkReader stdin
            & parseInputToEventStream
            & Stream.scan boundEvents
            & Stream.catMaybes
            & Stream.fold statCollector
    (arg:[]) <- getArgs
    case arg of
        "Tag" ->
            printStatsMap
                (evTid)
                (evCounter)
                (evTag)
                statsMap
        "Counter" ->
            printStatsMap
                (evTid)
                (evTag)
                (evCounter)
                statsMap
        "ThreadId" ->
            printStatsMap
                (evCounter)
                (evTag)
                (evTid)
                statsMap
