{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main (main) where

--------------------------------------------------------------------------------
-- Imports
--------------------------------------------------------------------------------

-- import Control.Concurrent (threadDelay)
import Data.Int (Int32, Int64)
import System.Environment (getArgs)
import Data.Function ((&))
import Data.List (foldl', uncons)
import Data.Map (Map)
import Data.Maybe (fromJust)
import Data.Word (Word8)
import Streamly.Data.Array (Array)
import Streamly.Data.Fold (Fold)
import Streamly.Data.Stream (Stream)
import Streamly.Internal.Data.Fold (Fold(..), Step(..))
import Streamly.Unicode.String (str)
import System.IO (stdin)
import Data.Text.Format.Numbers (prettyI)

import qualified Data.Text as Text
import qualified Data.Map as Map
import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Parser as Parser
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.FileSystem.Handle as Handle
import qualified Streamly.Internal.Data.Array as Array
import qualified Streamly.Internal.Data.Binary.Parser as Parser

import Stat

--------------------------------------------------------------------------------
-- Utils
--------------------------------------------------------------------------------

double :: Int -> Double
double = fromIntegral

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

type ModuleName = String
type LineNum = Int32
type PointId = (ModuleName, LineNum)
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

-- Use ParserK here?
-- There are only 1 inner bind, ParserD should work fine.
metricParser :: Parser.Parser Word8 IO Metric
metricParser = do
    size64 <- Parser.int64le
    let size = fromIntegral size64 - 8
    fmap Array.deserialize $ Parser.takeEQ size (Array.unsafeCreateOf size)

-- parseMany can be implemented in a recursive manner using parserK
parseInputToEventStream
    :: Stream IO (Array Word8) -> Stream IO Metric
parseInputToEventStream inp =
    fmap f $ Stream.parseMany metricParser $ Array.concat inp
    where
    f (Left err) = error $ show err
    f (Right v) = v

--------------------------------------------------------------------------------
-- Processing stats
--------------------------------------------------------------------------------

boundEvents :: Monad m => Fold m Metric (Maybe Event)
boundEvents = Fold step initial extract extract
    where
    initial = pure $ Partial (Nothing, Map.empty)

    alterFunc
        :: Metric
        -> Maybe [(PointId, Value)]
        -> (Maybe Event, Maybe [(PointId, Value)])
    alterFunc (Metric _ _ m l _ Start val) Nothing =
        (Nothing, Just [((m, l), val)])
    alterFunc (Metric _ _ m l _ Start val) (Just xs) =
        (Nothing, Just (((m, l), val):xs))
    alterFunc (Metric tid ns md ln counter End val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), stk1) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    nsStr = show ns
                    win = [str|#{nsStr}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just stk1
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc (Metric tid ns md ln counter Restart val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), stk1) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    nsStr = show ns
                    win = [str|#{nsStr}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just (((md, ln) ,val):stk1)
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc (Metric tid ns md ln counter Record val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), _) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    nsStr = show ns
                    win = [str|#{nsStr}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just stk
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc _ Nothing = (Nothing, Nothing)

    step (_, mp) uev@(Metric tid ns _ _ counter _ _) =
        pure $ Partial
             $ Map.alterF (alterFunc uev) (ns, tid, counter) mp

    extract (ev, _) = pure ev

statCollector :: Fold IO Event (Map EventId [(String, Int)])
statCollector =
    Fold.demuxToMap getEventId deriveFold

    where

    deriveFold _ = pure (Fold.lmap getEventVal stats)

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
    :: (Show a, Show b, Show c, Ord a, Ord b)
    => (EventId -> a)
    -> (EventId -> b)
    -> (EventId -> c)
    -> Map EventId [(String, Int)]
    -> IO ()
printStatsMap index1 index2 index3 mp0 =
    mapM_ printOneTable $ Map.toList $ anchorOnTidAndCounter mp0

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
        _ -> error "Undefined arg."
