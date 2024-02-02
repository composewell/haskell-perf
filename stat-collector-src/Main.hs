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
import Data.Char (ord)
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
import Control.Exception (catch, SomeException, displayException)

import qualified Data.Text as Text
import qualified Data.Map as Map
import qualified Streamly.Data.Fold as Fold
import qualified Streamly.Data.Stream.Prelude as Stream
import qualified Streamly.FileSystem.Handle as Handle
import qualified Streamly.Unicode.Stream as Unicode

--------------------------------------------------------------------------------
-- Utils
--------------------------------------------------------------------------------

double :: Int -> Double
double = fromIntegral

--------------------------------------------------------------------------------
-- Types
--------------------------------------------------------------------------------

data Boundary a b
    = Start a b
    | Record a b
    | Restart a b
    | End a b
    deriving (Read, Show, Ord, Eq)

getNameSpace :: Boundary NameSpace PointId -> NameSpace
getNameSpace (Start a _) = a
getNameSpace (Record a _) = a
getNameSpace (Restart a _) = a
getNameSpace (End a _) = a

data Counter
    = ThreadCpuTime
    | ProcessCpuTime
    | WallClockTime
    | Allocated
    | SchedOut
    deriving (Read, Show, Ord, Eq)

type NameSpace = String
type ModuleName = String
type LineNum = Int
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

data UnboundedEvent
    = UEvent (Boundary NameSpace PointId) ThreadId Counter Value
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
-- Start,<tag>,<tid>,<counterName>,<value>
-- Record,<tag>,<tid>,<counterName>,<value>
-- Restart,<tag>,<tid>,<counterName>,<value>
-- End,<tag>,<tid>,<counterName,<value>

-- Tag format:
-- NameSpace[ModuleName:LineNumber]

errorString :: String -> String -> String
errorString line reason = [str|Error:
Line: #{line}
Reason: #{reason}
|]

fIntegral :: (Monad m, Integral a) => Fold m Char a
fIntegral =
    Fold.foldl' (\b a -> 10 * b + ord1 a) 0
    where
    ord1 a =
        case ord a - 48 of
            x ->
                if x >= 0 || x <= 9
                then fromIntegral x
                else error "fIntegral: NaN"

fEventBoundary ::
    Monad m => Fold m Char (NameSpace -> PointId -> Boundary NameSpace PointId)
fEventBoundary =
    f <$> Fold.toList
    where
    f "Start" = Start
    f "Record" = Record
    f "Restart" = Restart
    f "End" = End
    f _ = error "fEventBoundary: undefined"

fCounterName :: Monad m => Fold m Char Counter
fCounterName = read <$> Fold.toList

fTag :: Monad m => Fold m Char (NameSpace, PointId)
fTag =
    (\a b c -> (a, (b, c)))
        <$> Fold.takeEndBy_ (== '[') Fold.toList
        <*> Fold.takeEndBy_ (== ':') Fold.toList
        <*> Fold.takeEndBy_ (== ']') fIntegral

fUnboundedEvent :: Monad m => Fold m Char UnboundedEvent
fUnboundedEvent =
    f
        <$> (Fold.takeEndBy_ (== ',') fEventBoundary)
        <*> (fTag <* Fold.one) -- fTag is a terminating fold
        <*> (Fold.takeEndBy_ (== ',') fIntegral)
        <*> (Fold.takeEndBy_ (== ',') fCounterName)
        <*> fIntegral

    where

    f a b c d e = UEvent (a (fst b) (snd b)) c d e

parseLineToEvent :: String -> IO (Either String UnboundedEvent)
parseLineToEvent line =
    catch
        (Right <$> Stream.fold fUnboundedEvent (Stream.fromList line))
        (\(e :: SomeException) ->
             pure (Left (errorString line (displayException e))))

parseInputToEventStream
    :: Stream IO (Array Word8) -> Stream IO UnboundedEvent
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

    alterFunc
        :: UnboundedEvent
        -> Maybe [(PointId, Value)]
        -> (Maybe Event, Maybe [(PointId, Value)])
    alterFunc (UEvent (Start _ point) _ _ val) Nothing =
        (Nothing, Just [(point, val)])
    alterFunc (UEvent (Start _ point) _ _ val) (Just xs) =
        (Nothing, Just ((point, val):xs))
    alterFunc (UEvent (End ns (md, ln)) tid counter val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), stk1) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    win = [str|#{ns}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just stk1
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc (UEvent (Restart ns point@(md, ln)) tid counter val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), stk1) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    win = [str|#{ns}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just ((point ,val):stk1)
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc (UEvent (Record ns (md, ln)) tid counter val) (Just stk) =
        case uncons stk of
            Just (((md1, ln1), prevVal), _) ->
                let lnStr = show ln
                    ln1Str = show ln1
                    win = [str|#{ns}[#{md1}:#{ln1Str}-#{md}:#{lnStr}]|]
                 in ( Just (Event (EventId tid counter win) (val - prevVal))
                    , Just stk
                    )
            Nothing -> error "boundEvents: Empty stack"
    alterFunc _ Nothing = (Nothing, Nothing)

    step (_, mp) uev@(UEvent b tid counter _) =
        pure $ Partial
             $ Map.alterF (alterFunc uev) (getNameSpace b, tid, counter) mp

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
