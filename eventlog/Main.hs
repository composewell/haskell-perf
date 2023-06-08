{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DoAndIfThenElse #-}
module Main (main) where

import Aggregator
  ( collectThreadCounter,
    translateThreadEvents,
  )
import Data.Either (isLeft)
import Data.Int (Int64)
import Data.IntMap (IntMap)
import Data.Ord (Down(..))
import Data.Map (Map)
import Data.Maybe (fromJust, isJust)
import Data.Text.Format.Numbers (prettyI)
import Data.Word (Word32, Word8)
import EventParser
  ( Counter (..),
    Location (..),
    Event (..),
    parseDataHeader,
    parseEvents,
    parseLogHeader,
  )
import Streamly.Data.Array (Array)
import Streamly.Data.Stream (Stream)
import Streamly.Data.StreamK (StreamK)
import Streamly.Internal.Data.Fold (Fold (..))
import System.Environment (getArgs)
import Text.Printf (printf)

import qualified Data.List as List
import qualified Data.Map as Map
import qualified Data.Text as Text
import qualified Streamly.Data.Fold as Fold
-- import qualified Streamly.Internal.Data.Fold as Fold (trace)
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.Data.StreamK as StreamK
import qualified Streamly.Data.Unfold as Unfold
import qualified Streamly.FileSystem.File as File
import qualified Streamly.Internal.Data.Fold.Container as Fold
    (demuxKvToMap, kvToMap)

-------------------------------------------------------------------------------
-- Utility functions, can go in streamly-core
-------------------------------------------------------------------------------

{-
{-# INLINE second #-}
second :: (Monad m, Eq a) => Fold m b c -> Fold m (a,b) (a,c)
second f = Fold.unzip (fmap fromJust Fold.the) f

{-# INLINE secondMaybe #-}
secondMaybe :: (Monad m, Eq a) =>
    Fold m b (Maybe c) -> Fold m (a,b) (Maybe (a,c))
secondMaybe f = fmap f1 (Fold.unzip (fmap fromJust Fold.the) f)

    where

    f1 (_, Nothing) = Nothing
    f1 (a, Just c) = Just (a, c)
-}

-------------------------------------------------------------------------------
-- Application
-------------------------------------------------------------------------------

double :: Int -> Double
double = fromIntegral

untilLeft :: Monad m => Fold m b1 b2 -> Fold m (Either (Maybe b1) b1) b2
untilLeft f =
      Fold.takeEndBy isLeft
    $ Fold.lmap (either id Just)
    $ Fold.catMaybes f

-- Statistics collection for each counter
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

{-# INLINE threadStats #-}
threadStats :: Fold IO (Either (Maybe Int64) Int64) [(String, Int)]
threadStats = untilLeft stats

{-# INLINE windowStats #-}
windowStats :: Fold IO (Either (Maybe Int64) Int64) [(String, Int)]
windowStats = Fold.many (untilLeft Fold.sum) stats

{-# INLINE toStats #-}
toStats ::
    Fold
        IO
        -- ((tid, window tag, counter), (location, value))
        ((Word32, String, Counter), (Location, Int64))
        -- Map (tid, window tag, counter) (Maybe [(stat name, value)])
        (Map (Word32, String, Counter) (Maybe [(String, Int)]))
toStats = Fold.demuxKvToMap (\k -> pure (f1 k))

    where

    f k1 collectStats =
          Fold.lmap (\x -> (k1, x))
        -- $ Fold.lmapM (\x -> print x >> pure x)
        $ Fold.scanMaybe collectThreadCounter
        $ Fold.postscan collectStats
        -- $ Fold.filter (\kv -> snd (snd kv !! 0) > 50000)
        -- $ Fold.trace print
        $ Fold.latest

    -- For the main thread
    f1 k1@(_, "default", _) = f k1 threadStats
    -- For windows inside the thread
    f1 k1@(_, _, _) = f k1 windowStats

{-# INLINE generateEvents #-}
generateEvents ::
       IntMap Int
    -> StreamK IO (Array Word8)
    -> Stream IO Event
generateEvents kv =
          Stream.unfoldMany Unfold.fromList
        . Stream.postscan translateThreadEvents
        -- . Stream.trace print
        . parseEvents kv

-- Ways to present:
-- For each thread rows of counters - cols of counter stats
-- For one counter rows of threads - cols of counter stats
-- For one counter rows of threads - cols of different runs

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
    maxLengths = List.foldl' (zipWith max) (head rowLengths) rowLengths
    separatorRow = map (\n -> replicate n '-') maxLengths
    fillRow r = zipWith (\n x -> fill n x) maxLengths r

getStatField :: String -> (k, [(String, Int)]) -> Maybe Int
getStatField x kv = List.lookup x $ snd kv

printWindowCounter ::
       [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 String
    -> (String, Counter)
    -> IO ()
printWindowCounter statsRaw tidMap (w, ctr) = do
    if w == "default"
        then
            putStrLn $ "Global thread wise stats for [" ++ show ctr ++ "]"
        else
            putStrLn
                $ "Window [" ++ w ++ "]" ++ " thread wise stats for ["
                    ++ show ctr ++ "]"
    let statsFiltered = filter select statsRaw
    let grandTotal =
            sum $ map (\x -> fromJust (getStatField "total" x)) statsFiltered
    let statsSorted = List.sortOn (Down . getStatField "total") statsFiltered
        statsString = map (\(k, v) -> (k, map toString v)) statsSorted
        allRows = map addTid statsString
        cnt = length allRows
        maxLines = 100
    printTable (header : take maxLines allRows)
    if cnt > maxLines
    then putStrLn $ "..." ++ show (cnt - maxLines) ++ " lines omitted ..."
    else return ()
    putStrLn $ "\nGrand total: " ++ Text.unpack (prettyI (Just ',') grandTotal)
    putStrLn ""

    where

    toString (k, v) = (k, Text.unpack $ prettyI (Just ',') v)
    header =
        ["tid"
        , "label"
        , "total"
        , "count"
        , "avg"
        , "minimum"
        , "maximum"
        , "stddev"
        ]
    addTid ((tid, _, _), v) =
        let r = Map.lookup tid tidMap
            lb = case r of
                    Just label -> label
                    Nothing -> "-"
         in printf "%d" tid : lb : map snd v
    select ((_, window, counter), _) = window == w && counter == ctr

windowLevelCounters :: [Counter]
windowLevelCounters =
    [ ProcessCPUTime
    , ProcessUserCPUTime
    , ProcessSystemCPUTime
    , GCCPUTime
    ]

-- XXX Instead of buffering the entire data and then process it, we can build
-- this report incrementally using a Fold, as the data comes. Different reports
-- can have different folds. That way we won't have to buffer the entire data
-- which could be extremely large. Also, we will be able to report online, in
-- real time. We will need a Map of windows, which will store a Map of tids
-- which will store a list or Map of counters.
printAllCounters ::
       [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 String
    -> [Counter]
    -> String
    -> IO ()
printAllCounters statsRaw tidMap ctrs w = do
    let
        windowTotals :: [((Word32, Counter), Int)]
        windowTotals = fmap toTotal $ filter selectWindow statsRaw

        tidList =
            fmap
                (\f -> fmap (fromIntegral . fst . fst) $ filter f windowTotals)
                (fmap selectCounter ctrs1)

    if null ctrs1
    then putStrLn "printAllCounters: no counters to print"
    else do
        -- Each tid must have all the counters present and in the same order.
        r <- Stream.fold Fold.the $ Stream.fromList tidList
        tids <-
            case r of
                Nothing -> error $ "A bug or something wrong with input data, "
                    ++ "Not all tids have all the counters present, "
                    ++ "or the order is wrong. ctrs: " ++ show ctrs1
                    ++ " tidList: " ++ show tidList
                Just x -> return x

        let
            allCounterTotals =
                fmap
                    (\f -> fmap snd $ filter f windowTotals)
                    (fmap selectCounter ctrs1)

            windowCounts = fmap toCounts $ filter selectWindow statsRaw
            oneCounterCounts = filter (selectCounter (head ctrs1)) windowCounts
            counts = fmap fromIntegral $ fmap snd $ oneCounterCounts

            allRows =
                  fmap (\(x:xs) ->
                          toString x
                        : getLabel (fromIntegral x)
                        : fmap toString xs
                       )
                $ List.sortOn (Down . (!! 2))
                $ List.transpose $ tids : counts : allCounterTotals

            -- Printing grand totals line at the bottom
            grandTotals = fmap sum allCounterTotals
            separator = replicate (length (head allRows)) " "
            summary =
                "-" : "-" : toString (sum counts) : fmap toString grandTotals

        if w == "default"
            then putStrLn $ "Global thread wise stat summary"
            else do
                putStrLn $ "Window [" ++ w ++ "]" ++ " thread wise stat summary"
                mapM_ (printWindowLevelCounter windowTotals)
                    [ProcessCPUTime, ProcessUserCPUTime, ProcessSystemCPUTime]
                if ":foreign" `List.isSuffixOf` w
                then return ()
                else do
                    putStrLn ""
                    let threadCPUTimeTotal =
                              sum
                            $ fmap snd
                            $ filter (selectCounter ThreadCPUTime) windowTotals
                    putStrLn $ "ThreadCPUTime:" ++ toString threadCPUTimeTotal
                    let gcCPUTime =
                              head
                            $ fmap snd
                            $ filter (selectCounter GCCPUTime) windowTotals
                    putStrLn $ "GcCPUTime:" ++ toString gcCPUTime
                    let processCPUTime =
                              head
                            $ fmap snd
                            $ filter (selectCounter ProcessCPUTime) windowTotals
                    let rtsCPUTime =
                            processCPUTime - gcCPUTime - threadCPUTimeTotal
                    putStrLn $ "RtsCPUTime:" ++ toString rtsCPUTime

        let cnt = length allRows
            maxLines = 100
        printTable ((header : take maxLines allRows) ++ [separator, summary])
        if cnt > maxLines
        then putStrLn $ "..." ++ show (cnt - maxLines) ++ " lines omitted ..."
        else return ()
        putStrLn ""

    where

    -- a "foreign" window does not have the allocated counter
    ctrs1 =
        if (":foreign" `List.isSuffixOf` w)
        then ctrs List.\\ [ThreadAllocated]
        else ctrs
    printWindowLevelCounter wt c = do
        -- Only one thread should have this
        let val = fmap snd $ filter (selectCounter c) wt
        case val of
            [] -> do
                {-
                -- a "foreign window does not have GCCPUTime
                putStrLn $ "printWindowLevelCounter: counter "
                        ++ show c ++ " not found in windowTotals"
                -}
                return ()
            [x] -> putStrLn $ show c ++ ": " ++ toString x
            _ -> error $ "Multiple values for counter " ++ show c
                        ++ " in window " ++ w

    toString = Text.unpack . prettyI (Just ',')
    header =
        ["tid"
        , "label"
        , "samples"
        ] ++ map show ctrs1
    selectWindow ((_, window, _), _) = window == w
    selectCounter c ((_, ctr), _) = ctr == c
    toTotal ((tid, _, ctr), v) = ((tid, ctr), fromJust $ List.lookup "total" v)
    toCounts ((tid, _, ctr), v) = ((tid, ctr), fromJust $ List.lookup "count" v)

    getLabel :: Word32 -> String
    getLabel tid =
        let r = Map.lookup tid tidMap
        in case r of
            Just label -> label
            Nothing -> "-"

-- XXX Are the events for a particular thread guaranteed to come in order. What
-- if a thread logged events to a particular capability buffer and then got
-- scheduled on another capability before its eventlog could be flushed from
-- the previous capability?
main :: IO ()
main = do
    (path:[]) <- getArgs
    let stream = File.readChunks path
    (kv, rest) <- parseLogHeader $ StreamK.fromStream stream
    -- putStrLn $ show kv
    events <- parseDataHeader rest
    (statsMap, tidMap) <-
        Stream.fold
            (Fold.partition toStats (Fold.kvToMap Fold.the))
            (fmap toEither $ generateEvents kv events)
    -- statsMap :: Map (tid, window tag, counter) (Maybe [(stat name, value)])
    -- putStrLn $ ppShow r
    -- putStrLn $ show tidMap
    let
        -- statsRaw :: [(tid, window tag, counter), [(stat name, value)]]
        statsRaw =
            -- TODO: get the sorting field from Config/CLI
              -- List.sortOn (getStatField "tid")
            -- TODO: get the threshold from Config/CLI
            -- $ filter (\x -> fromJust (getStatField "total" x) > 0)
              map (\(k, v) -> (k, filter (\(k1,_) -> k1 /= "latest") v))
            $ map (\(k, v) -> (k, fromJust v))
            $ filter (\(_, v) -> isJust v)
            $ Map.toList statsMap
    let windowCounterList =
              List.nub
            -- XXX Control this by config
            $ filter (\(w,_) -> not (":foreign" `List.isSuffixOf` w))
            $ filter (\(_,c) -> c `notElem` windowLevelCounters)
            $ map (\(_, window, counter) -> (window, counter))
            $ map fst statsRaw
    mapM_ checkLabel (Map.toList tidMap)

    putStrLn "--------------------------------------------------"
    putStrLn "Summary Stats"
    putStrLn "--------------------------------------------------"
    putStrLn ""

    -- TODO: filter the counters to be printed based on Config/CLI
    -- TODO: filter the windows or threads to be printed
    let ctrs = List.nub $ fmap snd windowCounterList
        wins = List.nub $ "default" : fmap fst windowCounterList
    let f = printAllCounters statsRaw (fmap fromJust tidMap) ctrs
     in mapM_ f wins

    putStrLn "--------------------------------------------------"
    putStrLn "Detailed Stats"
    putStrLn "--------------------------------------------------"
    putStrLn ""

    -- For each (window, counter) list all threads
    mapM_ (printWindowCounter statsRaw (fmap fromJust tidMap)) windowCounterList

    where

    toEither (CounterEvent tid tag ctr loc val) =
        Left ((tid, tag, ctr), (loc, fromIntegral val))
    toEither (LabelEvent tid label) = Right (tid, label)

    checkLabel (tid,Nothing) =
        error $ "Duplicate non-matching label events for thread: " ++ show tid
    checkLabel _ = pure ()
