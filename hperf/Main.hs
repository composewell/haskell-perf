-- Until we fix the use of "head"
{-# OPTIONS_GHC "-Wno-x-partial" #-}

module Main (main) where

import Perf.Eventlog.Aggregate
  ( collectThreadCounter,
    translateThreadEvents,
  )
import Control.Monad (when)
import Data.Either (isLeft)
import Data.Int (Int64)
import Data.IntMap (IntMap)
import Data.Ord (Down(..))
import Data.Map (Map)
import Data.Maybe (fromJust, isJust)
import Data.Text.Format.Numbers (prettyI)
import Data.Word (Word32, Word8)
import Perf.Eventlog.Parser
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
import Streamly.Internal.Data.Fold (Fold (..), postscanlMaybe)
import Streamly.Data.Scanl (Scanl)
import qualified Streamly.Data.Scanl as Scanl
import qualified Streamly.Internal.Data.Scanl as Scanl (scanlMany, cumulativeScan)
import qualified Streamly.Statistics.Scanl as Stats
import Options.Applicative
import Text.Printf (printf)

import qualified Data.List as List
import qualified Data.Map as Map
import qualified Data.Text as Text
import qualified Streamly.Data.Fold as Fold
-- import qualified Streamly.Internal.Data.Fold as Fold (trace)
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.Data.StreamK as StreamK
import qualified Streamly.Data.Unfold as Unfold
import qualified Streamly.FileSystem.FileIO as File
import qualified Streamly.FileSystem.Path as Path
import qualified Streamly.Internal.Data.Fold as Fold
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

-- | Modify the input of a scan to accept an "Either" input, the modified scan
-- keeps consuming right inputs until a left input arrives, which terminates
-- the scan.
untilLeft :: Monad m => Scanl m b1 b2 -> Scanl m (Either (Maybe b1) b1) b2
untilLeft f =
      Scanl.takeEndBy isLeft
    $ Scanl.lmap (either id Just)
    $ Scanl.catMaybes f

{-
{-# INLINE combineStats #-}
combineStats :: Fold IO (String, Int) (Map String Int)
combineStats = Fold.demuxKvToMap (pure . f)

    where

    f k =
        case k of
            "latest" -> fmap fromJust Fold.latest
            "total" -> Fold.sum
            "count" -> Fold.sum
            "avg" -> fmap (const 0) Fold.drain
            "minimum" -> fmap fromJust Fold.minimum
            "maximum" -> fmap fromJust Fold.maximum
            "stddev" -> fmap (const 0) Fold.drain
            _ -> error "Unknown stat"

-- | Combine stats collected from different windows
{-# INLINE combineWindowStats #-}
combineWindowStats ::
    Fold IO
        ((Word32, String, Counter), (String, Int))
        (Map (Word32, String, Counter) (Map String Int))
combineWindowStats = Fold.kvToMap combineStats
-}

-- Statistics collection for each counter

{-# INLINE statScanner #-}
statScanner :: Scanl IO Int64 [(String, Int)]
statScanner =
      Scanl.lmap (fromIntegral :: Int64 -> Int)
    $ Scanl.distribute
        [ fmap (\x -> ("latest", fromJust x)) Scanl.latest
        , fmap (\x -> ("total", x)) Scanl.sum
        , fmap (\x -> ("count", x)) Scanl.length
        , fmap (\x -> ("avg", round x)) (Scanl.lmap double Scanl.mean)
        , fmap (\x -> ("minimum", fromJust x)) Scanl.minimum
        , fmap (\x -> ("maximum", fromJust x)) Scanl.maximum
        , fmap (\x -> ("stddev", round x)) (Scanl.lmap double (Scanl.cumulativeScan Stats.incrStdDev))
        ]

{-# INLINE threadStats #-}
threadStats :: Scanl IO (Either (Maybe Int64) Int64) [(String, Int)]
threadStats = untilLeft statScanner

{-# INLINE windowStats #-}
windowStats :: Scanl IO (Either (Maybe Int64) Int64) [(String, Int)]
windowStats = Scanl.scanlMany (untilLeft Scanl.sum) statScanner

{-# INLINE toStats #-}
toStats ::
    Fold
        IO
        -- ((tid, window tag, counter), (location, value))
        ((Word32, String, Counter), (Location, Int64))
        -- Map (tid, window tag, counter) (Maybe [(stat name, value)])
        (Map (Word32, String, Counter) (Maybe [(String, Int)]))
toStats = Fold.demuxKvToMap (pure . Just . f1)

    where

    f k1 collectStats =
          Fold.lmap (\x -> (k1, x))
        --  $ Fold.lmapM (\x -> print x >> pure x)
        $ postscanlMaybe collectThreadCounter
        $ Fold.postscanl collectStats
        --  $ Fold.filter (\kv -> snd (snd kv !! 0) > 50000)
        --  $ Fold.trace print
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
          Stream.unfoldEach Unfold.fromList
        . Stream.postscanl translateThreadEvents
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
        (hdr:rest) -> putStrLn $ unlines $ hdr:unwords separatorRow:rest

    where

    rowLengths = map (map length) rows -- [[Int]]
    maxLengths = List.foldl' (zipWith max) (head rowLengths) rowLengths
    separatorRow = map (\n -> replicate n '-') maxLengths
    fillRow r = zipWith (\n x -> fill n x) maxLengths r

getStatField :: String -> (k, [(String, Int)]) -> Maybe Int
getStatField x kv = List.lookup x $ snd kv

showCounterDetailsForWindow ::
       Int
    -> [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 String
    -> (String, Counter)
    -> IO ()
showCounterDetailsForWindow maxLines statsRaw tidMap (w, ctr) = do
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
        statsOrdered = fmap (\(k,v) -> (k, orderStats v)) statsSorted
        statsString = map (\(k, v) -> (k, map toString v)) statsOrdered
        allRows = map addTid statsString
        cnt = length allRows
    printTable (colHeaders : take maxLines allRows)
    if cnt > maxLines
    then putStrLn $ "..." ++ show (cnt - maxLines) ++ " lines omitted ..."
    else return ()
    putStrLn $ "\nGrand total: " ++ Text.unpack (prettyI (Just ',') grandTotal)
    putStrLn ""

    where

    toString (k, v) = (k, Text.unpack $ prettyI (Just ',') v)
    statNames =
        [ "total"
        , "count"
        , "avg"
        , "minimum"
        , "maximum"
        , "stddev"
        ]
    colHeaders =
        ["tid"
        , "label"
        ] ++ statNames

    -- put in a fixed order so that code changes do not affect reporting
    orderStats xs =
        [ ("total", fromJust (lookup "total" xs))
        , ("count", fromJust (lookup "count" xs))
        , ("avg", fromJust (lookup "avg" xs))
        , ("minimum", fromJust (lookup "minimum" xs))
        , ("maximum", fromJust (lookup "maximum" xs))
        , ("stddev", fromJust (lookup "stddev" xs))
        ]

    addTid ((tid, _, _), v) =
        let r = Map.lookup tid tidMap
            lb = case r of
                    Just label -> label
                    Nothing -> "-"
         in printf "%d" tid : lb : map snd v
    select ((_, window, counter), _) = window == w && counter == ctr

-- XXX we can only use the Process level counters to report the entire
-- program's CPU time including all threads in the saummary data. Reporting
-- this in the Windows times is not useful. We can use the OS thread's
-- ThreadCPUTime only if there is only one Haskell thread running.
processLevelCounters :: [Counter]
processLevelCounters =
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

-- | Print a table for the given window, one row per thread that ever entered
-- the window,  listing accumulated value for each
-- counter.
showAllCountersForWindow ::
       Int
    -> Bool
    -> [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 String
    -> [Counter]
    -> String
    -> IO ()
showAllCountersForWindow maxLines concurrent stats tidMap ctrs w = do
    let
        windowTotals :: [((Word32, Counter), Int)]
        windowTotals = fmap toTotal $ filter selectWindow stats

        tidList =
            fmap
                (\f -> fmap (fromIntegral . fst . fst) $ filter f windowTotals)
                (fmap selectCounter ctrs1)

    if null ctrs1
    then putStrLn "showAllCountersForWindow: no counters to print"
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

            windowCounts = fmap toCounts $ filter selectWindow stats
            oneCounterCounts = filter (selectCounter (head ctrs1)) windowCounts
            counts = fmap snd $ oneCounterCounts

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
                -- When collapsing windows, if the windows are concurrent then
                -- we cannot combine the window level counters.
                putStrLn $ "Window [" ++ w ++ "]" ++ " thread wise stat summary"
                when (not concurrent) $ do
                    putStrLn ""
                    mapM_ (printProcessLevelCounter windowTotals)
                        [ProcessCPUTime]
                    mapM_ (\x -> putStr " " >> printProcessLevelCounter windowTotals x)
                        [ProcessUserCPUTime, ProcessSystemCPUTime]
                    if ":foreign" `List.isSuffixOf` w
                    then return ()
                    else do
                        putStrLn ""
                        mapM_ (printProcessLevelCounter windowTotals)
                            [ProcessCPUTime]
                        let threadCPUTimeTotal =
                                getProcessLevelCounter sum windowTotals ThreadCPUTime
                        putStrLn $ " ThreadCPUTime:" ++ toString threadCPUTimeTotal
                        let gcCPUTime =
                                getProcessLevelCounter head windowTotals GCCPUTime
                        putStrLn $ " GcCPUTime:" ++ toString gcCPUTime
                        let processCPUTime =
                                getProcessLevelCounter head windowTotals ProcessCPUTime
                        let rtsCPUTime =
                                processCPUTime - gcCPUTime - threadCPUTimeTotal
                        putStrLn $ " RtsCPUTime(*):" ++ toString rtsCPUTime

        let cnt = length allRows
        putStrLn ""
        printTable ((colHeaders : take maxLines allRows) ++ [separator, summary])
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
    getProcessLevelCounter f wt c = f $ fmap snd $ filter (selectCounter c) wt
    printProcessLevelCounter wt c = do
        -- Only one thread should have this
        let val = fmap snd $ filter (selectCounter c) wt
        case val of
            [] -> do
                {-
                -- a "foreign window does not have GCCPUTime
                putStrLn $ "printProcessLevelCounter: counter "
                        ++ show c ++ " not found in windowTotals"
                -}
                return ()
            [x] -> putStrLn $ show c ++ ": " ++ toString x
            _ -> error $ "Multiple values for counter " ++ show c
                        ++ " in window " ++ w

    toString = Text.unpack . prettyI (Just ',')
    colHeaders =
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

-- A window tag is of the format "tid:name", extract the "tid" from this.
getTidFromWindowTag :: String -> Maybe Word32
getTidFromWindowTag w =
    let (tid, r) = span (/= ':') w
    in if null r then Nothing else Just (read tid :: Word32)

-- | Combine stats from all windows with the same name but different thread-id.
-- This just gives us more samples for the same window, we are saying we don't
-- care in which thread's context the window code ran, just give us the timings
-- in the context of any thread.
--
-- We just erase the thread id from the window name and change it to 0, so that
-- all the threads now have the same thread-id 0.
foldWindowThreads ::
       [((Word32, String, Counter), [(String, Int)])]
    -> IO [((Word32, String, Counter), [(String, Int)])]
foldWindowThreads stats = do
    let changeWindowTidZero w =
            let (_, r) = span (/= ':') w
            in if null r then w else '0':r

        collapseTid ((tid, tag, ctr), v) =
            ((tid, changeWindowTidZero tag, ctr), v)

        matching ((tid, tag, _), _) =
            case getTidFromWindowTag tag of
                Nothing -> False
                Just x -> x == tid

    return $ fmap collapseTid $ filter matching stats

-------------------------------------------------------------------------------
-- CLI
-------------------------------------------------------------------------------

data ListSubCmd
    = ListCounters
    | ListWindows FilePath
    | ListThreads FilePath

data AnalyseConfig = AnalyseConfig
    { analyseFile :: FilePath
    , analyseFoldThreads :: Bool
    , analyseMaxLines :: Int
    , analyseDetailed :: Bool
    }

data Command
    = CmdList ListSubCmd
    | CmdAnalyse AnalyseConfig

listSubCmdParser :: Parser ListSubCmd
listSubCmdParser = subparser
    (  command "counters"
            (info (pure ListCounters)
                (progDesc "List all available performance counters"))
    <> command "windows"
            (info
                (ListWindows <$> argument str
                    (  metavar "EVENTLOG-FILE"
                    <> help "Path to the GHC eventlog file"
                    ))
                (progDesc "List all windows found in the eventlog file"))
    <> command "threads"
            (info
                (ListThreads <$> argument str
                    (  metavar "EVENTLOG-FILE"
                    <> help "Path to the GHC eventlog file"
                    ))
                (progDesc "List all threads found in the eventlog file"))
    )

analyseConfigParser :: Parser AnalyseConfig
analyseConfigParser = AnalyseConfig
    <$> argument str
            (  metavar "EVENTLOG-FILE"
            <> help "Path to the GHC eventlog file to analyse"
            )
    <*> switch
            (  long "fold-threads"
            <> short 'f'
            <> help "Instead of per thread windows, show all threads combined per window"
            )
    <*> option auto
            (  long "max-lines"
            <> short 'n'
            <> metavar "N"
            <> value 10
            <> showDefault
            <> help "Maximum number of thread rows to print per table"
            )
    -- By default prints one table per window consisting of all counters
    -- summary for that window.
    <*> switch
            (  long "detailed"
            <> short 'd'
            <> help "Print details for each counter for each window"
            )

commandParser :: Parser Command
commandParser = subparser
    (  command "list"
            (info (CmdList <$> listSubCmdParser)
                (progDesc "List available counters or windows"))
    <> command "analyse"
            (info (CmdAnalyse <$> analyseConfigParser)
                (progDesc "Analyse a GHC eventlog file"))
    )

optsInfo :: ParserInfo Command
optsInfo = info (commandParser <**> helper)
    (  fullDesc
    <> progDesc ("Analyse CPU cost, heap allocations, and Linux perf event "
             ++ "counters for Haskell threads and user-defined code windows.")
    <> header "hperf - Haskell performance analysis tool"
    )

-------------------------------------------------------------------------------
-- Report
-------------------------------------------------------------------------------

postProcess ::
    -- ((tid, window-tag, counter), [(stat-name, value)])
    -- XXX Instead of Maybe here can we use an empty list?
       [ ((Word32, String, Counter), Maybe [(String, Int)]) ]
    -> [ ((Word32, String, Counter), [(String, Int)]) ]
postProcess =
    -- TODO: get the sorting field from Config/CLI
      -- List.sortOn (getStatField "tid")
    -- TODO: get the threshold from Config/CLI
    --  $ filter (\x -> fromJust (getStatField "total" x) > 0)
      fmap (\(k, v) -> (k, filter (\(k1,_) -> k1 /= "latest") v))
    . fmap (\(k, v) -> (k, fromJust v))
    . filter (\(_, v) -> isJust v)

loadStats ::
       FilePath
    -> IO
    --  ( Map (tid, window tag, counter) (Maybe [(stat name, value)])
        ( Map (Word32, String, Counter) (Maybe [(String, Int)])
    --  , Map tid (Maybe thread-name))
        , Map Word32 (Maybe String)
        )
loadStats path = do
    let chunks = File.readChunks (Path.fromString_ path)
    (kv, rest) <- parseLogHeader $ StreamK.fromStream chunks
    -- putStrLn $ show kv
    eventChunks <- parseDataHeader rest
    let tagged = fmap tagEither $ generateEvents kv eventChunks
        toLabels = Fold.kvToMap Fold.the
        collector = Fold.partition toStats toLabels
    Stream.fold collector tagged

    where

    tagEither (CounterEvent tid tag ctr loc val) =
        Left ((tid, tag, ctr), (loc, fromIntegral val))
    tagEither (LabelEvent tid label) = Right (tid, label)

getStatMapTidMap ::
       FilePath
    -> IO
        ( [((Word32, String, Counter), [(String, Int)])]
        , Map Word32 (Maybe String)
        )
getStatMapTidMap path = do
    (statsMap, tidMap) <- loadStats path
    -- statsMap :: Map (tid, window tag, counter) (Maybe [(stat name, value)])
    -- putStrLn $ ppShow r
    -- putStrLn $ show tidMap
    -- rawStats :: [(tid, window tag, counter), [(stat name, value)]]
    let statsMap1 = postProcess $ Map.toList statsMap
    return (statsMap1, tidMap)

getWindowCounterList ::
    [((Word32, String, Counter), [(String, Int)])] -> [(String, Counter)]
getWindowCounterList stats =
      List.nub
    $ filter (\(_,c) -> c `notElem` processLevelCounters)
    $ map (\(_, window, counter) -> (window, counter))
    $ map fst stats

getAllStats ::
    Bool
    -> FilePath
    -> IO
        -- statsMap :: Map (tid, window tag, counter) [(stat name, value)]
        ( [((Word32, String, Counter), [(String, Int)])]
        , Map Word32 (Maybe String)
        , [((Word32, String, Counter), [(String, Int)])]
        , [(String, Counter)])
getAllStats mergeThreads path = do
    (statMap, tidMap) <- getStatMapTidMap path
    -- XXX Take a window argument from config/CLI and rename only specific
    -- windows or all windows depending on that.
    -- XXX we should rather use the mergeThreads flag whereever we need this.
    -- this makes understanding code difficult.
    foldedStats <-
        if mergeThreads
        then foldWindowThreads statMap
        else return statMap
    let windowCounterList = getWindowCounterList foldedStats
    return (statMap, tidMap, foldedStats, windowCounterList)

validateLabels :: Map Word32 (Maybe String) -> IO ()
validateLabels tidMap = mapM_ checkLabel (Map.toList tidMap)

    where

    checkLabel (tid, Nothing) =
        error $ "Duplicate non-matching label events for thread: " ++ show tid
    checkLabel _ = pure ()

showAllCountersPerWindow ::
       Int
    -> Bool
    -> [((Word32, String, Counter), [(String, Int)])]
    -> [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 (Maybe String)
    -> [(String, Counter)]
    -> IO ()
showAllCountersPerWindow maxLines foldWindowStats statsRaw statsFlattened tidMap windowCounterList = do
    -- TODO: filter the counters to be printed based on Config/CLI
    -- TODO: filter the windows or threads to be printed
    let ctrs = List.nub $ fmap snd windowCounterList
        wins = List.nub $ "default" : fmap fst windowCounterList
        -- hack - currently we do not compute avg and stddev in flattened
        getStats w = if w == "default" then statsRaw else statsFlattened

    let f w =
            showAllCountersForWindow
                maxLines foldWindowStats (getStats w) (fmap fromJust tidMap) ctrs w
     in mapM_ f wins

showOneCounterPerWindow ::
       Int
    -> [((Word32, String, Counter), [(String, Int)])]
    -> [((Word32, String, Counter), [(String, Int)])]
    -> Map Word32 (Maybe String)
    -> [(String, Counter)]
    -> IO ()
showOneCounterPerWindow maxLines rawStats foldedStats tidMap windowCounterList = do
    -- XXX TODO need to print summary info as well in this.

    -- hack - currently we do not compute avg and stddev in flattened
    let getStats w = if w == "default" then rawStats else foldedStats
    -- For each (window, counter) list all threads
    let f (w,c) = showCounterDetailsForWindow maxLines (getStats w) (fmap fromJust tidMap) (w,c)
     in mapM_ f windowCounterList

-------------------------------------------------------------------------------
-- Entry point
-------------------------------------------------------------------------------

-- XXX Add two different commands "hperf eventlog" and "hperf metrics", one for
-- eventlog analysis and the other for metrics collected by other methods.
--
-- XXX Are the events for a particular thread guaranteed to come in order. What
-- if a thread logged events to a particular capability buffer and then got
-- scheduled on another capability before its eventlog could be flushed from
-- the previous capability?
main :: IO ()
main = do
    cmd <- execParser optsInfo
    case cmd of
        CmdList ListCounters -> do
            putStrLn "Supported counters:"
            mapM_ (putStrLn . ("  " ++) . show) [minBound..maxBound :: Counter]
        CmdList (ListWindows path) -> do
            (_, _, _, windowCounterList) <- getAllStats True path
            let wins = List.nub $ "default" : fmap fst windowCounterList
            putStrLn "Available windows:"
            mapM_ (putStrLn . ("  " ++)) wins
        CmdList (ListThreads path) -> do
            -- XXX does not print all threads
            (_, tidMap) <- loadStats path
            validateLabels tidMap
            putStrLn "Threads (id, label):"
            mapM_ (\(tid, mlabel) ->
                putStrLn $ "  " ++ show tid ++ ", " ++ maybe "-" id mlabel)
                (Map.toList tidMap)
        CmdAnalyse AnalyseConfig
            -- XXX the mergeThreads should just print a single table with one
            -- line per window in the summary view.
            -- XXX We should have the following views:
            -- 1. overall summary, "default" window only (default)
            -- 2. thread-agnostic, one line per window, all the counter
            -- averages as columns. --windows
            -- 3. thread-aware, one table per window, one row per thread, all
            -- the counters as columns. --threads
            -- 4. thread-aware, one table per window per counter, one row per
            -- thread, counter attributes as columns. --counters
            { analyseFile = path
            , analyseFoldThreads = mergeThreads
            , analyseMaxLines = maxLines
            , analyseDetailed = detailed
            } -> do
            (statsMap, tidMap, foldedStats, windowCounterList) <- getAllStats mergeThreads path
            -- XXX Control this by config
            let windowCounterList1 =
                    filter (\(w,_) -> not (":foreign" `List.isSuffixOf` w))
                        windowCounterList
            validateLabels tidMap
            if detailed
            then showOneCounterPerWindow
                    maxLines statsMap foldedStats tidMap windowCounterList1
            else showAllCountersPerWindow
                    maxLines
                    mergeThreads statsMap foldedStats tidMap windowCounterList1
