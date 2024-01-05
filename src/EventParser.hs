{-# LANGUAGE CPP #-}
module EventParser
    (
      parseLogHeader
    , parseDataHeader
    , parseEvents
    , parsePrimOpEvents
    , Counter (..)
    , Location (..)
    , Event (..)
    )
where

-- import Debug.Trace
import Data.Char (ord, chr)
import Data.Function ((&))
import Data.Functor.Identity (Identity(..))
import Data.IntMap (IntMap)
import Data.Word (Word8, Word16, Word32, Word64)
import Streamly.Data.Array (Array)
import Streamly.Data.Parser (Parser)
import Streamly.Data.ParserK (ParserK)
import Streamly.Data.Stream (Stream)
import Streamly.Data.StreamK (StreamK)
import Streamly.Internal.Data.Binary.Parser (int16be, word16be, word32be, word64be)

import qualified Data.IntMap as Map
import qualified Streamly.Data.Array as Array
import qualified Streamly.Data.Fold as Fold
-- import qualified Streamly.Internal.Data.Fold.Container as Fold (toContainer)
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.Data.StreamK as StreamK
import qualified Streamly.Data.Parser as Parser
import qualified Streamly.Data.ParserK as ParserK
import qualified Streamly.Unicode.Stream as Unicode

{-
#define EVENT_HEADER_BEGIN    0x68647262
#define EVENT_HEADER_END      0x68647265
#define EVENT_DATA_BEGIN      0x64617462
#define EVENT_DATA_END        0xffff
#define EVENT_HET_BEGIN       0x68657462
#define EVENT_HET_END         0x68657465
#define EVENT_ET_BEGIN        0x65746200
#define EVENT_ET_END          0x65746500
-}

w8 :: String -> [Word8]
w8 = map (fromIntegral . ord)

---------------------
-- Event Type
---------------------

hetBegin :: [Word8]
hetBegin = w8 "hetb"

hetEnd :: [Word8]
hetEnd = w8 "hete"

eventTypeBegin :: [Word8]
eventTypeBegin = w8 "etb\0"

eventTypeEnd :: [Word8]
eventTypeEnd = w8 "ete\0"

--  *       EVENT_ET_BEGIN
--  *       Word16         -- unique identifier for this event
--  *       Int16          -- >=0  size of the event in bytes (minus the header)
--  *                      -- -1   variable size
--  *       Word32         -- length of the next field in bytes
--  *       Word8*         -- string describing the event
--  *       Word32         -- length of the next field in bytes
--  *       Word8*         -- extra info (for future extensions)
--  *       EVENT_ET_END
-- (event Id, event size)
eventType :: ParserK (Array Word8) IO (Int, Int)
eventType = do
    _ <- ParserK.adaptC (Parser.listEq eventTypeBegin)
    eventId <- ParserK.adaptC word16be
    -- ParserK.fromEffect $ print $ "Event id =" ++ show eventId
    eventSize <- ParserK.adaptC int16be
    -- ParserK.fromEffect $ print $ "Event size =" ++ show eventSize
    descLen <- ParserK.adaptC word32be
    _desc <- ParserK.adaptC (Parser.takeEQ (fromIntegral descLen) Fold.toList)
    -- ParserK.fromEffect $ putStrLn (map (chr . fromIntegral) desc)
    infoLen <- ParserK.adaptC word32be
    -- ParserK.fromEffect $ print $ "info len =" ++ show infoLen
    ParserK.adaptC (Parser.takeEQ (fromIntegral infoLen) Fold.drain)
    _ <- ParserK.adaptC (Parser.listEq eventTypeEnd)
    return (fromIntegral eventId, fromIntegral eventSize)

---------------------
-- Header
---------------------

headerBegin :: [Word8]
headerBegin = w8 "hdrb"

headerEnd :: [Word8]
headerEnd = w8 "hdre"

{-
{-# INLINE kvToMap #-}
kvToMap :: (Monad m) => Fold m a b -> Fold m (Int, a) (IntMap b)
kvToMap = Fold.toContainer fst . Fold.lmap snd
-}

parseEventTypes ::
       Map.IntMap Int
    -> StreamK IO (Array Word8)
    -> IO (Map.IntMap Int, StreamK IO (Array Word8))
parseEventTypes kv stream = do
    -- putStrLn "parsing eventType"
    (r, rest) <- StreamK.parseBreakChunks eventType stream
    case r of
        -- XXX When the parser fails the original stream should not change, we
        -- should be able to use "rest" here.
        Left _ -> return (kv, stream)
        Right (eventId, eventSize) -> do
            let kv1 = Map.insert eventId  eventSize kv
            parseEventTypes kv1 rest

headerPre :: ParserK (Array Word8) IO ()
headerPre = do
    _ <- ParserK.adaptC (Parser.listEq headerBegin)
    _ <- ParserK.adaptC (Parser.listEq hetBegin)
    return ()

headerPost :: ParserK (Array Word8) IO ()
headerPost = do
    _ <- ParserK.adaptC (Parser.listEq hetEnd)
    _ <- ParserK.adaptC (Parser.listEq headerEnd)
    return ()

{-
header :: ParserK Word8 IO (Map.IntMap Int)
header = do
    ParserK.fromParser (Parser.listEq headerBegin)
    ParserK.fromParser (Parser.listEq hetBegin)
    -- r <- ParserK.fromParser (Parser.many eventType (kvToMap Fold.one))
    r <- parseEventTypes
    ParserK.fromParser (Parser.listEq hetEnd)
    ParserK.fromParser (Parser.listEq headerEnd)
    return r
-}

-- XXX We can create a Parser monad based on parseBreak to compose parsers in a
-- chain, we can also create an Alternative instance for that.
parseLogHeader ::
    StreamK IO (Array Word8) -> IO (Map.IntMap Int, StreamK IO (Array Word8))
parseLogHeader stream = do
    (res, rest) <- StreamK.parseBreakChunks headerPre stream
    case res of
        Left err -> fail $ show err
        Right () -> do
            (kv, rest1) <- parseEventTypes Map.empty rest
            (res1, rest2) <- StreamK.parseBreakChunks headerPost rest1
            case res1 of
                Left err -> putStrLn "header end not found" >> (fail $ show err)
                Right () -> return (kv, rest2)

---------------------
-- Parse events
---------------------

dataBegin :: [Word8]
dataBegin = w8 "datb"

-- dataEnd :: [Word8]
-- dataEnd = [0xff, 0xff]

parseDataHeader :: StreamK IO (Array Word8) -> IO (StreamK IO (Array Word8))
parseDataHeader stream = do
    let p = ParserK.adaptC (Parser.listEq dataBegin)
    (res, rest) <- StreamK.parseBreakChunks p stream
    case res of
        Left err -> fail $ show err
        Right _ -> return rest

#define EVENT_RUN_THREAD           1 /* (thread)               */
#define EVENT_STOP_THREAD          2 /* (thread, status, blockinfo) */
#define EVENT_USER_MSG            19 /* (message ...)          */
#define EVENT_THREAD_LABEL        44 /* (thread, name_string)  */

#define EVENT_PRE_THREAD_CLOCK           300
#define EVENT_POST_THREAD_CLOCK          301
#define EVENT_PRE_THREAD_PAGE_FAULTS     302
#define EVENT_POST_THREAD_PAGE_FAULTS    303
#define EVENT_PRE_THREAD_CTX_SWITCHES    304
#define EVENT_POST_THREAD_CTX_SWITCHES   305
#define EVENT_PRE_THREAD_ALLOCATED       306
#define EVENT_POST_THREAD_ALLOCATED      307
#define EVENT_PRE_HW_CACHE_L1I           308
#define EVENT_POST_HW_CACHE_L1I          309
#define EVENT_PRE_HW_CACHE_L1I_MISS      310
#define EVENT_POST_HW_CACHE_L1I_MISS     311
#define EVENT_PRE_HW_CACHE_L1D           312
#define EVENT_POST_HW_CACHE_L1D          313
#define EVENT_PRE_HW_CACHE_L1D_MISS      314
#define EVENT_POST_HW_CACHE_L1D_MISS     315
#define EVENT_PRE_HW_CACHE_MISSES        316
#define EVENT_POST_HW_CACHE_MISSES       317
#define EVENT_PRE_HW_INSTRUCTIONS        318
#define EVENT_POST_HW_INSTRUCTIONS       319
#define EVENT_PRE_HW_BRANCH_MISSES       320
#define EVENT_POST_HW_BRANCH_MISSES      321
#define EVENT_PRE_THREAD_CPU_MIGRATIONS  322
#define EVENT_POST_THREAD_CPU_MIGRATIONS 323
#define EVENT_PRE_PROCESS_CPU_TIME       324
#define EVENT_PRE_FOREIGN_CPU_TIME       325
#define EVENT_PRE_GC_CPU_TIME            326
#define EVENT_PRE_USER_CPU_TIME          327
#define EVENT_PRE_SYSTEM_CPU_TIME        328

-- XXX We attach a user event to a thread by looking at the previous thread
-- start event. But when there are multiple capabilities this may not be
-- possible? We need to use the thread-id on the same capability as the user
-- event. Or we can emit the tid in the user event. How does ghc-events-analyze
-- handle this? or the user event can log the thread-id as part of the tag.

data Counter =
      ThreadCPUTime
    | ThreadCtxVoluntary
    | ThreadPageFaultMinor
    | L1iCacheHit
    | L1iCacheMiss
    | L1dCacheHit
    | L1dCacheMiss
    | ThreadCPUMigrations
    | BranchMisses
    | Instructions
    | LastLevelCacheMisses

    -- Keep after this line, the counters that are used only for Haskell code
    -- windows and not for "foreign" windows. We check < ThreadAllocated in
    -- the code.
    | ThreadAllocated
    | ProcessCPUTime
    | ForeignCPUTime
    | GCCPUTime
    | ProcessUserCPUTime
    | ProcessSystemCPUTime

    deriving (Show, Read, Eq, Ord)

-- data Location = Enter | Exit | Resume | Suspend deriving Show
data Location = Resume | Suspend | Exit | Purge deriving (Show, Read)

-- Event tid window counter start/stop value
data Event =
     CounterEvent Word32 String Counter Location Word64
   | LabelEvent Word32 String
   deriving Show

eventToCounter :: Word16 -> Maybe (Counter, Location)
eventToCounter ev =
    case ev of
        EVENT_PRE_THREAD_CLOCK -> Just (ThreadCPUTime, Resume)
        EVENT_PRE_THREAD_ALLOCATED -> Just (ThreadAllocated, Resume)
        EVENT_PRE_HW_CACHE_L1I -> Just (L1iCacheHit, Resume)
        EVENT_PRE_HW_CACHE_L1I_MISS -> Just (L1iCacheMiss, Resume)
        EVENT_PRE_HW_CACHE_L1D -> Just (L1dCacheHit, Resume)
        EVENT_PRE_HW_CACHE_L1D_MISS -> Just (L1dCacheMiss, Resume)
        EVENT_PRE_THREAD_PAGE_FAULTS -> Just (ThreadPageFaultMinor, Resume)
        EVENT_PRE_THREAD_CTX_SWITCHES -> Just (ThreadCtxVoluntary, Resume)
        EVENT_PRE_HW_CACHE_MISSES -> Just (LastLevelCacheMisses, Resume)
        EVENT_PRE_HW_INSTRUCTIONS -> Just (Instructions, Resume)
        EVENT_PRE_HW_BRANCH_MISSES -> Just (BranchMisses, Resume)
        EVENT_PRE_THREAD_CPU_MIGRATIONS -> Just (ThreadCPUMigrations, Resume)
        EVENT_PRE_PROCESS_CPU_TIME -> Just (ProcessCPUTime, Resume)
        EVENT_PRE_FOREIGN_CPU_TIME -> Just (ForeignCPUTime, Resume)
        EVENT_PRE_GC_CPU_TIME -> Just (GCCPUTime, Resume)
        EVENT_PRE_USER_CPU_TIME -> Just (ProcessUserCPUTime, Resume)
        EVENT_PRE_SYSTEM_CPU_TIME -> Just (ProcessSystemCPUTime, Resume)

        EVENT_POST_THREAD_CLOCK -> Just (ThreadCPUTime, Suspend)
        EVENT_POST_THREAD_ALLOCATED -> Just (ThreadAllocated, Suspend)
        EVENT_POST_HW_CACHE_L1I -> Just (L1iCacheHit, Suspend)
        EVENT_POST_HW_CACHE_L1I_MISS -> Just (L1iCacheMiss, Suspend)
        EVENT_POST_HW_CACHE_L1D -> Just (L1dCacheHit, Suspend)
        EVENT_POST_HW_CACHE_L1D_MISS -> Just (L1dCacheMiss, Suspend)
        EVENT_POST_THREAD_PAGE_FAULTS -> Just (ThreadPageFaultMinor, Suspend)
        EVENT_POST_THREAD_CTX_SWITCHES -> Just (ThreadCtxVoluntary, Suspend)
        EVENT_POST_HW_CACHE_MISSES -> Just (LastLevelCacheMisses, Suspend)
        EVENT_POST_HW_INSTRUCTIONS -> Just (Instructions, Suspend)
        EVENT_POST_HW_BRANCH_MISSES -> Just (BranchMisses, Suspend)
        EVENT_POST_THREAD_CPU_MIGRATIONS -> Just (ThreadCPUMigrations, Suspend)
        _ -> Nothing

-- XXX We can ensure that the required size of data is compacted into the next
-- array and then use a more efficient single array parser which only passes
-- around pos instead of the array.
--
-- XXX We should use a parseManyK

--  * Event :
--  *       Word16         -- event_type
--  *       Word64         -- time (nanosecs)
--  *       [Word16]       -- length of the rest (for variable-sized events only)
--  *       ... extra event-specific info ...
{-# INLINE event #-}
event :: IntMap Int -> Parser Word8 IO (Maybe Event)
event kv = do
    eventId <- word16be
    ts <- word64be
    -- XXX This does not get executed if we error out
    -- trace ("event = " ++ show eventId ++ " ts = " ++ show ts) (return ())
    case eventId of
        {-
        EVENT_RUN_THREAD -> do
            tid <- word32be
            return $ StartThreadCPUTimeWall tid ts
        EVENT_STOP_THREAD -> do
            tid <- word32be
            -- XXX Stop thread has two more fields
            return $ StopThreadCPUTimeWall tid ts
        -}
        EVENT_USER_MSG -> do
            len <- word16be
            tid <- word32be
            ctrType <- word16be
            msg <- Parser.takeEQ
                        (fromIntegral (len - 6))
                        (Fold.lmap (chr . fromIntegral) Fold.toList)
            let (loc, rest) = span (/= ':') msg
                tag = show tid ++ ":" ++ drop 1 rest
            -- Parser.fromEffect $ putStrLn $ "tid = " ++ show tid1 ++ " loc = " ++ loc ++ " tag = " ++ tag
            let counterId =
                    case eventToCounter ctrType of
                        Just ctr -> fst ctr
                        _ -> error "Invalid event in user window"
            -- Tag the windows with the creating thread so that if the same
            -- window is entered from multiple threads we can account it
            -- separately and we do not consider it a duplicate window event.
            case loc of
                "START" -> do
                    let ev = CounterEvent tid tag counterId Resume ts
                    -- trace ("Parsed: = " ++ show ev) (return ())
                    return $ Just ev
                "END" -> do
                    let ev = CounterEvent tid tag counterId Suspend ts
                    -- trace ("Parsed: = " ++ show ev) (return ())
                    return $ Just ev
                _ -> error $ "Invalid window location tag: " ++ loc
        EVENT_THREAD_LABEL -> do
            len <- word16be
            tid <- word32be
            label <- Parser.takeEQ
                        (fromIntegral (len - 4))
                        (Fold.lmap (chr . fromIntegral) Fold.toList)
            return $ Just $ LabelEvent tid label
        _ -> do
            case eventToCounter eventId of
                Just (ctr, loc) -> do
                    tid <- word32be
                    -- Parser.fromEffect $ putStr $ "event = " ++ show eventId ++ " ts = " ++ show ts
                    -- Parser.fromEffect $ putStrLn $ " tid = " ++ show tid
                    -- trace ("tid = " ++ show tid ++ " ctr = " ++ show ctr ++ " loc = " ++ show loc ++ " val = " ++ show ts) (return ())
                    return $ Just $ CounterEvent tid "" ctr loc ts
                _ -> do
                    -- Parser.fromEffect $ putStrLn ""
                    let r = Map.lookup (fromIntegral eventId) kv
                    len <-
                        case r of
                            Just x ->
                                if x == -1
                                then fmap fromIntegral word16be
                                else return x
                            Nothing -> error $ "Event not in the header: " ++ show eventId
                    _ <- Parser.takeEQ len Fold.drain
                    return Nothing

{-# INLINE parseEvents #-}
parseEvents :: IntMap Int -> StreamK IO (Array Word8) -> Stream IO Event
parseEvents kv =
      Stream.catMaybes
    . Stream.catRights
    . Stream.parseMany (event kv)
    . Stream.unfoldMany Array.reader
    . StreamK.toStream

parsePrimOpLineToEvent :: String -> Event
parsePrimOpLineToEvent line = runIdentity $ do
    res <-
        Stream.fromList line
            & Stream.foldMany (Fold.takeEndBy_ (== '/') Fold.toList)
            & Stream.toList
    case res of
        ["STAT", location, tid, counter, tag, val] ->
            pure $ CounterEvent
                (read tid) tag (read counter) (read location) (read val)
        _ -> error $ "Incorrect number of chunks"

parsePrimOpEvents :: Stream IO (Array Word8) -> Stream IO Event
parsePrimOpEvents inp =
    Unicode.decodeUtf8Chunks inp
        & Stream.foldMany
              (Fold.takeEndBy_
                   (== '\n')
                   (parsePrimOpLineToEvent <$> Fold.toList))
