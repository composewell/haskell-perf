{-# LANGUAGE CPP #-}
module Aggregator
    ( translateThreadEvents
    , collectThreadCounter
    )
where

import Data.Int (Int64)
import Data.Set (Set)
import Data.Word (Word32)
import EventParser (Event (..), Counter(..), Location(..))
import Streamly.Internal.Data.Fold (Fold(..), Step(..))
import Streamly.Internal.Data.Tuple.Strict (Tuple'(..))

import qualified Data.List as List
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set

-------------------------------------------------------------------------------
-- Event processing
-------------------------------------------------------------------------------

-- XXX It would be more intuitive for scans if we use "Partial s b" instead of
-- using extract. We can avoid having to save the result in state many a times.

-- All counter events are recorded against a unique key (tid, window, counter).
--
-- A user defined window is prefixed with the thread-id of the thread which
-- started it. Therefore, if the window code is entered by multiple threads,
-- each thread is assigned a different window name.
--
-- Each window accounts all threads active at the time when the window is
-- active. Therefore, any thread start/stop events are broadcast to all the
-- currently active windows.
--
-- When a window starts, many threads may already be existing in the system.
-- After the window is entered, events of all threads are broadcast to the
-- window. If a thread was already active when the window was entered, then we
-- may get a suspend event without first getting a resume event. Similar,
-- problem occurs at the end of the window.
--
-- XXX Currently, we ignore these are warnings. To get an accurate accounting
-- we can add APIs to GHC so that we stop all threads when the window starts or
-- exits.

-- TODO: When different threads enter the same window we can combine the data
-- from all the threads. i.e. when the window exits, we combine the results
-- into a common pool, we can use (tid 0, window, counter) key for that. This
-- behavior could be controlled based on a CLI option.

newtype Window = Window (Set Word32) -- thread ids that have joined the window

{-# INLINE translateThreadEvents #-}
translateThreadEvents :: Fold IO Event [Event]
translateThreadEvents = Fold step initial extract extract

    where

    initial = pure $ Partial $ Tuple' Map.empty []

    {-
    -- OneShot logs instead of pre/post brackets
    threadEvent2 mp tid ctr1 v1 ctr2 v2 =
        pure $ Partial $ Tuple' mp
            ((if v1 /= 0
            then [((tid, "default", ctr1), (OneShot, (fromIntegral v1)))]
            else []) ++
            (if v2 /= 0
            then [((tid, "default", ctr2), (OneShot, (fromIntegral v2)))]
            else []))
     -}

    bcastToWindows mp tid value ctr loc =
        -- XXX Can we use alterF to avoid two operations on the map
        case Map.lookup ctr mp of
            Just wmap ->
                let xs = event "default" : fmap event (Map.keys wmap)
                    wmap1 = fmap updateWindow wmap
                    -- XXX We can use an IORef to avoid too much churn
                    mp1 = Map.insert ctr wmap1 mp
                in (mp1, xs)
            Nothing -> (mp, [event "default"])

        where

        event tag = CounterEvent tid tag ctr loc value

        updateWindow (Window tidSet) = Window (Set.insert tid tidSet)

    threadEventBcast mp tid value ctr loc = do
        let (mp1, xs) = bcastToWindows mp tid value ctr loc
         in pure $ Partial $ Tuple' mp1 xs

    -- A foreign window cannot have any other events inside it therefore we do
    -- not need to remember it in the state.
    --
    -- Broadcast the event to all windows belonging to the counter including
    -- the "default" window so that it is counted as part of the thread timing.
    windowStartForeign mp tid tag value ctr = do
        let (mp1, xs) = bcastToWindows mp tid value ctr Resume
            e = CounterEvent tid tag ctr Resume value
         in pure $ Partial $ Tuple' mp1 (e : xs)

    windowEndForeign mp tid tag value ctr = do
        let (mp1, xs) = bcastToWindows mp tid value ctr Suspend
            e = CounterEvent tid tag ctr Exit value
         in pure $ Partial $ Tuple' mp1 (e : xs)

    windowStart mp tid tag value ctr = do
        -- trace ("Window start: " ++ tag ++ " ctr " ++ show ctr ++ " tid "
        --      ++ show tid ++ " value " ++ show value) (return ())
        let mp1 = Map.alter alter ctr mp
        pure $ Partial $ Tuple' mp1 [f tag]

        where

        window = Window Set.empty

        alter Nothing = Just $ Map.singleton tag window
        alter (Just wmap) =
            case Map.lookup tag wmap of
                Just _ ->
                    error $ "Duplicate window add: " ++ "window = " ++ tag
                            ++ " tid = " ++ show tid ++ " ctr = " ++ show ctr
                Nothing -> Just $ Map.insert tag window wmap

        f x = CounterEvent tid x ctr Resume value

    windowEnd mp tid tag value ctr = do
        case Map.lookup ctr mp of
            Nothing -> err1
            Just wmap ->
                case Map.lookup tag wmap of
                    Nothing -> err2
                    Just (Window tidSet) -> do
                        let wmap1 = Map.delete tag wmap
                            mp1 = Map.insert ctr wmap1 mp
                            -- Purge all the threads part of the window,
                            -- excluding the current thread.
                            xs = fmap (purge tag) (Set.toList (Set.delete tid tidSet))
                            xs1 = evict tag tid : xs
                        pure $ Partial $ Tuple' mp1 xs1

        where

        err1 = error $ "Counter has no active windows: ctr = "
                        ++ show ctr ++ " window = " ++ tag
                        ++ " tid = " ++ show tid
        err2 = error $ "Window not found in the windows for the counter:"
                        ++ show ctr ++ " window = " ++ tag
                        ++ " tid = " ++ show tid

        evict x t = CounterEvent t x ctr Exit value
        purge x t = CounterEvent t x ctr Purge value

    -- Broadcast "default" window events to all windows in the thread
    step (Tuple' mp _) (CounterEvent tid "" counter Resume value) =
        threadEventBcast mp tid value counter Resume
    step (Tuple' mp _) (CounterEvent tid "" counter Suspend value) =
        threadEventBcast mp tid value counter Suspend
    step _ (CounterEvent _ "" _ Exit _) = error "Unexpected Exit event"

    -- Broadcast "foreign"" window events to all windows in the thread
    -- including the "default" window to include them in the entire thread
    -- timings. Only thread level counters (ThreadCPUTime) are to be broadcast
    -- the rest fall through to the regular window handling below.
    step (Tuple' mp _) (CounterEvent tid tag ctr Resume value)
        | ":foreign" `List.isSuffixOf` tag && ctr < ThreadAllocated = do
        windowStartForeign mp tid tag value ctr
    step (Tuple' mp _) (CounterEvent tid tag ctr Suspend value)
        | ":foreign" `List.isSuffixOf` tag && ctr < ThreadAllocated = do
        windowEndForeign mp tid tag value ctr
    step _ (CounterEvent tid tag ctr ev _)
        | ":foreign" `List.isSuffixOf` tag =
        error $ "Unexpected event: tid = " ++ show tid ++ " ctr = "
            ++ show ctr ++ " loc = " ++ show ev

    -- User defined window events
    step (Tuple' mp _) (CounterEvent tid tag counter Resume value) =
        windowStart mp tid tag value counter
    step (Tuple' mp _) (CounterEvent tid tag counter Suspend value) =
        windowEnd mp tid tag value counter
    step _ (CounterEvent _ _ _ Exit _) = error "Unexpected Exit event"

    -- Pass non-counter events as it is
    step (Tuple' mp _) ev = pure $ Partial $ Tuple' mp [ev]

    extract (Tuple' _ xs) = pure xs

-- Note Exit event occurs exclusively for the thread that started a Window,
-- other threads see the Purge event, Purge cannot occur for window starter
-- thread. Exit and Purge are used to indicate end of window which results in a
-- Left value to be emitted.
data CollectState =
      CollectInit -- before the window is started, or after a Purge
    | CollectInit1 -- if we get a suspend first instead of Resume
    | CollectPartial Int64 -- after a Resume
    | CollectDone Int64 -- after a Suspend, continuing window return (Right v)
    | CollectExit Int64 -- After an Exit, end of window return (Left v)
    | CollectPurge -- After a Purge, end of window return (Left Nothing)

-- XXX Pass the window, counter as well for information in errors
{-# INLINE collectThreadCounter #-}
collectThreadCounter ::
    Fold IO ((Word32, String, Counter), (Location, Int64))
    (Maybe (Either (Maybe Int64) Int64))
collectThreadCounter = Fold step initial extract extract

    where

    initial = pure $ Partial CollectInit

    -- For non-default windows ignore events until a resume arrives.
    step CollectInit (_, (Resume, v)) = pure $ Partial $ CollectPartial v
    step CollectInit stat@((_,tag,_), (Suspend, _)) =
        if tag == ""
        then error $ "CollectInit: first event must be Resume, got Suspend."
                    ++ show stat
        else do
            putStrLn $ "Warning! ignoring first Suspend at the start of "
                ++ "window: " ++ show stat
            pure $ Partial CollectInit1
    step CollectInit stat =
        error $ "CollectInit: expecting Resume or Suspend " ++ show stat

    -- If we are in this state it means it is not the thread that started the
    -- window. That thread would always have a Resume event as first event.
    step CollectInit1 (_, (Resume, v)) = pure $ Partial $ CollectPartial v
    step CollectInit1 (_, (Purge, _)) = pure $ Partial CollectPurge
    step CollectInit1 stat =
        error $ "CollectInit1: expecting Resume or Purge" ++ show stat

    step (CollectDone _) (_, (Resume, v)) = pure $ Partial $ CollectPartial v
    step (CollectDone _) (_, (Purge, _)) = pure $ Partial CollectPurge
    step (CollectDone _) stat =
        error $ "CollectDone: expecting Resume or Purge" ++ show stat

    step (CollectExit _) (_, (Resume, v)) = pure $ Partial $ CollectPartial v
    step (CollectExit _) stat =
        error $ "CollectExit: expecting Resume." ++ show stat

    step CollectPurge (_, (Resume, v)) = pure $ Partial $ CollectPartial v
    step CollectPurge (_, (Suspend, _)) = pure $ Partial $ CollectInit1
    step CollectPurge stat =
        error $ "CollectPurge: expecting Resume or Suspend." ++ show stat

    step (CollectPartial old) stat@(_, (Suspend, new)) = do
        let delta = new - old
        if delta < 0
            then error $ "CollectPartial: Suspend: counter delta is negative:"
                    ++  "input = " ++ show stat ++ " old = " ++ show old
            else pure ()
        pure $ Partial $ CollectDone delta
    step (CollectPartial old) stat@(_, (Exit, new)) = do
        let delta = new - old
        if delta < 0
            then error $ "CollectPartial: Exit: counter delta is negative:"
                    ++  "input = " ++ show stat ++ " old = " ++ show old
            else pure ()
        pure $ Partial $ CollectExit delta
    -- Note: It is important to send a Purge immediately to all the active
    -- window counters. Otherwise, if the window is entered again without
    -- exiting the counter may record an incorrect and large value, if we got a
    -- susepnd event first. If we get a resume event first then we can discard
    -- the old state.
    --
    -- Also Purge causes the window-end event to be emitted, which is important
    -- for the count of window collections.
    step (CollectPartial _) _stat@(_, (Purge, _)) = do
        -- XXX we can just report once
        -- putStrLn $ "Warning! Purging uncollected window counter " ++ show stat
        pure $ Partial CollectPurge
    step (CollectPartial _) stat@(_, (Resume, v)) = do
        -- XXX Missing event after CTRL-C causes this, need to fix that.
        putStrLn $ "Warning! Lost uncollected counter " ++ show stat
        pure $ Partial $ CollectPartial v

    extract CollectInit = pure Nothing
    extract CollectInit1 = pure Nothing
    extract (CollectDone v) = pure (Just (Right v))
    extract (CollectExit v) = pure (Just (Left (Just v)))
    extract (CollectPurge) = pure (Just (Left Nothing))
    extract (CollectPartial _) = do
        -- putStrLn $ "End of log: ignoring pending event"
        pure Nothing
