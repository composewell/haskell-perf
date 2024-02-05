{-# LANGUAGE CPP #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE UnboxedTuples #-}
{-# LANGUAGE UnliftedFFITypes #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE QuasiQuotes #-}
{-# OPTIONS_GHC -Wno-missing-export-lists #-}
{-# OPTIONS_GHC -Wno-implicit-lift #-}

module Stat where

--------------------------------------------------------------------------------
-- Imports
--------------------------------------------------------------------------------

import Control.Monad (forM_)
import Data.Bits (shiftR, (.&.), setBit, testBit, complement)
-- import Data.IORef (newIORef, IORef, atomicModifyIORef)
import Data.Word (Word8)
import Foreign.C.Types (CInt(..))
import GHC.Conc.Sync (myThreadId, ThreadId(..))
import GHC.Exts (ThreadId#)
import GHC.IO(IO(..))
import GHC.IO.Unsafe (unsafePerformIO)
import GHC.Int(Int64(..), Int32(..))
import System.Environment (lookupEnv)
import System.IO
    (openFile, IOMode(..), Handle, BufferMode(..), hSetBuffering, hClose)
-- import Streamly.Unicode.String (str)
import System.CPUTime (getCPUTime)
import Data.Time.Clock (getCurrentTime, UTCTime(..), diffUTCTime)
import Streamly.Internal.Data.Array (Array(..))
import Streamly.FileSystem.Handle as Handle

import qualified Streamly.Data.StreamK as StreamK
import qualified Streamly.Data.Stream as Stream
import qualified Streamly.Internal.Data.Array as Array
import qualified Streamly.Internal.Data.MutByteArray as MBA

import Prelude
import Language.Haskell.TH

#define PRIM_OP_AVAILABLE
-- #undef PRIM_OP_AVAILABLE

#ifdef PRIM_OP_AVAILABLE
import GHC.Exts (threadCPUTime#)
#else
import GHC.Exts (RealWorld, Int32#, Int64#, State#)

threadCPUTime# ::
    State# RealWorld -> (# State# RealWorld, Int64#, Int64#, Int32#, Int32# #)
threadCPUTime# = undefined
#endif

--------------------------------------------------------------------------------
-- Thread stat
--------------------------------------------------------------------------------

type SrcLoc = Loc

data EvLoc
    = Start
    | Record
    | Restart
    | End
    deriving (Read, Show, Ord, Eq)
$(MBA.deriveSerialize [d|instance MBA.Serialize EvLoc|])

data Counter
    = ThreadCpuTime
    | ProcessCpuTime
    | WallClockTime
    | Allocated
    | SchedOut
    deriving (Read, Show, Ord, Eq)
$(MBA.deriveSerialize [d|instance MBA.Serialize Counter|])

data Metric =
    Metric
        { m_tid :: Int32
        , m_namespace :: String
        , m_modName :: String
        , m_lineNum :: Int32
        , m_counter :: Counter
        , m_location :: EvLoc
        , m_value :: Int64
        }
    deriving (Show)
$(MBA.deriveSerialize [d|instance MBA.Serialize Metric|])

{-# INLINE tenPow9 #-}
tenPow9 :: Int64
tenPow9 = 1000000000

--------------------------------------------------------------------------------
-- Perf handle
--------------------------------------------------------------------------------

-- Perf handle
-- Can we rely on the RTS to close the handle?
perfHandle :: Handle
perfHandle =
    unsafePerformIO $ do
        h <- openFile "perf.bin" WriteMode
        hSetBuffering h NoBuffering
        pure h

closePerfHandle :: IO ()
closePerfHandle = hClose perfHandle


--------------------------------------------------------------------------------
-- Window Type
--------------------------------------------------------------------------------

createBitMap :: Maybe [Int] -> IO MBA.MutByteArray
createBitMap mLineList = do
    let maxLines = 8000
    arr <- MBA.new maxLines
    forM_ [0..(maxLines - 1)]
        $ \i -> MBA.pokeAt i arr (0 :: Word8)
    case mLineList of
        Nothing -> pure arr
        Just [] -> do
            forM_ [0..(maxLines - 1)]
                $ \i -> MBA.pokeAt i arr (complement (0 :: Word8))
            pure arr
        Just lineList -> do
            forM_ lineList $ \ln -> pokeBitOn ln arr
            pure arr

--------------------------------------------------------------------------------
-- Helpers
--------------------------------------------------------------------------------

pokeBitOn :: Int -> MBA.MutByteArray -> IO ()
pokeBitOn i arr = do
    val <- MBA.peekAt (shiftR i 3) arr
    MBA.pokeAt (shiftR i 3) arr (setBit (val :: Word8) (i .&. 7))

testBitOn :: Int -> MBA.MutByteArray -> IO Bool
testBitOn i arr = do
    val <- MBA.peekAt (shiftR i 3) arr
    pure $ testBit (val :: Word8) (i .&. 7)

sizedSerialize :: MBA.Serialize a => a -> IO (Array Word8)
sizedSerialize a = do
    let len = MBA.addSizeTo 0 a + 8
    mbarr <- MBA.new len
    off0 <- MBA.serializeAt 0 mbarr (fromIntegral len :: Int64)
    off1 <- MBA.serializeAt off0 mbarr a
    pure $ Array mbarr 0 off1

--------------------------------------------------------------------------------
-- Setup Env
--------------------------------------------------------------------------------

envMeasurementWindows :: Maybe String
envMeasurementWindows = unsafePerformIO $ lookupEnv "MEASUREMENT_WINDOWS"

--------------------------------------------------------------------------------
-- Measurement
--------------------------------------------------------------------------------

foreign import ccall unsafe "rts_getThreadId" getThreadId :: ThreadId# -> CInt

{-# INLINE getThreadStatLowLevel #-}
getThreadStatLowLevel :: IO (Int64, Int64, Int32, Int32)
getThreadStatLowLevel = IO $ \s ->
   case threadCPUTime# s of
    (# s', sec, nsec, alloc, count_sched #) ->
        (# s', (I64# sec, I64# nsec, I32# alloc, I32# count_sched) #)

picoToNanoSeconds :: Integer -> Int64
picoToNanoSeconds x = fromIntegral (x `div` 1000)

epochTime :: UTCTime
epochTime = UTCTime (toEnum 0) 0

getThreadStat :: IO (Int32, Int64, Int64, Int64)
getThreadStat = do
    ThreadId tid <- myThreadId
    (sec, nsec, alloc, switches) <- getThreadStatLowLevel
    pure
        ( fromIntegral (getThreadId tid)
        , sec * tenPow9 + nsec
        , fromIntegral (alloc * 8)
        , fromIntegral switches
        )

printMetricList :: Handle -> [Metric] -> IO ()
printMetricList handle mList = do
    arr <-
        Array.fromChunksK
            $ StreamK.mapM sizedSerialize
            $ StreamK.fromStream
            $ Stream.fromList mList
    putChunk handle arr


eventGeneric ::
    (forall b. IO b -> m b) -> String -> EvLoc -> SrcLoc -> Handle -> m ()
eventGeneric liftio namespace evLoc srcLoc handle = liftio $ do
    (a, b, c, d) <- getThreadStat
    let modName = loc_module srcLoc
    let lnNum = (fromIntegral :: Int -> Int32) $ fst (loc_start srcLoc)
    pCpuTime <- picoToNanoSeconds <$> getCPUTime
    wTimeU <- getCurrentTime
    let wTime = round $ diffUTCTime wTimeU epochTime * 1e9
    let mList =
            [ Metric a namespace modName lnNum ThreadCpuTime evLoc b
            , Metric a namespace modName lnNum Allocated evLoc c
            , Metric a namespace modName lnNum SchedOut evLoc d
            , Metric a namespace modName lnNum ProcessCpuTime evLoc pCpuTime
            , Metric a namespace modName lnNum WallClockTime evLoc wTime
            ]
{-
    shouldStat <- testBitOn (fromEnum win) statEnv
    if True
    then printMetricList mList
    else pure ()
-}
    printMetricList handle mList

withEvLoc :: Q Exp -> Q Exp
withEvLoc f = do
    Loc a b c d e <- location
    appE f [| Loc a b c d e |]

start :: Q Exp
start = do
    Loc a b c d e <- location
    [|eventGeneric id "g" Start (Loc a b c d e) perfHandle|]

end :: Q Exp
end = do
    Loc a b c d e <- location
    [|eventGeneric id "g" End (Loc a b c d e) perfHandle|]

record :: Q Exp
record = do
    Loc a b c d e <- location
    [|eventGeneric id "g" Record (Loc a b c d e) perfHandle|]

restart :: Q Exp
restart = do
    Loc a b c d e <- location
    [|eventGeneric id "g" Restart (Loc a b c d e) perfHandle|]
