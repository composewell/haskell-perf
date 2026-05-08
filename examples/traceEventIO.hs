{-# LANGUAGE BangPatterns #-}

import Control.Monad.IO.Class (MonadIO(..))
import Debug.Trace (traceEventIO)
import Foreign.C.Types ( CUInt(..) )
import GHC.Conc (myThreadId, labelThread)
import System.Posix.Signals ( blockSignals, fullSignalSet )

foreign import ccall unsafe "unistd.h sleep"
    c_sleep :: CUInt -> IO CUInt

{-# INLINE withTracingFlow #-}
withTracingFlow :: MonadIO m => String -> m a -> m a
withTracingFlow tag action = do
    liftIO $ traceEventIO ("START:" ++ tag)
    !res <- action
    liftIO $ traceEventIO ("END:" ++ tag)
    pure res

emptyBlock :: IO ()
emptyBlock = return ()

sleepBlock :: IO ()
sleepBlock = do
    -- So that signals do not interrupt the sleep
    blockSignals fullSignalSet
    _ <- c_sleep 10
    return ()

{-# INLINE printSumLoop #-}
printSumLoop :: Int -> Int -> Int -> IO ()
printSumLoop _ _ 0 = print "All Done!"
printSumLoop chunksOf from times = do
    withTracingFlow "SUM" $ print $ sum [from..(from + chunksOf)]
    printSumLoop chunksOf (from + chunksOf) (times - 1)

main :: IO ()
main = do
    tid <- myThreadId
    labelThread tid "main-thread"

    withTracingFlow "EMPTY" emptyBlock
    withTracingFlow "SLEEP" sleepBlock
    withTracingFlow "LOOP" $ do
         printSumLoop 10000 1 100
