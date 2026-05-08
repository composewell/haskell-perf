{-# LANGUAGE BangPatterns #-}

import Control.Monad.IO.Class (MonadIO(..))
import Debug.Trace (traceEventIO)
import Foreign.C.Types
import System.Posix.Signals

foreign import ccall unsafe "unistd.h sleep"
    c_sleep :: CUInt -> IO CUInt


{-# INLINE withTracingFlow #-}
withTracingFlow :: MonadIO m => String -> m a -> m a
withTracingFlow tag action = do
    liftIO $ traceEventIO ("START:" ++ tag)
    !res <- action
    liftIO $ traceEventIO ("END:" ++ tag)
    pure res

sleepBlock :: IO ()
sleepBlock = do
    -- So that signals do not interrupt the sleep
    blockSignals fullSignalSet
    _ <- c_sleep 10
    return ()

{-
sleepWithEvents :: IO ()
sleepWithEvents = do
    traceEventIO "before sleep"
    sleepBlock
    traceEventIO "after sleep"
-}

emptyBlock :: IO ()
emptyBlock = return ()

main :: IO ()
main = do
    withTracingFlow "EMPTY" emptyBlock
    withTracingFlow "SLEEP" sleepBlock
    return ()
