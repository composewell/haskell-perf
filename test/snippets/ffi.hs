import Control.Concurrent
import Control.Monad
import Foreign.C.Types
import System.Posix.Signals
import Debug.Trace

foreign import ccall unsafe "unistd.h sleep"
    c_sleep :: CUInt -> IO CUInt

main :: IO ()
main = do
    tid <- myThreadId
    print tid
    forkIO (forever (putStrLn "hello" >> threadDelay 1000000))
    blockSignals fullSignalSet
    threadDelay 10000000
    putStrLn "before sleep"
    c_sleep 10
    putStrLn "after sleep"
    return ()
