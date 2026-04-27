import Foreign.C.Types
import System.Posix.Signals
import Debug.Trace

foreign import ccall unsafe "unistd.h sleep"
    c_sleep :: CUInt -> IO CUInt

main :: IO ()
main = do
    blockSignals fullSignalSet
    traceEventIO "before sleep"
    c_sleep 10
    traceEventIO "after sleep"
    return ()
