import Control.Concurrent (myThreadId, threadDelay, forkIO)

loop :: IO ()
loop = do
    tid <- myThreadId
    print tid
    threadDelay 10000000
    loop

main :: IO ()
main = do
    -- tid <- forkIO (return ())
    loop
