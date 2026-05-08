import Control.Concurrent
import Control.Monad
import System.Posix.Process (getProcessID)

consoleLoop :: IO ()
consoleLoop = do
    line <- getLine
    putStrLn line
    consoleLoop

threadLoop :: IO ()
threadLoop = do
    tid <- myThreadId
    putStrLn $ "running thread: " ++ show tid
    -- tight loop  with occasional delay
    forever $ go (0 :: Integer)

    where

    go n =
        if n `mod` 1000000 == 0
        then threadDelay 1
        else go (n+1)

main :: IO ()
main = do
    pid <- getProcessID
    putStrLn $ "pid: " ++ show pid
    tid <- myThreadId
    putStrLn $ "main thread: " ++ show tid
    tid1 <- forkIO threadLoop
    putStrLn $ "forked thread: " ++ show tid1
    tid2 <- forkIO threadLoop
    putStrLn $ "forked thread: " ++ show tid2
    consoleLoop -- `catch` (\(e :: IOException) -> return ())
