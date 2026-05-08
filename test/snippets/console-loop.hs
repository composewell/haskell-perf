loop :: IO ()
loop = do
    line <- getLine
    -- let n = read line :: Int
    -- putStrLn $ "You entered: " ++ show n
    putStrLn line
    loop

main :: IO ()
main = loop -- `catch` (\(e :: IOException) -> return ())
