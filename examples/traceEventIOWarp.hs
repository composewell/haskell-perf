import GHC.Conc (labelThread, myThreadId)
import Debug.Trace (traceEventIO)

withEventLog :: Application -> Application
withEventLog app request respond = do
    tid <- myThreadId
    labelThread tid "server"
    traceEventIO ("START:server")
    app request respond1

    where

    respond1 r = do
        res <- respond r
        traceEventIO ("END:server")
        return res

main :: IO ()
main =
    runSettings settings $ withEventLog $ app

