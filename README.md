# haskell-perf

Enable unrestricted use of perf counters:

```
# echo -1 > /proc/sys/kernel/perf_event_paranoid
```

GHC Patch: https://github.com/composewell/ghc/tree/ghc-8.8.4-eventlog-enhancements
