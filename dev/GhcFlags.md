# GHC Flags

There are 2 types of GHC flags.
1. Compiler flags
2. RTS flags

The RTS flags are in `ghc/rts/RtsFlags.c`.
To add an RTS flag one needs to edit this file and make some follow-up changes.

The Compiler flags are in `ghc/compiler/main/DynFlags.hs`.
One needs to edit this file to add a dynamic flag.

## FAQ:

**Q:** What is the difference between an RTS flag and a Compiler flag?

A Compiler flag is a flag that the compiler takes. ie. `ghc`. Whereas the RTS
flag is the flag that the executable generated by the compiler takes.


**Q:** The Compiler flags seem to set some C Flags for a few options (for
example "-DTRACING" when "-eventlog" is enabled, etc.). This is
meta-programming. How does this work? Does a mini compilation happen
everytime we invoke GHC?

We can describe what happens with eventlog which might answer the above
question. WRT eventlog, the compiler has 2 versions of the RTS. One with
eventlog enabled and one with eventlog disabled. The executable is linked with
the RTS with the eventlog enabled when the `-eventlog` flag is enabled.

## TODO:

We need to add 2 types of flags:
1. A Compiler flag to enable perf counter structures.
2. An RTS flag to enable the update of the counters.

This is similar to the eventlog approach.
- `-eventlog` (Compiler flag) to enable the eventlog.
- `-l` (RTS flag) to actually do the accounting.
