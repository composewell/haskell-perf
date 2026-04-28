module Streamly.Metrics.Perf.Type
    (
      PerfMetrics (..)
    , checkMonotony
    )
where

import Data.Word (Word64)
import Streamly.Metrics.Type
    (GaugeMax(..), Seconds(..), Bytes(..), Indexable(..))

-- Use Counter/Gauge as the outer constructor and Bytes/Seconds as the inner
-- constuctor.
--
-- The order is important, related stats are grouped/sorted in that order for
-- presentation purposes.
data PerfMetrics =
  -- | MonotonicTime and GcElapsedTime both should provide the same figures.
    MonotonicTime !(Seconds Double)
    -- XXX Make this hierarchical (include rusage data)
    -- data ProcessCPUTime = total user system
  -- | In a single threaded system with unbound threads 'ProcessCPUTime',
  -- 'ThreadCPUTime', ('RuUtime' + 'RuStime) and 'GcCpuTime' would be the same.
  -- Note that a binary built with `-threaded` and using `-N1` RTS options is
  -- not the same as single threaded because the GC thread may still run in a
  -- separate OS thread.
  --
  -- Note that if the perf stats measurement pre and post calls straddle over a
  -- blocking Haskell thread then the stats may include cpuTime for all other
  -- threads that might have run until our post call stats collection occurs.
  -- So it may not reflect the accurate measurement of perf stats of the call
  -- in question.
  --
  | ProcessCPUTime !(Seconds Double)
  | ThreadCPUTime !(Seconds Double)

    -- XXX Make this hierarchical
    -- data ElapsedTime = ElapsedTime total mutator gc
    -- GC Memory Stats
  | GcElapsedTime !(Seconds Double)
      -- | 'GcMutatorElapsedTime' and 'GcGcElapsedTime' should add up to
      -- 'GcElapsedTime'.
      | GcMutatorElapsedTime !(Seconds Double)
      | GcGcElapsedTime !(Seconds Double)

    -- XXX Make this hierarchical
    -- data CPUTime = total mutator gc
  | GcCpuTime !(Seconds Double)
      -- | 'GcMutatorCpuTime' and 'GcGcCpuTime' should add up to 'GcCpuTime'.
      -- XXX Note that GHC measures GcCputTime to be just the utime using
      -- rusage, it should be utime + stime to match PROCESS_CPU_TIME clock.
      | GcMutatorCpuTime !(Seconds Double)
      | GcGcCpuTime !(Seconds Double)

  | GcAllocatedBytes !(Bytes Word64)
  | GcCopiedBytes !(Bytes Word64)
  | GcMaxLiveBytes !(GaugeMax (Bytes Word64))
      | GcMaxLargeObjectBytes !(GaugeMax (Bytes Word64))
      | GcMaxCompactBytes !(GaugeMax (Bytes Word64))
  | GcMaxMemInUse !(GaugeMax (Bytes Word64))

    -- rusage Stats
    -- | 'RuUtime' and 'RuStime' should add up to 'ProcessCPUTime'.
  | RuUtime     !(Seconds Double)
  | RuStime     !(Seconds Double)
  | RuMaxrss    !(GaugeMax (Bytes Word64))
  | RuIxrss     !(GaugeMax (Bytes Word64))
  | RuIdrss     !(GaugeMax (Bytes Word64))
  | RuIsrss     !(GaugeMax (Bytes Word64))
  | RuMinflt    !Word64
  | RuMajflt    !Word64
  | RuNswap     !Word64
  | RuInblock   !Word64
  | RuOublock   !Word64
  | RuMsgsnd    !Word64
  | RuMsgrcv    !Word64
  | RuNsignals  !Word64
  | RuNvcsw     !Word64
  | RuNivcsw    !Word64
  | Count       !Word64
    deriving (Show)

#define UNARY_OP_ONE(constr,op) op (constr a) = constr (op a)
#define UNARY_OP(op) \
    UNARY_OP_ONE(MonotonicTime,op); \
    UNARY_OP_ONE(ProcessCPUTime,op); \
    UNARY_OP_ONE(ThreadCPUTime,op); \
    UNARY_OP_ONE(GcAllocatedBytes,op); \
    UNARY_OP_ONE(GcCopiedBytes,op); \
    UNARY_OP_ONE(GcMaxLiveBytes,op); \
    UNARY_OP_ONE(GcMaxLargeObjectBytes,op); \
    UNARY_OP_ONE(GcMaxCompactBytes,op); \
    UNARY_OP_ONE(GcMaxMemInUse,op); \
    UNARY_OP_ONE(GcMutatorCpuTime,op); \
    UNARY_OP_ONE(GcMutatorElapsedTime,op); \
    UNARY_OP_ONE(GcGcCpuTime,op); \
    UNARY_OP_ONE(GcGcElapsedTime,op); \
    UNARY_OP_ONE(GcCpuTime,op); \
    UNARY_OP_ONE(GcElapsedTime,op); \
    UNARY_OP_ONE(RuUtime,op); \
    UNARY_OP_ONE(RuStime,op); \
    UNARY_OP_ONE(RuMaxrss,op); \
    UNARY_OP_ONE(RuIxrss,op); \
    UNARY_OP_ONE(RuIdrss,op); \
    UNARY_OP_ONE(RuIsrss,op); \
    UNARY_OP_ONE(RuMinflt,op); \
    UNARY_OP_ONE(RuMajflt,op); \
    UNARY_OP_ONE(RuNswap,op); \
    UNARY_OP_ONE(RuInblock,op); \
    UNARY_OP_ONE(RuOublock,op); \
    UNARY_OP_ONE(RuMsgsnd,op); \
    UNARY_OP_ONE(RuMsgrcv,op); \
    UNARY_OP_ONE(RuNsignals,op); \
    UNARY_OP_ONE(RuNvcsw,op); \
    UNARY_OP_ONE(RuNivcsw,op); \
    UNARY_OP_ONE(Count,op);

#define INFIX_OP_ONE(constr,op) constr a op constr b = constr (a op b)
#define FUNC_OP_ONE(constr,op) constr a `op` constr b = constr (a `op` b)

#define INFIX_OP(op) \
    INFIX_OP_ONE(MonotonicTime,op); \
    INFIX_OP_ONE(ProcessCPUTime,op); \
    INFIX_OP_ONE(ThreadCPUTime,op); \
    INFIX_OP_ONE(GcAllocatedBytes,op); \
    INFIX_OP_ONE(GcCopiedBytes,op); \
    INFIX_OP_ONE(GcMaxLiveBytes,op); \
    INFIX_OP_ONE(GcMaxLargeObjectBytes,op); \
    INFIX_OP_ONE(GcMaxCompactBytes,op); \
    INFIX_OP_ONE(GcMaxMemInUse,op); \
    INFIX_OP_ONE(GcMutatorCpuTime,op); \
    INFIX_OP_ONE(GcMutatorElapsedTime,op); \
    INFIX_OP_ONE(GcGcCpuTime,op); \
    INFIX_OP_ONE(GcGcElapsedTime,op); \
    INFIX_OP_ONE(GcCpuTime,op); \
    INFIX_OP_ONE(GcElapsedTime,op); \
    INFIX_OP_ONE(RuUtime,op); \
    INFIX_OP_ONE(RuStime,op); \
    INFIX_OP_ONE(RuMaxrss,op); \
    INFIX_OP_ONE(RuIxrss,op); \
    INFIX_OP_ONE(RuIdrss,op); \
    INFIX_OP_ONE(RuIsrss,op); \
    INFIX_OP_ONE(RuMinflt,op); \
    INFIX_OP_ONE(RuMajflt,op); \
    INFIX_OP_ONE(RuNswap,op); \
    INFIX_OP_ONE(RuInblock,op); \
    INFIX_OP_ONE(RuOublock,op); \
    INFIX_OP_ONE(RuMsgsnd,op); \
    INFIX_OP_ONE(RuMsgrcv,op); \
    INFIX_OP_ONE(RuNsignals,op); \
    INFIX_OP_ONE(RuNvcsw,op); \
    INFIX_OP_ONE(RuNivcsw,op); \
    INFIX_OP_ONE(Count,op); \
    x1 op x2 = error $ "Cannot operate on different types of metrics" \
        ++ show x1 ++ " / " ++ show x2;

checkMonotony :: PerfMetrics -> PerfMetrics -> Bool
-- XXX can we just use >= on Perfmetrics?
checkMonotony (MonotonicTime t1) (MonotonicTime t2) = t2 >= t1
checkMonotony (ProcessCPUTime t1) (ProcessCPUTime t2) = t2 >= t1
checkMonotony (ThreadCPUTime t1) (ThreadCPUTime t2) = t2 >= t1
checkMonotony (GcElapsedTime t1) (GcElapsedTime t2) = t2 >= t1
checkMonotony (GcMutatorElapsedTime t1) (GcMutatorElapsedTime t2) = t2 >= t1
checkMonotony (GcGcElapsedTime t1) (GcGcElapsedTime t2) = t2 >= t1
checkMonotony (GcCpuTime t1) (GcCpuTime t2) = t2 >= t1
checkMonotony (GcMutatorCpuTime t1) (GcMutatorCpuTime t2) = t2 >= t1
checkMonotony (GcGcCpuTime t1) (GcGcCpuTime t2) = t2 >= t1
checkMonotony _ _ = True

-- XXX Can we derive this generically?
instance Num PerfMetrics where
    fromInteger val = Count (fromInteger val)
    UNARY_OP(signum)
    UNARY_OP(abs)
    INFIX_OP(+)
    INFIX_OP(-)
    INFIX_OP(*)

#define DIV_OP_ONE(constr) constr a / Count b = constr (a / b)
#define DIV_DOUBLE(a,b) (a / fromIntegral b)
#define DIV_SECONDS(constr) constr (Seconds a) / (Count b) \
    = constr (Seconds DIV_DOUBLE(a,b))
#define DIV_ROUND(a,b) (round (fromIntegral a / fromIntegral b :: Double))
#define DIV_BYTES(constr) constr (Bytes a) / (Count b) \
    = constr (Bytes DIV_ROUND(a,b))
#define DIV_MAX_BYTES(constr) constr (GaugeMax (Bytes a)) / (Count _) = \
    constr (GaugeMax (Bytes a))
#define DIV_COUNT(constr) constr a / Count b = constr DIV_ROUND(a,b)

instance Fractional PerfMetrics where
    DIV_SECONDS(MonotonicTime)
    DIV_SECONDS(ProcessCPUTime)
    DIV_SECONDS(ThreadCPUTime)
    DIV_SECONDS(GcMutatorCpuTime)
    DIV_SECONDS(GcMutatorElapsedTime)
    DIV_SECONDS(GcGcCpuTime)
    DIV_SECONDS(GcGcElapsedTime)
    DIV_SECONDS(GcCpuTime)
    DIV_SECONDS(GcElapsedTime)
    DIV_BYTES(GcAllocatedBytes)
    DIV_BYTES(GcCopiedBytes)
    DIV_MAX_BYTES(GcMaxLiveBytes)
    DIV_MAX_BYTES(GcMaxLargeObjectBytes)
    DIV_MAX_BYTES(GcMaxCompactBytes)
    DIV_MAX_BYTES(GcMaxMemInUse)
    DIV_SECONDS(RuUtime)
    DIV_SECONDS(RuStime)
    DIV_MAX_BYTES(RuMaxrss)
    DIV_MAX_BYTES(RuIxrss)
    DIV_MAX_BYTES(RuIdrss)
    DIV_MAX_BYTES(RuIsrss)
    DIV_COUNT(RuMinflt)
    DIV_COUNT(RuMajflt)
    DIV_COUNT(RuNswap)
    DIV_COUNT(RuInblock)
    DIV_COUNT(RuOublock)
    DIV_COUNT(RuMsgsnd)
    DIV_COUNT(RuMsgrcv)
    DIV_COUNT(RuNsignals)
    DIV_COUNT(RuNvcsw)
    DIV_COUNT(RuNivcsw)
    Count a / Count _ = Count a
    x1 / x2 =
        error
            $ "Undefined fractional operation on PerfMetrics "
                ++ show x1 ++ " / " ++ show x2
    fromRational = error "fromRational: not supported for PerfMetrics"

instance Indexable PerfMetrics where
    getIndex (Count _) = 0
    getIndex (MonotonicTime _) = 1
    getIndex (GcElapsedTime _) = 2
    getIndex (GcMutatorElapsedTime _) = 3
    getIndex (GcGcElapsedTime _) = 4

    getIndex (ProcessCPUTime _) = 5
    getIndex (RuUtime _) = 6
    getIndex (RuStime _) = 7
    getIndex (ThreadCPUTime _) = 8
    getIndex (GcCpuTime _) = 9
    getIndex (GcMutatorCpuTime _) = 10
    getIndex (GcGcCpuTime _) = 11

    getIndex (GcAllocatedBytes _) = 12
    getIndex (GcCopiedBytes _) = 13
    getIndex (GcMaxLiveBytes _) = 14
    getIndex (GcMaxLargeObjectBytes _) = 14
    getIndex (GcMaxCompactBytes _) = 14
    getIndex (GcMaxMemInUse _) = 14
    getIndex (RuMaxrss    _) = 15
    getIndex (RuIxrss     _) = 16
    getIndex (RuIdrss     _) = 17
    getIndex (RuIsrss     _) = 18
    getIndex (RuMinflt    _) = 19
    getIndex (RuMajflt    _) = 20
    getIndex (RuNswap     _) = 21
    getIndex (RuInblock   _) = 22
    getIndex (RuOublock   _) = 23
    getIndex (RuMsgsnd    _) = 24
    getIndex (RuMsgrcv    _) = 25
    getIndex (RuNsignals  _) = 26
    getIndex (RuNvcsw     _) = 27
    getIndex (RuNivcsw    _) = 28
