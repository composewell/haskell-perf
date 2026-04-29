{-# LANGUAGE DerivingVia #-}
-- |
-- Module      : Streamly.Metrics.Type
-- Copyright   : (c) 2021 Composewell Technologies
-- License     : Apache-2.0
-- Maintainer  : streamly@composewell.com
-- Stability   : experimental
-- Portability : GHC
--
-- A 'Metric' represents a sample of some value labeled with a unique
-- identifier.  Metrics may be ordered with respect to each other or with
-- respect to time e.g. 'Sequence' or 'Time'.  Metrics may have different
-- semantics e.g. 'Counter' or 'Gauge'.
--
module Streamly.Metrics.Type
    (
    -- * Semantics
    -- | A counter counts how many events of a type have occurred whereas a
    -- gauge measures what is the value of a certain property at a given point
    -- of time. A counter is always @monotonic@ whereas a gauge is usually
    -- @volatile@.
      Log (..)
    , Counter (..)
    , GaugeMax (..)

    -- * Ordering
    -- | Metrics may be ordered with respect to each other or with respect to
    -- time.
    , Timestamped (..)
    , Sequenced (..)

    -- * Units
    , Seconds (..)
    , Bytes (..)

    -- * Utilities
    , Indexable(..)
    , showList
    )
where

import Data.List (sortBy)
import Streamly.Internal.Data.Time.Units (AbsTime)
import Text.Printf (printf, PrintfArg)
import Prelude hiding (showList)

-------------------------------------------------------------------------------
-- Metric semantics
-------------------------------------------------------------------------------

-- Semantics define what operations make sense on a metric or how you can
-- combine, aggregate or collapse multiple samples of a metric.
--
-- Metrics can be aggregated in different ways:
--
-- * a -> a -> a (catenate, sum)
-- * a -> Array a -> Array a (cons)
--
-- If the type is a Monoid we can represent all interesting operations on the
-- metrici in terms of Monoid.  If "a" represents a metric then "Array a" also
-- represents a metric, if "a" is a Monoid "Array a" is also a Monoid. XXX do
-- we make the monoid instance of "Array a" append the type "a" rather than the
-- array itself like the Tee type or do we need a newtype wrapper for that?

-- | A log is the simplest metric. It has no semantics except catenation of
-- multiple logs.
--
newtype Log a = Log a

-- | Represents a monotonically increasing value like time or number of events.
--
-- Counters should support addition and subtraction operations. We can diff the
-- values of a counter at two points to figure out the change that occurred in
-- that duration.  On the other hand, maximum, minimum, range or average values
-- usually do not make sense for a counter.
--
-- It does not make sense to add different snapshots of the same counter,
-- however it makes sense to aggregate counters from multiple independent
-- sources e.g. to aggregate total number of events, or to add durations.
-- Counters could be continuous of discrete. Time is an example of a continuous
-- counter whereas instruction count is an example of a discrete counter.
--
-- Time is a fundamental counter, in fact it is a reference counter for
-- all changes, changes are measured with respect to time.

newtype Counter a =
    Counter a
        deriving (Num, Fractional)

instance Show a => Show (Counter a) where
    show (Counter a) = show a

-- | Represents the current utilization level of some resource e.g. memory
-- in use. A gauge can increase or decrease from a previous value.
--
-- It usually does not make sense to add gauge metrics collected from multiple
-- independent sources. However, maximum, minimum, range or average values make
-- sense for multiple snapshots of a gauge or the value of a gauge from
-- multiple sources.  Thus, a gauge type can be further classified into
-- maximum (e.g. peak memory) or minimum (e.g. available memory), average (e.g.
-- load average) or range.
--
newtype GaugeMax a = GaugeMax a

instance Show a => Show (GaugeMax a) where
    show (GaugeMax a) = show a

instance (Num a, Ord a) => Num (GaugeMax a) where
    fromInteger a = GaugeMax (fromInteger a)
    abs (GaugeMax a) = GaugeMax (abs a)
    signum (GaugeMax a) = GaugeMax (signum a)
    (GaugeMax a) * (GaugeMax b) = GaugeMax (a * b)

    -- XXX Abusing Num instance, we can use a separate type class
    -- For a GaugeMax these are defined as maximum of the two values
    (GaugeMax a) - (GaugeMax b) = GaugeMax (max a b)
    (GaugeMax a) + (GaugeMax b) = GaugeMax (max a b)

-------------------------------------------------------------------------------
-- Metrics ordering
-------------------------------------------------------------------------------

-- How do we distinguish (if at all) different samples of the same metric.

-- | A value tagged with a sequence number. Useful when we want to order the
-- samples of a value with respect to each other but do not care about the
-- ordering with respect to time.
--
data Sequenced a = Sequenced Int a

-- | A value tagged with a timestamp. Useful when we want to measure samples
-- with respect to time.
--
data Timestamped a = Timestamped AbsTime a    -- XXX use a tuple and derive Ord?

-- XXX When zipping trees of metrics we need to traverse the keys like file
-- paths or a file system tree. That means a metric should probably support a
-- lookup operation like a Map.
--
-- The simplest would be a one level Map as a metric or a metric set. Should we
-- separate these to concepts of a metric and a metric set. A metric would be
-- like a file whereas a metric set would be like a dir tree (MetricTree).
--
-- ZipList and a single level Map are a one level metric tree.

-------------------------------------------------------------------------------
-- Units
-------------------------------------------------------------------------------

-- XXX Use the common time units code from streamly instead.
--
-- | Describe a relative unit i.e. a unit in terms of another unit. A relative
-- unit has a label and a ratio which when multiplied with the unit gives us
-- the other unit. For example, if the known time unit is seconds, we can
-- describe a millisecond as @Unit "ms" (1/1000)@.
data RelativeUnit a = RelativeUnit String a deriving Show

-- Given seconds, choose appropriate unit based on the size
secondsConverter :: (Ord a, Fractional a) => a -> RelativeUnit a
secondsConverter k
    | k < 0      = secondsConverter (-k)
    | k >= 1     = RelativeUnit "s" 1
    | k >= 1e-3  = RelativeUnit "ms" 1e3
    | k >= 1e-6  = RelativeUnit "μs" 1e6
    | otherwise  = RelativeUnit "ns" 1e9

newtype Seconds a = Seconds a deriving (Num, Fractional, Eq, Ord)

instance (Show a, Ord a, Fractional a, PrintfArg a) => Show (Seconds a) where
    show (Seconds t) =
        let (RelativeUnit label multiplier) = secondsConverter t
         in printf "%.2f" (t * multiplier) ++ " " ++ label

-- Given bytes, choose appropriate unit based on the size
bytesConverter :: (Num a, Ord a) => a -> RelativeUnit a
bytesConverter k
    | k < 0              = bytesConverter (-k)
    | k >= 2^(30 :: Int) = RelativeUnit "GiB" (2^(30 :: Int))
    | k >= 2^(20 :: Int) = RelativeUnit "MiB" (2^(20 :: Int))
    | k >= 2^(10 :: Int) = RelativeUnit "KiB" (2^(10 :: Int))
    | otherwise          = RelativeUnit "Bytes" 1

newtype Bytes a = Bytes a deriving (Num, Eq, Ord, Fractional)

instance (Show a, Num a, Ord a, PrintfArg a, Integral a) => Show (Bytes a)

    where

    show (Bytes b) =
        let (RelativeUnit label multiplier) = bytesConverter b
            n = fromIntegral b / fromIntegral multiplier :: Double
         in printf "%.2f" n ++ " " ++ label

-------------------------------------------------------------------------------
-- Utilities
-------------------------------------------------------------------------------

-- Give indices to different values of the type 'a'
class Indexable a where
    getIndex :: a -> Int

-- Show a list of values of an Indexable type sorted in ascending index order
showList :: (Show a, Indexable a) => [a] -> String
showList xs = unlines $ show <$> sortBy f xs

    where

    f p1 p2 = compare (getIndex p1) (getIndex p2)
