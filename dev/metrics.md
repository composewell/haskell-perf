# Metrics

This library is about representing metrics in general, combining and processing
them. For measurement and collection, see streamly-measure.

## Defining a Metric

A metric represents a value of any type that is a Monoid. This enables
the metric to have a default starting value, and multiple instances of a
metric can be combined together.

A metric is a key and a value. The key is required to identify different
metrics collected by a system.

A metric is a timeseries.  Multiple instances of a metric are generated
by the program over a period of time. Therefore, a metric has a sequence
(or timestamp) associated with it so the collected metrics can be
reordered or presented as a timeseries.

## Representation

The timeseries can be represented by an array. And then a stream of arrays.

If it is a simple ordered sequence we can just have:

(key, Array val)

The index of the array represents the sequence.

Or an "Ord" tagged value (timestamp just has to be an instance of Ord):

(key, Array (timestamp, val))

Or to compress the timestamps:

(key, basetime, Array (reltime, val))

The array may be sorted or unsorted.

## Collection

The collector generates a stream of metrics. The raw collector would generate
events that can be sent as a stream to a collecting channel. The channel could
be a part of the reader monad.

Each collector would specify a key for the metric being collected, we have
omitted that for simplicity.

(bench f) would collect performance stats and generate an event when f
finishes. (every n (bench f)) would aggregate and send every n results.

(collect n (bench f)) would invoke (bench f) n times and aggregate
all the events and send the result as a single event. Or it could be (run n
(every n (bench f))) or (bench (run n f)).

(log message) would generate a log event.

We could use different event processors e.g.:

(send (bench f)) would send the event to some collector channel.
(report (bench f)) would print the event on console.

## Aggregation

For efficiency of collection and transmission, metrics can be aggregated at the
source itself using time windows and then batches may be transmitted.

Aggregations can be done by Monoid instances.

## Composite Metrics

To represent multiple metrics we can have:

* A Map of metrics
* An array of metrics
* A record of metrics

An Array or record or map of metrics is also a Metric and they can be combined
together like a Monoid.

We can have combinators to combine a stream of metrics into arrays and then
stream of arrays.

## Serialization

Metrics can be serialized so that they can be persisted or sent over network.

## Examples of Metrics

A log is also a metric whose combining method is a concatenation with newlines.
The key could map to the log file.
