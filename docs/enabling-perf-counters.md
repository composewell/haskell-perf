## Enable Linux perf counters

Enable unrestricted use of perf counters:

```
# echo -1 > /proc/sys/kernel/perf_event_paranoid
```

## Disable CPU scaling

Set the scaling governer of all your cpus to `performance`:

```
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
echo performance > /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
...
...
echo performance > /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor
```
