## **GHC 8.8.4**

### **Original Work**

* **Tag:** `ghc-8.8.4-eventlog-enhancements`

  * Introduced `threadCPUTime#` primop early in development.
  * CPU time was initially measured using `clock_gettime` before and after thread execution in the scheduler.
  * Later transitioned to **eventlog-based measurement**, making the primop unused in practice.

#### ↳ **Follow-up Work**

* **Tag:** `ghc-8.8.4-threadCPUTime-add-alloc-sched-count`

  * Revived `threadCPUTime#` primop hooks.
  * Added additional counters:

    * Scheduler count
    * Allocation tracking
  * Enabled **in-memory performance data collection**, allowing:

    * CPU time
    * Allocations
    * Scheduler count
  * Works **without requiring eventlogs**.

---

## **GHC 8.10.7**

### **Eventlog Enhancements Port**

* **Tag:** `ghc-8.10.7-eventlog-enhancements`

  * Port of eventlog-based counter work from 8.8.4.

#### ↳ **Primop Enhancements**

* **Tag:** `ghc-8.10.7-primop-enhancements`

  * Port of in-memory performance counter collection using primops.

##### ↳ **Debug Enhancements**

* **Tag:** `ghc-8.10.7-debug-enhancements`

  * Added tracing hooks in `forkOn`.
  * Purpose:

    * Track and account for all threads created
    * Covers both:

      * GHC runtime-created threads
      * User-created threads

---

## **GHC 9.2.8**

### **File Lock Debugging**

* **Tag:** `ghc-9.2.8-file-lock-debug`

  * Introduced debugging code for a file locking bug in GHC.

### **Performance Counters (threadCPUTime#)**

* **Tag:** `ghc-9.2.8-perf-counters-1-rc1`

  * Port of the 8.10.7 `primop-enhancements` work.

#### ↳ **Memory Leak Profiling**

* **Tag:** `ghc-9.2.8-leak-profiling-1-rc1`

  * Added work related to memory leak profiling.
