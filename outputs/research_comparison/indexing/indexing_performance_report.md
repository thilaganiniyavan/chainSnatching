# Research Comparison: Forensic Indexing Scalability

> **NOTE:** This is a SYNTHETIC SCALABILITY EXPERIMENT, not real CCTV pipeline results. It evaluates theoretical algorithmic scaling.

As the number of forensic events grows from 10 to 100,000, the Linear Scan query latency grows linearly (O(n)), rendering it unsuitable for large municipal deployments. The current Inverted Index maintains near O(1) lookup latency regardless of database size.
