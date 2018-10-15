---
title: "Hinting device suitability: from a hardware agnostic perspective"
abstract: "

Measuring performance-critical characteristics of application workloads is important both for developers, who must understand and optimize the performance of codes, as well as designers and integrators of HPC systems, who must ensure that compute architectures are suitable for the intended workloads.
However, if these workload characteristics are tied to architectural features that are specific to a particular system, they may not generalize well to alternative or future systems.
An architecture-independent method ensures an accurate characterization of inherent program behaviour, without bias due to architecture-dependent features that vary widely between different types of accelerators.

This work presents the first architecture-independent workload characterization framework for heterogeneous compute platforms, proposing a set of metrics determining the suitability and performance of an application on any parallel HPC architecture.
The tool, AIWC, is capable of characterizing OpenCL workloads currently in use in the supercomputing setting and is deployed as part of the open-source Oclgrind simulator.
AIWC simulates an OpenCL device by directly interpreting LLVM instructions, and the resulting metrics may be used for performance prediction and developer feedback to guide device-specific optimizations.
An evaluation of the metrics collected over a subset of the Extended OpenDwarfs Benchmark Suite is also presented.
"
keywords: "workload characterization, benchmarking, HPC"
date: "`r format(Sys.time(), '%B %d, %Y')`"
bibliography: ./bibliography/bibliography.bib
---

<!--IEEE needs the keywords to be set here :(-->
\iftoggle{IEEE-BUILD}{
\begin{IEEEkeywords}
workload characterisation, analysis
\end{IEEEkeywords}
}{}



----------------------------------------------------------------

Porting large HPC codes, such as those seen in weather forecasting and othes supercomputing workloads, from conventional CPU architectures to accelerators is intensive on the developer.
Even heterogeneous languages -- like OpenCL -- which support having a single implementation of a code often don't alleviate this process since algorithmic optimisations are needed to fully utilise the selected accelerator hardware.
Worst still, the classic methods of focusing optimisation effort on the longest running kernels is often cumbersome and wasteful -- consider cases where kernels are quite already efficient but the low hanging fruit are missed, for some of the worst kernels which have shorter running times have simple optimisations which matter once the energy usage, operating costs and size of these systems are considered.
We propose a methodology to examine a kernels suitability/goodness-of-fit to a desired accelerator by examining the inherent properties of the workflow, with the aims of guiding the optimisation methods of the developer.
The methodlogy is evaluated by comparing the suggested programming practices of CPU and GPU specific algorithmic optimisations, on portable OpenCL codes, and how architectural independent analysis can identify poor adherence to these practices.
A selection of each of these practices was taken from their respective source code and ported to OpenCL.
The practices for CPU devices was taken from the "Intel 64 and IA-32 Architectures Optimization Reference Manual", while the GPU practices were based on the "CUDA C Best Practices Guide, Design Guide"

\todo[inline]{cite previous works}
AIWC is used to perform the analysis.
Executed runtimes are collected by timing the codes on selected devices.

\todo[inline]{predictive model? should we add it to this paper? does it add anything?}

# AIWC Extensions and Derived Metrics

The tool AIWC was originally built for automated device selection.
This differs to the new goal of guiding a developer for kernel optimization.
For device selection, a large number of features were selected to find any meaningful workload characteristics to form the basis of a predictive model.
However, statistical models, such as the random forest model -- used with AIWC previously [ref @aiwc] -- excels at identifying the and can handle redundencies over many dimensions of data.
Developers require a smaller subset of AIWC metrics to compare when considering optimisation, as such additional derived metrics and extensions to the AIWC plugin are described.

\todo[inline]{fix this reference to the DRSN HPCS paper}

## Communication

The **Communication** metric shows the fraction of memory transfers / the number of operations executed by a kernel.
It has an implicit barrier between kernel invocations.
To computed this metric extensions around *host to device* and *device to host* was added to AIWC and correspond to the `clEnqueueWriteBuffer` and `clEnqueueReadBuffer` OpenCL functions.
Since, these memory transfers occur outside of OpenCL kernels, these extensions were added to the global scope of the plugin and are written to file during the destruction of the AIWC plugin -- during the termination of the Oclgrind application.
To implement this extension, two new vectors were added and a global variable.
These are responsible for tracking the name of kernel succeeding the host to device transfer, the name of kernel preceding the device to host transfer and the name of the last encountered kernel, respectively.
The mapping of memory transfers to kernel is non-trivial and the following assumption was made: ``data is never transferred from the device to the host before a kernel is executed''.
This assumption allows the correction of the kernel name for *device to host* transfers by shifting their occurance in the *host to device* vector, and is needed since the name of the kernel region is only known by the time of a *device to host* operation.
The directly computed metrics are presented as the number of *device to host* and *host to device* transfers surrounding each kernel.
A caveat of this approach is, since the kernels are unaware of the number of memory transfers surrounding it, the derivation of the **Communication** metric must be performed outside of the AIWC tool.
This is performed automatically during the statistical processing scripts presented in the associated artefact.

\todo[inline]{cite the artefact}

## Utilization / Occupancy

Resource pressure is computed as the number of registers needed to complete the kernel implementation of the algorithm.
However, since there are no registers used in LLVM intermediate representation -- indeed, this is one of the final stages of compiler optimization required where performance is device critical.
Instead of examining the final machine codes used, we examine the number of labels -- or unique instruction names -- required to complete a kernel.
This higher level of abstraction serves as a baseline measure of resource pressure, since it provides the maximum possible number of registers required to complete the kernel execution.
A caveat of this approach is that it is an abstraction and as such free from any vendor specific compiler optimisation which would attempt to minimize the number of registers required to finish completion.

The number of threads active is also device specific, we instead present the granularity of the algorithim, which shows the degree of parallelism.
This is an indirect measure of threads active since we show that the algorithm can support this amount of concurrent processing, which in turn can identify bottlenecks in either the target architecture -- if it supports fewer cores than there are threads -- or vice versa.
  

# CPU best practices

# GPU best practices

Performance optimization revolves around three basic strategies:

1) Maximizing parallel execution,
2) Optimizing memory usage to achieve maximum memory bandwidth
3) Optimizing instruction usage to achieve maximum instruction throughput

The focus of strategy 1) is to ensure as much data parallelism is exposed as possible, the suitabilty of a kernel mapping effectively on a GPU device is thus expected to be shown by having a high "parallelism" AIWC metrics. \todo[inline]{look at the list and speculate around the principal metrics.}

# Evaluation

Per kernel breakdown

## Case Study 1

## Case Study 2

## Case Study 3

# Automatic Suggestions

If AIWC metric *x* falls within range *y* we can conclude the algorithm is best suited to *z* devices.
However, we can also speculate around the ideal range for each device, or at least identify the threshold around what is suitable \todo[inline]{how do we suggest whether to go up or down an AIWC metric range -- or which code characteristics improve this?}

\todo[inline]{study of two different matrix multiply codes that examine how AIWC metrics change over cache-critical and cache-oblivious workloads}
