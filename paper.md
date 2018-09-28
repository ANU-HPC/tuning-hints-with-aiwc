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

#CPU best practices

#GPU best practices

Performance optimization revolves around three basic strategies:

1) Maximizing parallel execution,
2) Optimizing memory usage to achieve maximum memory bandwidth
3) Optimizing instruction usage to achieve maximum instruction throughput

The focus of strategy 1) is to ensure as much data parallelism is exposed as possible, the suitabilty of a kernel mapping effectively on a GPU device is thus expected to be shown by having a high "parallelism" AIWC metrics. \todo[inline]{look at the list and speculate around the principal metrics.}

#Evaluation

Per kernel breakdown

##Case Study 1

##Case Study 2

##Case Study 3

#Automatic Suggestions

If AIWC metric *x* falls within range *y* we can conclude the algorithm is best suited to *z* devices.
However, we can also speculate around the ideal range for each device, or at least identify the threshold around what is suitable \todo[inline]{how do we suggest whether to go up or down an AIWC metric range -- or which code characteristics improve this?}

\todo[inline]{study of two different matrix multiply codes that examine how AIWC metrics change over cache-critical and cache-oblivious workloads}
