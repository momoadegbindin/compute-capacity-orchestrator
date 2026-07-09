# Terminology
This document explains project terms in plain language.

## GPU capacity
The amount of accelerator compute available to run jobs.

## Job
A request for compute. In this project, a job has a release time, duration, GPU demand, priority, and optional deadline.

## Queue
The set of jobs waiting for capacity.

## Scheduler
The decision rule that chooses which queued jobs start now and where they run.

## Utilization
The fraction of available GPUs currently used by running jobs.

## Deadline risk
A signal that a job may finish later than its target time if it waits too long.

### bounded-time decisions
Schedulers must return an answer quickly; it cannot think forever.

### congestion
Too many jobs waiting for too little GPU capacity.

### workload-mix changes
Today mostly inference jobs, tomorrow mostly training jobs, etc.

### capacity shocks
Some GPUs become unavailable, or capacity drops suddenly.