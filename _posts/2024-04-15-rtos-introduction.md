---
layout: post
title: "Introduction to Real-Time Operating Systems"
date: 2024-04-15
categories: [embedded]
excerpt: "An overview of RTOS fundamentals — what makes an OS 'real-time', key properties, and common implementations."
---

# Introduction to Real-Time Operating Systems

A Real-Time Operating System (RTOS) is a specialized OS designed to respond to external events within strict timing constraints. Unlike general-purpose operating systems, an RTOS prioritizes deterministic response time over average throughput.

## What Makes It "Real-Time"?

The defining characteristic is **determinism** (确定性) — the system must guarantee that tasks complete within specified deadlines. Missing a deadline is considered a system failure.

## Key Properties

1. **Determinism** — bounded worst-case execution time
2. **Predictability** — consistent timing behavior
3. **Priority-based scheduling** — higher-priority tasks preempt lower ones
4. **Fast interrupt handling** — minimal interrupt latency
5. **Task management** — lightweight context switching

## Common RTOS Implementations

- **FreeRTOS** — widely used in IoT and embedded devices
- **RT-Thread** — popular in the Chinese embedded ecosystem (国内嵌入式生态)
- **uC/OS** — classic educational and commercial RTOS
- **QNX** — used in automotive and safety-critical systems

## Application Domains

- Industrial control systems
- Medical devices
- Aerospace and defense
- Automotive electronics

Future posts will explore RTOS internals — task scheduling, memory management, and inter-task communication.
