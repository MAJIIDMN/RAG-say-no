---
title: Workers
url: https://quokka.example/docs/workers/
---
# Workers

## Start a worker

Run `quokka worker emails` to start a worker that listens on the `emails` queue. A worker can listen on several queues by listing them, for example `quokka worker emails reports`.

## Concurrency

The `--concurrency` option sets how many jobs a worker runs at the same time. The default is the number of CPU cores.

## Graceful shutdown

When a worker receives SIGTERM it stops taking new jobs and waits up to 30 seconds for running jobs to finish before exiting.
