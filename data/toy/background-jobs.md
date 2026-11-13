---
title: Background Jobs
url: https://quokka.example/docs/background-jobs/
---
# Background Jobs

Quokka is a fictional task-queue library used only to test this repository.

## Enqueue a job

Call `queue.enqueue(func, *args)` to schedule a function. The call returns a `Job` object immediately and the function runs later in a worker process.

```python
from quokka import Queue

queue = Queue("emails")
job = queue.enqueue(send_welcome_email, user_id=42)
```

## Retries

A failed job is retried up to 3 times by default. Set `max_retries` on `enqueue` to change it. The delay between attempts doubles each time, starting at 2 seconds.

## Job timeout

Each job has a default timeout of 180 seconds. A job that runs longer is stopped and marked as failed.
