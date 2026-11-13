---
title: Configuration
url: https://quokka.example/docs/configuration/
---
# Configuration

## Broker URL

Quokka reads the broker address from the `QUOKKA_BROKER_URL` environment variable. Only Redis brokers are supported, for example `redis://localhost:6379/0`.

## Result storage

Job results are kept for 24 hours by default. Set `QUOKKA_RESULT_TTL` to a number of seconds to change how long results are stored.

## Logging

Quokka logs to standard error at the INFO level. Set `QUOKKA_LOG_LEVEL=DEBUG` for more detail.
