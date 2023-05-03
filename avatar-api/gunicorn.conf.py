# https://github.com/benoitc/gunicorn/blob/master/examples/example_config.py

# Bind & deployment

bind = '0.0.0.0:5000'
reload = False

# Connections
# The dashboard backend should be capable of supporting multiple workers,
# however initialization is currently an issue when running in multiple threads.
workers = 1  # if DashboardConfig.debug else 4
threads = 4
backlog = 64
timeout = 300

# Logging
loglevel = 'info'
