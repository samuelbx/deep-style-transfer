import logging

def get_logger():
  console_logs_lvl = logging.DEBUG  # Options: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
  console_logs_format = (
    "%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s"
  )

  logger = logging.getLogger(__name__)
  logger.setLevel(console_logs_lvl)
  logging.basicConfig(format=console_logs_format)

  file_logs_lvl = logging.INFO
  file_logs_format = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
  file_handler = logging.FileHandler("logs.txt", mode="w")
  file_handler.setLevel(file_logs_lvl)
  file_handler.setFormatter(logging.Formatter(file_logs_format))
  logger.addHandler(file_handler)

  return logger