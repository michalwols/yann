import os


class ConfigError(Exception):
  pass


class Vars:
  def __init__(self, env_prefix=None):
    self.env_prefix = env_prefix

  def __call__(self, option, required=False, cast=None, null=False):
    env_var_name = '_'.join((self.env_prefix, option)) \
      if self.env_prefix else option
    if env_var_name in os.environ:
      val = os.environ[env_var_name]
      val = cast(val) if cast else val
      if not null and not val:
        raise ConfigError(
          f"{option} cannot be null"
        )
      return val

    if hasattr(self, option):
      val = getattr(self, option)
      if not null and not val:
        raise ConfigError(
          f"{option} cannot be null"
        )
      return val

    if required:
      raise ConfigError(
        f"Required configuration value was not provided ({env_var_name})"
      )

  def __contains__(self, item):
    return item in os.environ or hasattr(self, item)


class YannConfig(Vars):
  BATCH_SIZE = 34
