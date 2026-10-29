import os
import sys
from django.apps import AppConfig


class AssessmentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'assessment'

    def ready(self):
        # Check if running via manage.py CLI
        is_manage_py = len(sys.argv) > 0 and 'manage.py' in os.path.basename(sys.argv[0])
        if is_manage_py:
            # Check if this is runserver
            subcommands = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
            is_runserver = len(subcommands) > 0 and subcommands[0] == 'runserver'
            if not is_runserver:
                # Any management task (migrate, check, test, makemigrations, etc.)
                return
            # For runserver with autoreload, only preload in the serving worker
            # (RUN_MAIN == 'true'), completely bypassing the supervisor monitor process.
            # If --noreload is passed, preload directly.
            has_noreload = '--noreload' in sys.argv
            is_worker = os.environ.get('RUN_MAIN') == 'true'
            if not (has_noreload or is_worker):
                return

        # Start asynchronous background model preloading & graph warmup
        from .ml_model import warm_up_in_background
        warm_up_in_background()
