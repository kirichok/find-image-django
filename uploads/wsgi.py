"""
WSGI config for uploads project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.9/howto/deployment/wsgi/
"""

import os
import uploads.core.matching.kie_find_by_hash as finder

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "uploads.settings")

application = get_wsgi_application()

print('Started loading the hash files to RAM')
finder.loadHashFiles('/home/user/py-compare-images/images/hash/', True, 100, '.jpg')
print('Successful loaded the hash files to RAM')