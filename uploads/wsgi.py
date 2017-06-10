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
# finder.loadHashFiles(120000, '/home/user/py-compare-images/images/hash/', '.des')
finder.loadHashFiles(55000, '/home/evgeniy/projects/python/open-cv/images/hash/', '.des')
print('Successful loaded the hash files to RAM')