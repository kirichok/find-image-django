from __future__ import unicode_literals

from django.apps import AppConfig
import uploads.core.matching.kie_find_by_hash as finder


class CoreConfig(AppConfig):
    name = 'uploads.core'

    def ready(self):
        print('Started loading the hash files to RAM')
        finder.loadHashFiles('/home/user/py-compare-images/images/hash/', True, 100, '.des.jpg')
        print('Successful loaded the hash files to RAM')
