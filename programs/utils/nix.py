# -*- coding: utf-8 -*-

from __future__ import division, print_function
from os import system

def fixUnixPermissions(pathFix, group):
    """Quick script to recursively chown path to group and allow write-access

    Args:
        pathFix (str): Path to chown to group
        group (str): Group to chown to

    Returns: Chowns path to group and allows write-access

    """
    system('chown -R :{0} {1}'.format(group, pathFix))
    system("find {0} -exec chown :{1} '{{}}' \;".format(pathFix, group))
    system("find {0} -type d -exec chmod 775 '{{}}' \;".format(pathFix))
    system("find {0} -type f -exec chmod g+rw '{{}}' \;".format(pathFix))
    system("find {0} -type l -exec chmod g+rw '{{}}' \;".format(pathFix))
