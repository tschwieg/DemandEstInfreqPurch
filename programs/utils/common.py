# -*- coding: utf-8 -*-

from __future__ import division, print_function
from sys import getsizeof, stdout, argv, version_info
from collections import Iterable as Iter
from os import path, linesep, makedirs
from numpy import dot

if version_info >= (3, 0):
    from itertools import reduce

def makedirsTry(pathTry, warn_exists = False):

    """Call makedirs but don't throw an error if target exists"""

    try:
        makedirs(pathTry)
    except OSError:
        if not path.isdir(pathTry):
            raise
        elif warn_exists:
            print("WARNING: '{0}' exists!".format(pathTry))


class Logger():

    """Hack to output everything printed to a log"""

    def __init__(self, logfile = None):
        if logfile is None:
            logfile = path.splitext(argv[0])[0] + '.log'

        self.terminal = stdout
        self.log = open(logfile, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def whos(scope,
         types = None,
         sort = True,
         listOnly = False,
         notStartsWith = '_',
         maxObjects = 20):
    """Print the name, size, and type of objects in memory

    Args:
        scope (dict): Dictionary with scope of objects.

    Kwargs:
        types (list): List of types to print. Default: None (all types)
        sort (bool): Whether to reverse sort the objects in memory by size. Default: True.
        listOnly (bool): Only list objects in memory (without totals). Default: False.
        notStartsWith (str): Skip if stars with this character (e.g. '_' are often system objects)
        maxObjects (int): Maximum number of objects to print. Default: 20 (None prints all).

    Returns: Prints memory occupied by objects in requested scope.

    """
    if scope is None:
        # globalKeys = globals().keys() + locals().keys()
        globalKeys = globals().keys()
    else:
        globalKeys = scope.keys()

    maxKeyLen  = 0
    for gKey in globalKeys:
        maxKeyLen = maxKeyLen if maxKeyLen > len(gKey) else len(gKey)

    strWhos   = "{0:%s}" % maxKeyLen
    keepSizes = []
    keepKeys  = []
    keepTypes = []
    if types is None:
        if scope is None:
            # for var, obj in globals().items() + locals().items():
            for var, obj in globals().items():
                if notStartsWith is not None:
                    if var.startswith(notStartsWith):
                        continue
                try:
                    keepSizes += [getsizeof(obj)]
                except:
                    keepSizes += [None]
                keepKeys  += [var]
                keepTypes += [type(obj).__name__]
        else:
            for var, obj in scope.items():
                if notStartsWith is not None:
                    if var.startswith(notStartsWith):
                        continue
                try:
                    keepSizes += [getsizeof(obj)]
                except:
                    keepSizes += [None]
                keepKeys  += [var]
                keepTypes += [type(obj).__name__]
    else:
        if scope is None:
            # for var, obj in globals().items() + locals().items():
            for var, obj in globals().items():
                if notStartsWith is not None:
                    if var.startswith(notStartsWith):
                        continue
                if type(obj) in types:
                    try:
                        keepSizes += [getsizeof(obj)]
                    except:
                        keepSizes += [None]
                    keepKeys  += [var]
                    keepTypes += [type(obj).__name__]
        else:
            for var, obj in scope.items():
                if notStartsWith is not None:
                    if var.startswith(notStartsWith):
                        continue
                if type(obj) in types:
                    try:
                        keepSizes += [getsizeof(obj)]
                    except:
                        keepSizes += [None]
                    keepKeys  += [var]
                    keepTypes += [type(obj).__name__]

    if sort:
        keepKeys  = [x for (y, x) in sorted(zip(keepSizes, keepKeys))]
        keepTypes = [x for (y, x) in sorted(zip(keepSizes, keepTypes))]
        keepKeys.reverse()
        keepTypes.reverse()
        keepSizes.sort()
        keepSizes.reverse()

    totalSize    = 0
    totalObjects = 0
    totalSkipped = 0
    listInfo     = []
    sepLength    = 0
    for objectName, sizeInBytes, objectType in zip(keepKeys, keepSizes, keepTypes):
        if sizeInBytes is None:
            strSize = "          NaN      "
            totalSkipped += 1
        else:
            totalSize    += sizeInBytes
            totalObjects += 1
            if sizeInBytes < 1024:
                strSize = "    {0:9.1f} bytes".format(sizeInBytes)
            elif sizeInBytes < 1024 ** 2:
                strSize = "    {0:9.1f} KiB  ".format(sizeInBytes / 1024)
            elif sizeInBytes < 1024 ** 3:
                strSize = "    {0:9.1f} MiB  ".format(sizeInBytes / 1024 ** 2)
            else:
                strSize = "    {0:9.1f} GiB  ".format(sizeInBytes / 1024 ** 3)

        listInfo += [strWhos.format(objectName) + strSize + "        " + objectType]

    printList = (maxObjects > 0 or maxObjects is None) and len(listInfo) > 0
    if maxObjects is not None and len(listInfo) > maxObjects:
        excluded = len(listInfo) - maxObjects
        listInfo = listInfo[:maxObjects] + ["... and {0} more objects".format(excluded)]

    for info in listInfo:
        sepLength = sepLength if sepLength > len(info) else len(info)

    strSep = sepLength * '-'
    if listOnly and printList:
        print(linesep.join([strSep] + listInfo + [strSep]))
    elif listOnly and not printList:
        print("Cannot only print {0} objects; specify maxObjects > 0.".format(maxObjects))
    else:
        if totalSize < 1024:
            strTotal = "{0:.2f} bytes".format(totalSize)
        elif totalSize < 1024 ** 2:
            strTotal = "{0:.2f} KiB".format(totalSize / 1024)
        elif totalSize < 1024 ** 3:
            strTotal = "{0:.2f} MiB".format(totalSize / 1024 ** 2)
        else:
            strTotal = "{0:.2f} GiB".format(totalSize / 1024 ** 3)

        msgTotal = ["{0:,} objects in memory occupying {1}.".format(totalObjects, strTotal)]
        if totalSkipped > 1:
            msgTotal += ["Note: A further {0:,} items' size could not be tallied.".format(totalSkipped)]
        elif totalSkipped == 1:
            msgTotal += ["Note: A further 1 item's size could not be tallied."]

        if types is not None:
            if len(types) > 1:
                msgTotal += ["Tallied objects of type:"]
                for strType in types:
                    msgTotal += ["    - " + str(strType)]
            else:
                msgTotal += ['Tallied objects of type "{0}"'.format(str(types[0]))]

        print(linesep.join(msgTotal))
        if printList:
            print(linesep.join([strSep] + listInfo + [strSep]))


# Backwards-compatible list flattening
# http://stackoverflow.com/questions/2158395/
def flatten(l):
    """Flatten ist l"""
    if version_info >= (3, 0):
        for el in l:
            if isinstance(el, Iter) and not isinstance(el, (str, bytes)):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
    else:
        for el in l:
            if isinstance(el, Iter) and not isinstance(el, basestring):
                for sub in flatten(el):
                    yield sub
            else:
                yield el


def mdot(*args):
    return reduce(lambda x, y: dot(y, x), args[::-1])
