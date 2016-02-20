#!/usr/bin/env python

description = """Convert datreant Treant, Group and mdsynthesis Sim state files
              from the deprecated pytables HDF5 format to the current JSON 
              format.

              """

import os
import sys
import json
import fcntl
import logging
import warnings
from functools import wraps

import tables
import numpy as np

from datreant.core import Treant, Group


class File(object):
    """File object base class. Implements file locking and reloading methods.

    """

    def __init__(self, filename, **kwargs):
        """Create File instance for interacting with file on disk.

        All files in datreant should be accessible by high-level methods
        without having to worry about simultaneous reading and writing by other
        processes. The File object includes methods and infrastructure for
        ensuring shared and exclusive locks are consistently applied before
        reads and writes, respectively. It handles any other low-level tasks
        for maintaining file integrity.

        :Arguments:
            *filename*
                name of file on disk object corresponds to

        """
        self.filename = os.path.abspath(filename)
        self.handle = None

    def get_location(self):
        """Get File basedir.

        :Returns:
            *location*
                absolute path to File basedir

        """
        return os.path.dirname(self.filename)

    @staticmethod
    def _read(func):
        """Decorator for opening state file for reading and applying shared
        lock.

        Applying this decorator to a method will ensure that the file is opened
        for reading and that a shared lock is obtained before that method is
        executed. It also ensures that the lock is removed and the file closed
        after the method returns.

        """
        @wraps(func)
        def inner(self, *args, **kwargs):
            # open the file using the actual reader
            self.handle = self._open_file_r()
            try:
                out = func(self, *args, **kwargs)
            finally:
                self.handle.close()

            return out

        return inner

    @staticmethod
    def _write(func):
        """Decorator for opening state file for writing and applying exclusive lock.

        Applying this decorator to a method will ensure that the file is opened
        for appending and that an exclusive lock is obtained before that method
        is executed. It also ensures that the lock is removed and the file
        closed after the method returns.

        """
        @wraps(func)
        def inner(self, *args, **kwargs):
            self.handle = self._open_file_w()
            try:
                out = func(self, *args, **kwargs)
            finally:
                self.handle.close()

            return out

        return inner

    def _open_r(self):
        """Open file with intention to write.

        Not to be used except for debugging files.

        """
        self.handle = self._open_file_r()

    def _open_w(self):
        """Open file with intention to write.

        Not to be used except for debugging files.

        """
        self.handle = self._open_file_w()

    def _close(self):
        """Close file.

        Not to be used except for debugging files.

        """
        self.handle.close()


class TreantFileHDF5(File):
    """Treant file object; syncronized access to Treant data.

    """
    def __init__(self, filename, **kwargs):
        """Initialize Treant state file.

        This is the base class for all Treant state files. It generates data
        structure elements common to all Treants. It also implements
        low-level I/O functionality.

        :Arguments:
            *filename*
                path to file

        """
        # filter NaturalNameWarnings from pytables, when they arrive
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

        super(TreantFileHDF5, self).__init__(filename)

    def _open_file_r(self):
        return tables.open_file(self.filename, 'r')

    def _open_file_w(self):
        return tables.open_file(self.filename, 'a')

    @File._read
    def get_version(self):
        """Get Treant version.

        :Returns:
            *version*
                version of Treant

        """
        table = self.handle.get_node('/', 'version')
        return table.cols.version[0]

    @File._read
    def get_tags(self):
        """Get all tags as a list.

        :Returns:
            *tags*
                list of all tags
        """
        table = self.handle.get_node('/', 'tags')
        return [x['tag'] for x in table.read()]

    @File._read
    def get_categories(self):
        """Get all categories as a dictionary.

        :Returns:
            *categories*
                dictionary of all categories
        """
        table = self.handle.get_node('/', 'categories')
        return {x['category']: x['value'] for x in table.read()}


class GroupFileHDF5(TreantFileHDF5):
    """Main Group state file.

    This file contains all the information needed to store the state of a
    Group object. It includes accessors, setters, and modifiers for all
    elements of the data structure, as well as the data structure definition.

    """
    # add new paths to include them in member searches
    memberpaths = ['abspath', 'relCont']

    def __init__(self, filename, **kwargs):
        """Initialize Group state file.

        :Arguments:
           *filename*
              path to file

        """
        super(GroupFileHDF5, self).__init__(filename, **kwargs)

    @File._read
    def get_member(self, uuid):
        """Get all stored information on the specified member.

        Returns a dictionary whose keys are column names and values the
        corresponding values for the member.

        :Arguments:
            *uuid*
                uuid of the member to retrieve information for

        :Returns:
            *memberinfo*
                a dictionary containing all information stored for the
                specified member
        """
        table = self.handle.get_node('/', 'members')
        fields = table.dtype.names

        memberinfo = None
        for row in table.where("uuid == '{}'".format(uuid)):
            memberinfo = row.fetch_all_fields()

        if memberinfo:
            memberinfo = {x: y for x, y in zip(fields, memberinfo)}

        return memberinfo

    @File._read
    def get_members(self):
        """Get full member table.

        Sometimes it is useful to read the whole member table in one go instead
        of doing multiple reads.

        :Returns:
            *memberdata*
                structured array giving full member data, with
                each row corresponding to a member
        """
        table = self.handle.get_node('/', 'members')
        return table.read()

    @File._read
    def get_members_uuid(self):
        """List uuid for each member.

        :Returns:
            *uuids*
                array giving treanttype of each member, in order
        """
        table = self.handle.get_node('/', 'members')
        return table.read()['uuid']

    @File._read
    def get_members_treanttype(self):
        """List treanttype for each member.

        :Returns:
            *treanttypes*
                array giving treanttype of each member, in order
        """
        table = self.handle.get_node('/', 'members')
        return table.read()['treanttype']

    @File._read
    def get_members_basedir(self):
        """List basedir for each member.

        :Returns:
            *basedirs*
                structured array containing all paths to member basedirs
        """
        table = self.handle.get_node('/', 'members')
        return table.read()[self.memberpaths]

mapping = {'Treant': [TreantFileHDF5, Treant],
           'Group':  [GroupFileHDF5, Group]}

class SimFileHDF5(TreantFileHDF5):
    """Main Sim state file.

    This file contains all the information needed to store the state of a
    Sim object. It includes accessors, setters, and modifiers for all
    elements of the data structure, as well as the data structure definition.

    """
    def __init__(self, filename, **kwargs):
        """Initialize Sim state file.

        :Arguments:
            *filename*
                path to file

        """
        super(SimFileHDF5, self).__init__(filename, **kwargs)

    @File._read
    def get_MDS_version(self):
        """Get Sim MDS version.

        :Returns:
            *version*
                MDS version of Treant

        """
        table = self.handle.get_node('/', 'mds_version')
        return table.cols.version[0]

    @File._read
    def get_default(self):
        """Get default universe.

        :Returns:
            *default*
                name of default universe; if no default
                universe, returns ``None``

        """
        table = self.handle.get_node('/', 'default')
        default = table.cols.default[0]

        if default == 'None':
            default = None

        return default

    @File._read
    def list_universes(self):
        """List universe names.

        :Returns:
            *universes*
                list giving names of all defined universes

        """
        group = self.handle.get_node('/', 'universes')

        return group.__members__

    @File._read
    def get_universe(self, universe):
        """Get topology and trajectory paths for the desired universe.

        Returns multiple path types, including absolute paths (abspath)
        and paths relative to the Sim object (relCont).

        :Arguments:
            *universe*
                given name for selecting the universe

        :Returns:
            *topology*
                structured array containing all paths to topology
            *trajectory*
                structured array containing all paths to trajectory(s)

        """
        try:
            # get topology file
            table = self.handle.get_node('/universes/{}'.format(universe),
                                         'topology')
            topology = table.read()

            # get trajectory files
            table = self.handle.get_node('/universes/{}'.format(universe),
                                         'trajectory')
            trajectory = table.read()

        except tables.NoSuchNodeError:
            raise KeyError(
                    "No such universe '{}'; add it first.".format(universe))

        return (topology, trajectory)

    @File._read
    def get_resnums(self, universe):
        """Get the resnum definition for the given universe.

        :Arguments:
            *universe*
                name of universe the resnum definition applies to

        :Returns:
            *resnums*
                list of the resnums for each atom in topology; None if
                no resnums defined
        """
        try:
            table = self.handle.get_node(
                '/universes/{}'.format(universe), 'resnums')
            resnums = [x['resnum'] for x in table.iterrows()]
        except tables.NoSuchNodeError:
            resnums = None

        return resnums

    @File._read
    def list_selections(self, universe):
        """List selection names.

        :Arguments:
            *universe*
                name of universe the selections apply to

        :Returns:
            *selections*
                list giving names of all defined selections for the given
                universe

        """
        try:
            group = self.handle.get_node(
                '/universes/{}'.format(universe), 'selections')
        except tables.NoSuchNodeError:
            raise KeyError("No such universe '{}';".format(universe) +
                           " cannot copy selections.")

        return group.__members__

    @File._read
    def get_selection(self, universe, handle):
        """Get a stored atom selection for the given universe.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name to use for the selection

        :Returns:
            *selection*
                list of the selection strings making up the atom selection
        """
        try:
            table = self.handle.get_node(
                '/universes/{}/selections'.format(universe), handle)
            selection = [x for x in table.read()]
        except tables.NoSuchNodeError:
            raise KeyError(
                    "No such selection '{}'; add it first.".format(handle))

        return selection

try: 
    from mdsynthesis.backends.statefiles import Sim
    mapping.update({'Sim': [SimFileHDF5, Sim]})
except ImportError:

    pass
if __name__ == '__main__':

    
    import argparse
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("statefile", metavar="STATEFILE", nargs='+')
    args = parser.parse_args()

    for sf in args.statefile:
        filename = os.path.basename(sf)
        dirname = os.path.dirname(sf)

        treanttype, uuid, ext = filename.split('.')

        # check that file meets criteria; if not, skip
        if ((treanttype not in ['Treant', 'Group', 'Sim']) or
           (ext != 'h5') or (len(uuid) != 36)):
            print "Unknown file '{}'; skipping".format(sf)
            continue

        print "Converting '{}' ...".format(sf)

        jsonfile = os.path.join(dirname, '.'.join((treanttype, uuid, 'json')))

        with open(jsonfile, 'w') as js:
            json.dump({}, js)

        h5_state = mapping[treanttype][0](sf)
        json_state = mapping[treanttype][1](jsonfile)

        # transfer generic TreantFile components
        json_state.tags.add(*h5_state.get_tags())
        json_state.categories.add(**h5_state.get_categories())

        # if a Group, get member records
        if treanttype == 'Group':
            for m_uuid in h5_state.get_members_uuid():
                memberdict = h5_state.get_member(m_uuid)

                json_state._add_member(m_uuid, memberdict['treanttype'],
                                      memberdict['abspath'])


        # if a Sim, get Universes, Selections, Default, Resnums
        if treanttype == 'Sim':
            for uname in h5_state.list_universes():
                top, traj = h5_state.get_universe(uname)

                json_state.universes.add(uname, top['abspath'][0],
                                        traj['abspath'])

                json_state.universes[uname]

                # selections
                for selname in h5_state.list_selections(uname):
                    sel = h5_state.get_selection(uname, selname)
                    json_state.selections.add(selname, *sel)

                # resnums
                json_state.resnums(uname, h5_state.get_resnums(uname))

            # default universe
            json_state.default = h5_state.get_default()

        print "Finished converting '{}'".format(sf)

    print "Finished converting all files."
