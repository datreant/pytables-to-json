import os
import sys
import fcntl
import logging
import warnings
from functools import wraps

import tables
import numpy as np

import datreant
from .core import File


# max length in characters for all paths
pathlength = 511

# max character length of strings used for handles, tags, categories
namelength = 55


class File(object):
    """File object base class. Implements file locking and reloading methods.

    """

    def __init__(self, filename, logger=None, **kwargs):
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
            *logger*
                logger to send warnings and errors to

        """
        self.filename = os.path.abspath(filename)
        self.handle = None
        self.fd = None
        self.fdlock = None

        self._start_logger(logger)

        # we apply locks to a proxy file to avoid creating an HDF5 file
        # without an exclusive lock on something; important for multiprocessing
        proxy = "." + os.path.basename(self.filename) + ".proxy"
        self.proxy = os.path.join(os.path.dirname(self.filename), proxy)

        # we create the file if it doesn't exist; if it does, an exception is
        # raised and we catch it; this is necessary to ensure the file exists
        # so we can use it for locks
        try:
            fd = os.open(self.proxy, os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except OSError as e:
            # if we get the error precisely because the file exists, continue
            if e.errno == 17:
                pass
            else:
                raise

    def get_location(self):
        """Get File basedir.

        :Returns:
            *location*
                absolute path to File basedir

        """
        return os.path.dirname(self.filename)

    def _start_logger(self, logger):
        """Start up the logger.

        """
        # delete current logger
        try:
            del self.logger
        except AttributeError:
            pass

        # log to standard out if no logger given
        if not logger:
            self.logger = logging.getLogger(
                '{}'.format(self.__class__.__name__))
            self.logger.setLevel(logging.INFO)

            if not any([isinstance(x, logging.StreamHandler)
                        for x in self.logger.handlers]):
                ch = logging.StreamHandler(sys.stdout)
                cf = logging.Formatter(
                        '%(name)-12s: %(levelname)-8s %(message)s')
                ch.setFormatter(cf)
                self.logger.addHandler(ch)
        else:
            self.logger = logger

    def _shlock(self, fd):
        """Get shared lock on file.

        Using fcntl.lockf, a shared lock on the file is obtained. If an
        exclusive lock is already held on the file by another process,
        then the method waits until it can obtain the lock.

        :Arguments:
            *fd*
                file descriptor

        :Returns:
            *success*
                True if shared lock successfully obtained
        """
        fcntl.lockf(fd, fcntl.LOCK_SH)

        return True

    def _exlock(self, fd):
        """Get exclusive lock on file.

        Using fcntl.lockf, an exclusive lock on the file is obtained. If a
        shared or exclusive lock is already held on the file by another
        process, then the method waits until it can obtain the lock.

        :Arguments:
            *fd*
                file descriptor

        :Returns:
            *success*
                True if exclusive lock successfully obtained
        """
        fcntl.lockf(fd, fcntl.LOCK_EX)

        return True

    def _unlock(self, fd):
        """Remove exclusive or shared lock on file.

        WARNING: It is very rare that this is necessary, since a file must be
        unlocked before it is closed. Furthermore, locks disappear when a file
        is closed anyway.  This method will remain here for now, but may be
        removed in the future if not needed (likely).

        :Arguments:
            *fd*
                file descriptor

        :Returns:
            *success*
                True if lock removed
        """
        fcntl.lockf(fd, fcntl.LOCK_UN)

        return True

    def _open_fd_r(self):
        """Open read-only file descriptor for application of advisory locks.

        Because we need an active file descriptor to apply advisory locks to a
        file, and because we need to do this before opening a file with
        PyTables due to the risk of caching stale state on open, we open
        a separate file descriptor to the same file and apply the locks
        to it.

        """
        self.fd = os.open(self.proxy, os.O_RDONLY)

    def _open_fd_rw(self):
        """Open read-write file descriptor for application of advisory locks.

        """
        self.fd = os.open(self.proxy, os.O_RDWR)

    def _close_fd(self):
        """Close file descriptor used for application of advisory locks.

        """
        # close file descriptor for locks
        os.close(self.fd)
        self.fd = None

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
            if self.fdlock:
                out = func(self, *args, **kwargs)
            else:
                self._open_fd_r()
                self._shlock(self.fd)
                self.fdlock = 'shared'

                # open the file using the actual reader
                self.handle = self._open_file_r()
                try:
                    out = func(self, *args, **kwargs)
                finally:
                    self.handle.close()
                    self._unlock(self.fd)
                    self._close_fd()
                    self.fdlock = None
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
            if self.fdlock == 'exclusive':
                out = func(self, *args, **kwargs)
            else:
                self._open_fd_rw()
                self._exlock(self.fd)
                self.fdlock = 'exclusive'

                # open the file using the actual writer
                self.handle = self._open_file_w()
                try:
                    out = func(self, *args, **kwargs)
                finally:
                    self.handle.close()
                    self._unlock(self.fd)
                    self.fdlock = None
                    self._close_fd()
            return out

        return inner

    def _open_r(self):
        """Open file with intention to write.

        Not to be used except for debugging files.

        """
        self._open_fd_r()
        self._shlock(self.fd)
        self.fdlock = 'shared'
        self.handle = self._open_file_r()

    def _open_w(self):
        """Open file with intention to write.

        Not to be used except for debugging files.

        """
        self._open_fd_rw()
        self._exlock(self.fd)
        self.fdlock = 'exclusive'
        self.handle = self._open_file_w()

    def _close(self):
        """Close file.

        Not to be used except for debugging files.

        """
        self.handle.close()
        self._unlock(self.fd)
        self.fdlock = None
        self._close_fd()


class TreantFileHDF5(File):
    """Treant file object; syncronized access to Treant data.

    """
    class _Version(tables.IsDescription):
        """Table definition for storing version number of file schema.

        All strings limited to hardcoded size for now.

        """
        # version of datreant file schema corresponds to allows future-proofing
        # of old objects so that formats of new releases can be automatically
        # built from old ones
        version = tables.StringCol(15)

    class _Coordinator(tables.IsDescription):
        """Table definition for coordinator info.

        This information is kept separate from other metadata to allow the
        Coordinator to simply stack tables to populate its database. It doesn't
        need entries that store its own path.

        Path length fixed size for now.
        """
        # absolute path of coordinator
        abspath = tables.StringCol(pathlength)

    class _Tags(tables.IsDescription):
        """Table definition for tags.

        """
        tag = tables.StringCol(namelength)

    class _Categories(tables.IsDescription):
        """Table definition for categories.

        """
        category = tables.StringCol(namelength)
        value = tables.StringCol(namelength)

    def __init__(self, filename, logger=None, **kwargs):
        """Initialize Treant state file.

        This is the base class for all Treant state files. It generates data
        structure elements common to all Treants. It also implements
        low-level I/O functionality.

        :Arguments:
            *filename*
                path to file
            *logger*
                Treant's logger instance

        :Keywords:
            *treanttype*
                Treant type
            *coordinator*
                directory in which coordinator state file can be found [None]
            *categories*
                user-given dictionary with custom keys and values; used to
                give distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search
            *version*
                version of datreant file was generated with

        .. Note:: kwargs passed to :meth:`create`

        """
        # filter NaturalNameWarnings from pytables, when they arrive
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

        super(TreantFileHDF5, self).__init__(filename, logger=logger)

        # if file does not exist, it is created; if it does exist, it is
        # updated
        try:
            self.create(**kwargs)
        except OSError:
            # in case the file exists but is read-only; we can't update but may
            # still want to use it
            if os.path.exists(self.filename):
                pass
            # if the file doesn't exist, we still want an exception
            else:
                raise

    def _open_file_r(self):
        return tables.open_file(self.filename, 'r')

    def _open_file_w(self):
        return tables.open_file(self.filename, 'a')

    @File._write
    def create(self, **kwargs):
        """Build state file and common data structure elements.

        :Keywords:
            *coordinator*
                directory in which coordinator state file can be found [None]
            *categories*
                user-given dictionary with custom keys and values; used to
                give distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search
            *version*
                version of datreant file was generated with
        """
        # update schema and version of file
        version = self.update_schema()
        self.update_version(version)

        # coordinator table
        self.update_coordinator(kwargs.pop('coordinator', None))

        # tags table
        tags = kwargs.pop('tags', list())
        self.add_tags(*tags)

        # categories table
        categories = kwargs.pop('categories', dict())
        self.add_categories(**categories)

    @File._read
    def get_version(self):
        """Get Treant version.

        :Returns:
            *version*
                version of Treant

        """
        table = self.handle.get_node('/', 'version')
        return table.cols.version[0]

    # TODO: need a proper schema update mechanism
    @File._write
    def update_schema(self):
        """Update schema of file.

        :Returns:
            *version*
                version number of file's new schema
        """
        try:
            table = self.handle.get_node('/', 'version')
            version = table.cols.version[0]
        except tables.NoSuchNodeError:
            version = datreant.__version__

        return version

    @File._write
    def update_version(self, version):
        """Update version of Treant.

        :Arugments:
            *version*
                new version of Treant
        """
        try:
            table = self.handle.get_node('/', 'version')
            table.cols.version[0] = version
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'version', self._Version, 'version')
            table.row['version'] = version
            table.row.append()

    @File._read
    def get_coordinator(self):
        """Get absolute path to Coordinator.

        :Returns:
            *coordinator*
                absolute path to Coordinator directory

        """
        table = self.handle.get_node('/', 'coordinator')
        out = table.cols.abspath[0]

        if out == 'None':
            out = None
        return out

    @File._write
    def update_coordinator(self, coordinator):
        """Update Treant location.

        :Arguments:
            *coordinator*
                absolute path to Coordinator directory
        """
        try:
            table = self.handle.get_node('/', 'coordinator')
            if coordinator:
                table.cols.abspath[0] = os.path.abspath(coordinator)
            else:
                table.cols.abspath[0] = None
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'coordinator', self._Coordinator,
                'coordinator information')
            if coordinator:
                table.row['abspath'] = os.path.abspath(coordinator)
            else:
                table.row['abspath'] = None
            table.row.append()

    @File._read
    def get_tags(self):
        """Get all tags as a list.

        :Returns:
            *tags*
                list of all tags
        """
        table = self.handle.get_node('/', 'tags')
        return [x['tag'] for x in table.read()]

    @File._write
    def add_tags(self, *tags):
        """Add any number of tags to the Treant.

        Tags are individual strings that serve to differentiate Treants from
        one another. Sometimes preferable to categories.

        :Arguments:
            *tags*
                Tags to add. Must be convertable to strings using the str()
                builtin.

        """
        try:
            table = self.handle.get_node('/', 'tags')
        except tables.NoSuchNodeError:
            table = self.handle.create_table('/', 'tags', self._Tags, 'tags')

        # ensure tags are unique (we don't care about order)
        tags = set([str(tag) for tag in tags])

        # remove tags already present in metadata from list
        tags = tags.difference(set(table.read()['tag']))

        # add new tags
        for tag in tags:
            table.row['tag'] = tag
            table.row.append()

    @File._write
    def del_tags(self, *tags, **kwargs):
        """Delete tags from Treant.

        Any number of tags can be given as arguments, and these will be
        deleted.

        :Arguments:
            *tags*
                Tags to delete.

        :Keywords:
            *all*
                When True, delete all tags [``False``]

        """
        table = self.handle.get_node('/', 'tags')
        purge = kwargs.pop('all', False)

        if purge:
            table.remove()
            table = self.handle.create_table('/', 'tags', self._Tags, 'tags')

        else:
            # remove redundant tags from given list if present
            tags = set([str(tag) for tag in tags])

            # TODO: improve performance
            # get matching rows
            rowlist = list()
            for row in table:
                for tag in tags:
                    if (row['tag'] == tag):
                        rowlist.append(row.nrow)

            # must include a separate condition in case all rows will be
            # removed due to a limitation of PyTables
            if len(rowlist) == table.nrows:
                table.remove()
                table = self.handle.create_table(
                    '/', 'tags', self._Tags, 'tags')
            else:
                rowlist.sort()
                j = 0
                # delete matching rows; have to use j to shift the register as
                # we delete rows
                for i in rowlist:
                    table.remove_row(i - j)
                    j = j + 1

    @File._read
    def get_categories(self):
        """Get all categories as a dictionary.

        :Returns:
            *categories*
                dictionary of all categories
        """
        table = self.handle.get_node('/', 'categories')
        return {x['category']: x['value'] for x in table.read()}

    @File._write
    def add_categories(self, **categories):
        """Add any number of categories to the Treant.

        Categories are key-value pairs of strings that serve to differentiate
        Treants from one another. Sometimes preferable to tags.

        If a given category already exists (same key), the value given will
        replace the value for that category.

        :Keywords:
            *categories*
                Categories to add. Keyword used as key, value used as value.
                Both must be convertible to strings using the str() builtin.

        """
        try:
            table = self.handle.get_node('/', 'categories')
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'categories', self._Categories, 'categories')

        table = self.handle.get_node('/', 'categories')

        # remove categories already present in metadata from dictionary
        # TODO: more efficient way to do this?
        for row in table:
            for key in categories.keys():
                if (row['category'] == key):
                    row['value'] = str(categories[key])
                    row.update()
                    # dangerous? or not since we are iterating through
                    # categories.keys() and not categories?
                    categories.pop(key)

        # add new categories
        for key in categories.keys():
            table.row['category'] = key
            table.row['value'] = str(categories[key])
            table.row.append()

    @File._write
    def del_categories(self, *categories, **kwargs):
        """Delete categories from Treant.

        Any number of categories (keys) can be given as arguments, and these
        keys (with their values) will be deleted.

        :Arguments:
            *categories*
                Categories to delete.

        :Keywords:
            *all*
                When True, delete all categories [``False``]

        """
        table = self.handle.get_node('/', 'categories')
        purge = kwargs.pop('all', False)

        if purge:
            table.remove()
            table = self.handle.create_table(
                '/', 'categories', self._Categories, 'categories')
        else:
            # remove redundant categories from given list if present
            categories = set([str(category) for category in categories])

            # get matching rows
            rowlist = list()
            for row in table:
                for category in categories:
                    if (row['category'] == category):
                        rowlist.append(row.nrow)

            # must include a separate condition in case all rows will be
            # removed due to a limitation of PyTables
            if len(rowlist) == table.nrows:
                table.remove()
                table = self.handle.create_table(
                    '/', 'categories', self._Categories, 'categories')
            else:
                rowlist.sort()
                j = 0
                # delete matching rows; have to use j to shift the register as
                # we delete rows
                for i in rowlist:
                    table.remove_row(i - j)
                    j = j + 1


class GroupFileHDF5(TreantFileHDF5):
    """Main Group state file.

    This file contains all the information needed to store the state of a
    Group object. It includes accessors, setters, and modifiers for all
    elements of the data structure, as well as the data structure definition.

    """
    # add new paths to include them in member searches
    memberpaths = ['abspath', 'relCont']

    class _Members(tables.IsDescription):

        """Table definition for the members of the Group.

        Stores for each member its treant type, uuid, and two versions of
        the path to the member treant: the absolute path (abspath) and the
        relative path from the Group object's directory (relCont). This allows
        the Group object to use some heuristically good starting points when
        trying to find missing files using a Foxhound.

        """
        # unique identifier for treant
        uuid = tables.StringCol(uuidlength)

        # treant type
        treanttype = tables.StringCol(namelength)

        abspath = tables.StringCol(pathlength)
        relCont = tables.StringCol(pathlength)

    def __init__(self, filename, logger=None, **kwargs):
        """Initialize Group state file.

        :Arguments:
           *filename*
              path to file
           *logger*
              logger to send warnings and errors to

        :Keywords:
           *coordinator*
              directory in which coordinator state file can be found [None]
           *categories*
              user-given dictionary with custom keys and values; used to
              give distinguishing characteristics to object for search
           *tags*
              user-given list with custom elements; used to give distinguishing
              characteristics to object for search
        """
        super(GroupFileHDF5, self).__init__(filename, logger=logger, **kwargs)

    def create(self, **kwargs):
        """Build Group data structure.

        :Keywords:
           *coordinator*
              directory in which Coordinator state file can be found [``None``]
           *categories*
              user-given dictionary with custom keys and values; used to
              give distinguishing characteristics to object for search
           *tags*
              user-given list with custom elements; used to give distinguishing
              characteristics to object for search

        .. Note:: kwargs passed to :meth:`create`

        """
        super(GroupFileHDF5, self).create(treanttype='Group', **kwargs)

        self._make_membertable()

    @File._write
    def _make_membertable(self):
        """Make member table.

        Used only on file creation.

        """
        try:
            table = self.handle.get_node('/', 'members')
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'members', self._Members, 'members')

    @File._write
    def add_member(self, uuid, treanttype, basedir):
        """Add a member to the Group.

        If the member is already present, its basedir paths will be updated
        with the given basedir.

        :Arguments:
            *uuid*
                the uuid of the new member
            *treanttype*
                the treant type of the new member
            *basedir*
                basedir of the new member in the filesystem

        """
        try:
            table = self.handle.get_node('/', 'members')
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'members', self._Members, 'members')

        # check if uuid already present
        rownum = [row.nrow for row in table.where("uuid=='{}'".format(uuid))]
        if rownum:
            # if present, update location
            table.cols.abspath[rownum[0]] = os.path.abspath(basedir)
            table.cols.relCont[rownum[0]] = os.path.relpath(
                    basedir, self.get_location())
        else:
            table.row['uuid'] = uuid
            table.row['treanttype'] = treanttype
            table.row['abspath'] = os.path.abspath(basedir)
            table.row['relCont'] = os.path.relpath(
                    basedir, self.get_location())
            table.row.append()

    @File._write
    def del_member(self, *uuid, **kwargs):
        """Remove a member from the Group.

        :Arguments:
            *uuid*
                the uuid(s) of the member(s) to remove

        :Keywords:
            *all*
                When True, remove all members [``False``]

        """
        table = self.handle.get_node('/', 'members')
        purge = kwargs.pop('all', False)

        if purge:
            table.remove()
            table = self.handle.create_table(
                '/', 'members', self._Members, 'members')

        else:
            # remove redundant uuids from given list if present
            uuids = set([str(uid) for uid in uuid])

            # get matching rows
            # TODO: possibly faster to use table.where
            rowlist = list()
            for row in table:
                for uuid in uuids:
                    if (row['uuid'] == uuid):
                        rowlist.append(row.nrow)

            # must include a separate condition in case all rows will be
            # removed due to a limitation of PyTables
            if len(rowlist) == table.nrows:
                table.remove()
                table = self.handle.create_table(
                    '/', 'members', self._Members, 'members')
            else:
                rowlist.sort()
                j = 0
                # delete matching rows; have to use j to shift the register as
                # we delete rows
                for i in rowlist:
                    table.remove_row(i - j)
                    j = j + 1

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


class SimFileHDF5(TreantFile):
    """Main Sim state file.

    This file contains all the information needed to store the state of a
    Sim object. It includes accessors, setters, and modifiers for all
    elements of the data structure, as well as the data structure definition.

    """
    class _MDSversion(tables.IsDescription):
        """Table definition for storing version number of file schema.

        All strings limited to hardcoded size for now.

        """
        # version of MDS file schema corresponds to allows future-proofing
        # of old objects so that formats of new releases can be automatically
        # built from old ones
        version = tables.StringCol(15)

    class _Default(tables.IsDescription):
        """Table definition for storing default universe preference.

        Stores which universe is marked as default.

        """
        default = tables.StringCol(namelength)

    class _Topology(tables.IsDescription):
        """Table definition for storing universe topology paths.

        Two versions of the path to a topology are stored: the absolute path
        (abspath) and the relative path from the Sim object's directory
        (relCont). This allows the Sim object to use some heuristically good
        starting points when trying to find missing files using Finder.

        """
        abspath = tables.StringCol(pathlength)
        relCont = tables.StringCol(pathlength)

    class _Trajectory(tables.IsDescription):
        """Table definition for storing universe trajectory paths.

        The paths to trajectories used for generating the Universe
        are stored in this table.

        See UniverseTopology for path storage descriptions.

        """
        abspath = tables.StringCol(255)
        relCont = tables.StringCol(255)

    class _Resnums(tables.IsDescription):
        """Table definition for storing resnums.

        """
        resnum = tables.UInt32Col()

    def __init__(self, filename, logger=None, **kwargs):
        """Initialize Sim state file.

        :Arguments:
            *filename*
                path to file
            *logger*
                logger to send warnings and errors to

        :Keywords:
            *name*
                user-given name of Treant object
            *coordinator*
                directory in which coordinator state file can be found [None]
            *categories*
                user-given dictionary with custom keys and values; used to
                give distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search

        """
        super(SimFile, self).__init__(filename, logger=logger, **kwargs)

    def create(self, **kwargs):
        """Build Sim data structure.

        :Keywords:
            *name*
                user-given name of Sim object
            *coordinator*
                directory in which Coordinator state file can be found
                [``None``]
            *categories*
                user-given dictionary with custom keys and values; used to give
                distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search

        .. Note:: kwargs passed to :meth:`create`

        """
        super(SimFile, self).create(treanttype='Sim', **kwargs)

        self._make_universegroup()
        try:
            self.get_default()
        except tables.NoSuchNodeError:
            self.update_default()

    @File._write
    def _make_universegroup(self):
        """Make universes and universe groups.

        Intended for file initialization.

        """
        try:
            group = self.handle.get_node('/', 'universes')
        except tables.NoSuchNodeError:
            group = self.handle.create_group('/', 'universes', 'universes')

    @File._write
    def _make_default_table(self):
        """Make table for storing default universe.

        Used only on file creation.

        """
        try:
            table = self.handle.get_node('/', 'default')
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'default', self._Default, 'default')

    @File._read
    def get_MDS_version(self):
        """Get Sim MDS version.

        :Returns:
            *version*
                MDS version of Treant

        """
        table = self.handle.get_node('/', 'mds_version')
        return table.cols.version[0]

    # TODO: need a proper schema update mechanism
    @File._write
    def update_MDS_schema(self):
        """Update MDS schema of file.

        :Returns:
            *version*
                version number of file's new schema
        """
        try:
            table = self.handle.get_node('/', 'mds_version')
            version = table.cols.version[0]
        except tables.NoSuchNodeError:
            version = mdsynthesis.__version__

        return version

    @File._write
    def update_MDS_version(self, version):
        """Update MDS version of Sim.

        :Arugments:
            *version*
                new MDS version of Treant
        """
        try:
            table = self.handle.get_node('/', 'mds_version')
            table.cols.version[0] = version
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'mds_version', self._MDSversion, 'mds_version')
            table.row['version'] = version
            table.row.append()

    @File._write
    def update_default(self, universe=None):
        """Mark the given universe as the default.

        :Arguments:
            *universe*
                name of universe to mark as default; if ``None``,
                remove default preference
        """
        try:
            table = self.handle.get_node('/', 'default')
            table.cols.default[0] = universe
        except tables.NoSuchNodeError:
            table = self.handle.create_table(
                '/', 'default', self._Default, 'default')
            table.row['default'] = universe
            table.row.append()

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

    @File._write
    def add_universe(self, universe, topology, *trajectory):
        """Add a universe definition to the Sim object.

        A Universe is an MDAnalysis object that gives access to the details
        of a simulation trajectory. A Sim object can contain multiple universe
        definitions (topology and trajectory pairs), since it is often
        convenient to have different post-processed versions of the same
        raw trajectory.

        :Arguments:
            *universe*
                given name for selecting the universe
            *topology*
                path to the topology file
            *trajectory*
                path to the trajectory file; multiple files may be given
                and these will be used in order as frames for the trajectory

        """

        # build this universe's group; if it exists, do nothing
        try:
            group = self.handle.create_group(
                '/universes', universe, universe, createparents=True)
        except tables.NodeError:
            self.handle.remove_node(
                '/universes/{}'.format(universe), 'topology')
            self.handle.remove_node(
                '/universes/{}'.format(universe), 'trajectory')

        # construct topology table
        table = self.handle.create_table(
            '/universes/{}'.format(universe), 'topology', self._Topology,
            'topology')

        # add topology paths to table
        table.row['abspath'] = os.path.abspath(topology)
        table.row['relCont'] = os.path.relpath(topology, self.get_location())
        table.row.append()

        # construct trajectory table
        table = self.handle.create_table(
            '/universes/{}'.format(universe), 'trajectory', self._Trajectory,
            'trajectory')

        # add trajectory paths to table
        for segment in trajectory:
            table.row['abspath'] = os.path.abspath(segment)
            table.row['relCont'] = os.path.relpath(segment,
                                                   self.get_location())
            table.row.append()

        # construct selection group; necessary to catch NodError
        # exception when a Universe is re-added because selections are
        # maintained
        try:
            group = self.handle.create_group(
                '/universes/{}'.format(universe), 'selections', 'selections')
        except tables.NodeError:
            pass

    @File._write
    def del_universe(self, universe):
        """Delete a universe definition.

        Deletes any selections associated with the universe.

        :Arguments:
            *universe*
                name of universe to delete
        """
        try:
            self.handle.remove_node('/universes', universe, recursive=True)
        except tables.NoSuchNodeError:
            raise KeyError(
                    "No such universe '{}';".format(universe) +
                    " nothing to remove.")

    @File._write
    def rename_universe(self, universe, newname):
        """Rename a universe definition.

        :Arguments:
            *universe*
                name of universe to rename
            *newname*
                new name of universe
        """
        try:
            self.handle.rename_node('/universes', newname, name=universe)
        except tables.NoSuchNodeError:
            raise KeyError(
                    "No such universe '{}';".format(universe) +
                    " nothing to rename.")
        except tables.NodeError:
            raise ValueError(
                    "A universe '{}' already exists;".format(universe) +
                    " remove or rename it first.")

    @File._write
    def update_resnums(self, universe, resnums):
        """Update resnum definition for the given universe.

        Resnums are useful for referring to residues by their canonical resid,
        for instance that stored in the PDB. By giving a resnum definition
        for the universe, this definition can be applied to the universe
        on activation.

        Will overwrite existing definition if it exists.

        :Arguments:
            *universe*
                name of universe to associate resnums with
            *resnums*
                list giving the resnum for each atom in the topology, in index
                order
        """
        try:
            table = self.handle.create_table(
                '/universes/{}'.format(universe), 'resnums', self._Resnums,
                'resnums')
        except tables.NoSuchNodeError:
            self.logger.info(
                "Universe definition '{}'".format(universe) +
                " does not exist. Add it first.")
            return
        except tables.NodeError:
            self.logger.info(
                "Replacing existing resnums for '{}'.".format(universe))
            self.handle.remove_node(
                '/universes/{}'.format(universe), 'resnums')
            table = self.handle.create_table(
                '/universes/{}'.format(universe), 'resnums', self._Resnums,
                'resnums')

        # add resnums to table
        for item in resnums:
            table.row['resnum'] = item
            table.row.append()

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

    @File._write
    def del_resnums(self, universe):
        """Delete resnum definition from specified universe.

        :Arguments:
            *universe*
                name of universe to remove resnum definition from
        """
        self.handle.remove_node('/universes/{}'.format(universe), 'resnums')

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

    @File._write
    def add_selection(self, universe, handle, *selection):
        """Add an atom selection definition for the named Universe definition.

        AtomGroups are needed to obtain useful information from raw coordinate
        data. It is useful to store AtomGroup selections for later use, since
        they can be complex and atom order may matter.

        Will overwrite existing definition if it exists.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name to use for the selection
            *selection*
                selection string or numpy array of indices; multiple selections
                may be given and their order will be preserved, which is
                useful for e.g. structural alignments

        """
        # construct selection table
        if isinstance(selection[0], np.ndarray):
            selection = selection[0]

        try:
            array = self.handle.create_array(
                '/universes/{}/selections'.format(universe), handle, selection,
                handle)
        except tables.NodeError:
            self.logger.info(
                "Replacing existing selection '{}'.".format(handle))
            self.handle.remove_node(
                '/universes/{}/selections'.format(universe), handle)
            table = self.handle.create_array(
                '/universes/{}/selections'.format(universe), handle, selection,
                handle)

    @File._write
    def del_selection(self, universe, handle):
        """Delete an atom selection from the specified universe.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name of the selection

        """
        try:
            self.handle.remove_node(
                '/universes/{}/selections'.format(universe), handle)
        except tables.NoSuchNodeError:
            raise KeyError(
                    "No such selection '{}';".format(handle) +
                    " nothing to remove.")


class FileSerial(File):
    """File object base class for serialization formats, such as JSON.

    """
    def _open_file_r(self):
        return open(self.filename, 'r')

    def _open_file_w(self):
        return open(self.filename, 'w')

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
            if self.fdlock:
                out = func(self, *args, **kwargs)
            else:
                self._open_fd_r()
                self._shlock(self.fd)
                self.fdlock = 'shared'

                try:
                    out = func(self, *args, **kwargs)
                finally:
                    self._unlock(self.fd)
                    self._close_fd()
                    self.fdlock = None
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
            if self.fdlock == 'exclusive':
                out = func(self, *args, **kwargs)
            else:
                self._open_fd_rw()
                self._exlock(self.fd)
                self.fdlock = 'exclusive'

                try:
                    out = func(self, *args, **kwargs)
                finally:
                    self._unlock(self.fd)
                    self.fdlock = None
                    self._close_fd()
            return out

        return inner

    @staticmethod
    def _pull_push(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            try:
                self._pull_record()
            except IOError:
                self._init_record()
            out = func(self, *args, **kwargs)
            self._push_record()
            return out
        return inner

    @staticmethod
    def _pull(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            self._pull_record()
            out = func(self, *args, **kwargs)
            return out
        return inner

    def _pull_record(self):
        self.handle = self._open_file_r()
        self._record = self._deserialize(self.handle)
        self.handle.close()

    def _deserialize(self, handle):
        """Deserialize full record from open file handle.
        """
        raise NotImplementedError

    def _push_record(self):
        self.handle = self._open_file_w()
        self._serialize(self._record, self.handle)
        self.handle.close()

    def _serialize(self, record, handle):
        """Serialize full record to open file handle.
        """
        raise NotImplementedError


class TreantFile(MixinJSON, FileSerial):
    def __init__(self, filename, logger=None, **kwargs):
        """Initialize Treant state file.

        This is the base class for all Treant state files. It generates data
        structure elements common to all Treants. It also implements
        low-level I/O functionality.

        :Arguments:
            *filename*
                path to file
            *logger*
                Treant's logger instance

        :Keywords:
            *treanttype*
                Treant type
            *name*
                user-given name of Treant object
            *categories*
                user-given dictionary with custom keys and values; used to
                give distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search
            *version*
                version of datreant file was generated with

        .. Note:: kwargs passed to :meth:`create`

        """
        super(FileSerial, self).__init__(filename, logger=logger)

        # if file does not exist, it is created; if it does exist, it is
        # updated
        try:
            self.create(**kwargs)
        except OSError:
            # in case the file exists but is read-only; we can't update but may
            # still want to use it
            if os.path.exists(self.filename):
                pass
            # if the file doesn't exist, we still want an exception
            else:
                raise

    def _init_record(self):
        self._record = dict()
        self._record['tags'] = list()
        self._record['categories'] = dict()

    def create(self, **kwargs):
        """Build state file and common data structure elements.

        :Keywords:
            *name*
                user-given name of Treant object
            *categories*
                user-given dictionary with custom keys and values; used to
                give distinguishing characteristics to object for search
            *tags*
                user-given list with custom elements; used to give
                distinguishing characteristics to object for search
            *version*
                version of datreant file was generated with
        """
        # update schema and version of file
        version = self.update_schema()
        self.update_version(version)

        # tags table
        tags = kwargs.pop('tags', list())
        self.add_tags(*tags)

        # categories table
        categories = kwargs.pop('categories', dict())
        self.add_categories(**categories)

    @FileSerial._read
    @FileSerial._pull
    def get_version(self):
        """Get Treant version.

        :Returns:
            *version*
                version of Treant

        """
        return self._record['version']

    # TODO: need a proper schema update mechanism
    @FileSerial._write
    @FileSerial._pull_push
    def update_schema(self):
        """Update schema of file.

        :Returns:
            *version*
                version number of file's new schema
        """
        try:
            version = self._record['version']
        except KeyError:
            version = datreant.__version__

        return version

    @FileSerial._write
    @FileSerial._pull_push
    def update_version(self, version):
        """Update version of Treant.

        :Arugments:
            *version*
                new version of Treant
        """
        self._record['version'] = version

    @FileSerial._read
    @FileSerial._pull
    def get_tags(self):
        """Get all tags as a list.

        :Returns:
            *tags*
                list of all tags
        """
        return self._record['tags']

    @FileSerial._write
    @FileSerial._pull_push
    def add_tags(self, *tags):
        """Add any number of tags to the Treant.

        Tags are individual strings that serve to differentiate Treants from
        one another. Sometimes preferable to categories.

        :Arguments:
            *tags*
                tags to add; must be single numbers, strings, or boolean
                values; tags that are not these types are not added

        """
        # ensure tags are unique (we don't care about order)
        # also they must be of a certain set of types
        tags = set([tag for tag in tags
                    if (isinstance(tag, (int, float, string_types, bool)) or
                        tag is None)])

        # remove tags already present in metadata from list
        tags = tags.difference(set(self._record['tags']))

        # add new tags
        self._record['tags'].extend(tags)

    @FileSerial._write
    @FileSerial._pull_push
    def del_tags(self, *tags, **kwargs):
        """Delete tags from Treant.

        Any number of tags can be given as arguments, and these will be
        deleted.

        :Arguments:
            *tags*
                Tags to delete.

        :Keywords:
            *all*
                When True, delete all tags [``False``]

        """
        purge = kwargs.pop('all', False)

        if purge:
            self._record['tags'] = list()
        else:
            # remove redundant tags from given list if present
            tags = set([str(tag) for tag in tags])
            for tag in tags:
                # remove tag; if not present, continue anyway
                try:
                    self._record['tags'].remove(tag)
                except ValueError:
                    pass

    @FileSerial._read
    @FileSerial._pull
    def get_categories(self):
        """Get all categories as a dictionary.

        :Returns:
            *categories*
                dictionary of all categories
        """
        return self._record['categories']

    @FileSerial._write
    @FileSerial._pull_push
    def add_categories(self, **categories):
        """Add any number of categories to the Treant.

        Categories are key-value pairs of strings that serve to differentiate
        Treants from one another. Sometimes preferable to tags.

        If a given category already exists (same key), the value given will
        replace the value for that category.

        :Keywords:
            *categories*
                categories to add; keyword used as key, value used as value;
                values must be single numbers, strings, or boolean values;
                values that are not these types are not added

        """
        for key, value in categories.items():
            if (isinstance(value, (int, float, string_types, bool)) or
                    value is None):
                self._record['categories'][key] = value

    @FileSerial._write
    @FileSerial._pull_push
    def del_categories(self, *categories, **kwargs):
        """Delete categories from Treant.

        Any number of categories (keys) can be given as arguments, and these
        keys (with their values) will be deleted.

        :Arguments:
            *categories*
                Categories to delete.

        :Keywords:
            *all*
                When True, delete all categories [``False``]

        """
        purge = kwargs.pop('all', False)

        if purge:
            self._record['categories'] = dict()
        else:
            for key in categories:
                # continue even if key not already present
                self._record['categories'].pop(key, None)


class GroupFile(TreantFile):
    """Main Group state file.

    This file contains all the information needed to store the state of a
    Group object. It includes accessors, setters, and modifiers for all
    elements of the data structure, as well as the data structure definition.

    """
    # add new paths to include them in member searches
    memberpaths = ['abs', 'rel']
    _fields = ['uuid', 'treanttype', 'abs', 'rel']

    def __init__(self, filename, logger=None, **kwargs):
        """Initialize Group state file.

        :Arguments:
           *filename*
              path to file
           *logger*
              logger to send warnings and errors to
           *categories*
              user-given dictionary with custom keys and values; used to
              give distinguishing characteristics to object for search
           *tags*
              user-given list with custom elements; used to give distinguishing
              characteristics to object for search
        """
        super(GroupFile, self).__init__(filename, logger=logger, **kwargs)

    def _init_record(self):
        super(GroupFile, self)._init_record()
        self._record['members'] = list()

    @FileSerial._write
    @FileSerial._pull_push
    def add_member(self, uuid, treanttype, basedir):
        """Add a member to the Group.

        If the member is already present, its basedir paths will be updated
        with the given basedir.

        :Arguments:
            *uuid*
                the uuid of the new member
            *treanttype*
                the treant type of the new member
            *basedir*
                basedir of the new member in the filesystem

        """
        # check if uuid already present
        uuids = [member[0] for member in self._record['members']]

        if uuid not in uuids:
            self._record['members'].append([uuid,
                                            treanttype,
                                            os.path.abspath(basedir),
                                            os.path.relpath(
                                                basedir, self.get_location())])

    @FileSerial._write
    @FileSerial._pull_push
    def del_member(self, *uuid, **kwargs):
        """Remove a member from the Group.

        :Arguments:
            *uuid*
                the uuid(s) of the member(s) to remove

        :Keywords:
            *all*
                When True, remove all members [``False``]

        """
        purge = kwargs.pop('all', False)

        if purge:
            self._record['members'] = list()
        else:
            # remove redundant uuids from given list if present
            uuids = set([str(uid) for uid in uuid])

            # get matching rows
            # TODO: possibly faster to use table.where
            memberlist = list()
            for i, member in enumerate(self._record['members']):
                for uuid in uuids:
                    if (member[0] == uuid):
                        memberlist.append(i)

            memberlist.sort()
            j = 0
            # delete matching entries; have to use j to shift the register as
            # we remove entries
            for i in memberlist:
                self._record['members'].pop(i - j)
                j = j + 1

    @FileSerial._read
    @FileSerial._pull
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
        memberinfo = None
        for member in self._record['members']:
            if member[0] == uuid:
                memberinfo = member

        if memberinfo:
            memberinfo = {x: y for x, y in zip(self._fields, memberinfo)}

        return memberinfo

    @FileSerial._read
    @FileSerial._pull
    def get_members(self):
        """Get full member table.

        Sometimes it is useful to read the whole member table in one go instead
        of doing multiple reads.

        :Returns:
            *memberdata*
                dict giving full member data, with fields as keys and in member
                order
        """
        out = {key: [] for key in self._fields}

        for member in self._record['members']:
            for i, key in enumerate(self._fields):
                out[key].append(member[i])

        return out

    @FileSerial._read
    @FileSerial._pull
    def get_members_uuid(self):
        """List uuid for each member.

        :Returns:
            *uuids*
                list giving treanttype of each member, in order
        """
        return [member[0] for member in self._record['members']]

    @FileSerial._read
    @FileSerial._pull
    def get_members_treanttype(self):
        """List treanttype for each member.

        :Returns:
            *treanttypes*
                list giving treanttype of each member, in order
        """
        return [member[1] for member in self._record['members']]

    @FileSerial._read
    @FileSerial._pull
    def get_members_basedir(self):
        """List basedir for each member.

        :Returns:
            *basedirs*
                list containing all paths to member basedirs, in member order
        """
        return [member[2:] for member in self._record['members']]


class SimFile(TreantFile):
    filepaths = ['abs', 'rel']

    def _init_record(self):
        super(SimFile, self)._init_record()
        self._record['mds'] = dict()
        self._record['mds']['universes'] = dict()
        self._record['mds']['default'] = None

    @FileSerial._read
    @FileSerial._pull
    def get_mds_version(self):
        """Get Sim mdsynthesis version.

        :Returns:
            *version*
                mdsynthesis version of Sim

        """
        return self._record['mds']['version']

    # TODO: need a proper schema update mechanism
    @FileSerial._write
    @FileSerial._pull_push
    def update_mds_schema(self):
        """Update mdsynthesis-specific schema of file.

        :Returns:
            *version*
                version number of file's new schema
        """
        try:
            version = self._record['version']
        except KeyError:
            version = mdsynthesis.__version__

        return version

    @FileSerial._write
    @FileSerial._pull_push
    def update_mds_version(self, version):
        """Update mdsynthesis version of Sim.

        :Arugments:
            *version*
                new mdsynthesis version of Treant
        """
        self._record['mds']['version'] = version

    @FileSerial._write
    @FileSerial._pull_push
    def update_default(self, universe=None):
        """Mark the given universe as the default.

        :Arguments:
            *universe*
                name of universe to mark as default; if ``None``, remove
                default preference
        """
        self._record['mds']['default'] = universe

    @FileSerial._read
    @FileSerial._pull
    def get_default(self):
        """Get default universe.

        :Returns:
            *default*
                name of default universe; if no default universe, returns
                ``None``

        """
        return self._record['mds']['default']

    @FileSerial._read
    @FileSerial._pull
    def list_universes(self):
        """List universe names.

        :Returns:
            *universes*
                list giving names of all defined universes

        """
        return self._record['mds']['universes'].keys()

    @FileSerial._read
    @FileSerial._pull
    def get_universe(self, universe):
        """Get topology and trajectory paths for the desired universe.

        Returns multiple path types, including absolute paths (abspath)
        and paths relative to the Sim object (relCont).

        :Arguments:
            *universe*
                given name for selecting the universe

        :Returns:
            *topology*
                dictionary containing all paths to topology
            *trajectory*
                dictionary containing all paths to trajectories

        """
        top = self._record['mds']['universes'][universe]['top']
        outtop = {key: value for key, value in zip(self.filepaths, top)}

        trajs = self._record['mds']['universes'][universe]['traj']
        outtraj = {key: [] for key in self.filepaths}

        for traj in trajs:
            for i, key in enumerate(self.filepaths):
                outtraj[key].append(traj[i])

        return outtop, outtraj

    @FileSerial._write
    @FileSerial._pull_push
    def add_universe(self, universe, topology, *trajectory):
        """Add a universe definition to the Sim object.

        A Universe is an MDAnalysis object that gives access to the details
        of a simulation trajectory. A Sim object can contain multiple universe
        definitions (topology and trajectory pairs), since it is often
        convenient to have different post-processed versions of the same
        raw trajectory.

        :Arguments:
            *universe*
                given name for selecting the universe
            *topology*
                path to the topology file
            *trajectory*
                path to the trajectory file; multiple files may be given
                and these will be used in order as frames for the trajectory

        """
        # if universe schema already exists, don't overwrite it
        if universe not in self._record['mds']['universes']:
            self._record['mds']['universes'][universe] = dict()

        udict = self._record['mds']['universes'][universe]

        # add topology paths
        udict['top'] = [os.path.abspath(topology),
                        os.path.relpath(topology, self.get_location())]

        # add trajectory paths
        udict['traj'] = list()
        for segment in trajectory:
            udict['traj'].append(
                    [os.path.abspath(segment),
                     os.path.relpath(segment, self.get_location())])

        # add selections schema
        if 'sels' not in udict:
            udict['sels'] = dict()

        # add resnums schema
        if 'resnums' not in udict:
            udict['resnums'] = None

        # if no default universe, make this default
        if not self._record['mds']['default']:
            self._record['mds']['default'] = universe

    @FileSerial._write
    @FileSerial._pull_push
    def del_universe(self, universe):
        """Delete a universe definition.

        Deletes any selections associated with the universe.

        :Arguments:
            *universe*
                name of universe to delete
        """
        del self._record['mds']['universes'][universe]

    @FileSerial._write
    @FileSerial._pull_push
    def rename_universe(self, universe, newname):
        """Rename a universe definition.

        :Arguments:
            *universe*
                name of universe to rename
            *newname*
                new name of universe
        """
        if newname in self._record['mds']['universes']:
            raise ValueError(
                    "Universe '{}' already exixts;".format(newname) +
                    " remove it first")
        udicts = self._record['mds']['universes']
        udicts[newname] = udicts.pop(universe)

    @FileSerial._read
    @FileSerial._pull
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
        return self._record['mds']['universes'][universe]['sels'].keys()

    @FileSerial._read
    @FileSerial._pull
    def get_selection(self, universe, handle):
        """Get a stored atom selection for the given universe.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name to use for the selection

        :Returns:
            *selection*
                list of the selection strings making up the atom selection; may
                also be a list of atom indices

        """
        selections = self._record['mds']['universes'][universe]['sels']

        return selections[handle]

    @FileSerial._write
    @FileSerial._pull_push
    def add_selection(self, universe, handle, *selection):
        """Add an atom selection definition for the named Universe definition.

        AtomGroups are needed to obtain useful information from raw coordinate
        data. It is useful to store AtomGroup selections for later use, since
        they can be complex and atom order may matter.

        Will overwrite existing definition if it exists.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name to use for the selection
            *selection*
                selection string or numpy array of indices; multiple selections
                may be given and their order will be preserved, which is
                useful for e.g. structural alignments

        """
        outsel = list()
        for sel in selection:
            if isinstance(sel, np.ndarray):
                outsel.append(sel.tolist())
            elif isinstance(sel, string_types):
                outsel.append(sel)

        self._record['mds']['universes'][universe]['sels'][handle] = outsel

    @FileSerial._write
    @FileSerial._pull_push
    def del_selection(self, universe, handle):
        """Delete an atom selection from the specified universe.

        :Arguments:
            *universe*
                name of universe the selection applies to
            *handle*
                name of the selection

        """
        del self._record['mds']['universes'][universe]['sels'][handle]

    @FileSerial._write
    @FileSerial._pull_push
    def update_resnums(self, universe, resnums):
        """Update resnum definition for the given universe.

        Resnums are useful for referring to residues by their canonical resid,
        for instance that stored in the PDB. By giving a resnum definition
        for the universe, this definition can be applied to the universe
        on activation.

        Will overwrite existing definition if it exists.

        :Arguments:
            *universe*
                name of universe to associate resnums with
            *resnums*
                list giving the resnum for each atom in the topology, in index
                order
        """
        try:
            udict = self._record['mds']['universes'][universe]
        except KeyError:
            self.logger.info(
                "Universe definition '{}'".format(universe) +
                " does not exist. Add it first.")
            return

        # add resnums
        udict['resnums'] = resnums

    @FileSerial._read
    @FileSerial._pull
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
        return self._record['mds']['universes'][universe]['resnums']

    @FileSerial._write
    @FileSerial._pull_push
    def del_resnums(self, universe):
        """Delete resnum definition from specified universe.

        :Arguments:
            *universe*
                name of universe to remove resnum definition from
        """
        self._record['mds']['universes'][universe]['resnums'] = None
