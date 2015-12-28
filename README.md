# Script for pytables to JSON conversion of datreant/mdsynthesis state files

This repo gives a single script that converts the deprecated pytables HDF5
state files used in [`datreant`](https://github.com/datreant/datreant) and
[`mdsynthesis`](https://github.com/datreant/MDSynthesis) packages. This script should
be used on files of the form:

    Treant.<uuid>.h5

where `Treant` may be any of [`Treant`, `Group`, `Sim`], and <uuid> is a string
of characters. It will create a file with the same name but a different extension
in the same directory:

    Treant.<uuid>.json

This file will work with the latest versions of `datreant`/`mdsynthesis` as the
state file for that `Treant`/`Sim`. The old `.h5` file can be left if you're
paranoid, but otherwise it is no longer used by the library. 

