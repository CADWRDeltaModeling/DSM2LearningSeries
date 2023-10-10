# tdbcConfig.sh --
#
# This shell script (for sh) is generated automatically by TDBC's configure
# script. It will create shell variables for most of the configuration options
# discovered by the configure script. This script is intended to be included
# by the configure scripts for TDBC extensions so that they don't have to
# figure this all out for themselves.
#
# The information in this file is specific to a single platform.
#
# RCS: @(#) $Id$

# TDBC's version number
tdbc_VERSION=1.0.6
TDBC_VERSION=1.0.6

# Name of the TDBC library - may be either a static or shared library
tdbc_LIB_FILE=tdbc106t.dll
TDBC_LIB_FILE=tdbc106t.dll

# String to pass to the linker to pick up the TDBC library from its build dir
tdbc_BUILD_LIB_SPEC="@tdbc_BUILD_LIB_SPEC@"
TDBC_BUILD_LIB_SPEC="@tdbc_BUILD_LIB_SPEC@"

# String to pass to the linker to pick up the TDBC library from its installed
# dir.
tdbc_LIB_SPEC="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbc106t.dll"
TDBC_LIB_SPEC="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbc106t.dll"

# Name of the TBDC stub library
tdbc_STUB_LIB_FILE="tdbcstub106.lib"
TDBC_STUB_LIB_FILE="tdbcstub106.lib"

# String to pass to the linker to pick up the TDBC stub library from its
# build directory
tdbc_BUILD_STUB_LIB_SPEC="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\Release_AMD64_VC1900\tdbcstub106.lib"
TDBC_BUILD_STUB_LIB_SPEC="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\Release_AMD64_VC1900\tdbcstub106.lib"

# String to pass to the linker to pick up the TDBC stub library from its
# installed directory
tdbc_STUB_LIB_SPEC="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbcstub106.lib"
TDBC_STUB_LIB_SPEC="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbcstub106.lib"

# Path name of the TDBC stub library in its build directory
tdbc_BUILD_STUB_LIB_PATH="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\Release_AMD64_VC1900\tdbcstub106.lib"
TDBC_BUILD_STUB_LIB_PATH="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\Release_AMD64_VC1900\tdbcstub106.lib"

# Path name of the TDBC stub library in its installed directory
tdbc_STUB_LIB_PATH="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbcstub106.lib"
TDBC_STUB_LIB_PATH="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6\tdbcstub106.lib"

# Location of the top-level source directories from which TDBC was built.
# This is the directory that contains doc/, generic/ and so on.  If TDBC
# was compiled in a directory other than the source directory, this still
# points to the location of the sources, not the location where TDBC was
# compiled.
tdbc_SRC_DIR="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\.."
TDBC_SRC_DIR="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\.."

# String to pass to the compiler so that an extension can find installed TDBC
# headers
tdbc_INCLUDE_SPEC="-IC:\bld\tk_1537274647171\work\tcl8.6.8\win\..\include"
TDBC_INCLUDE_SPEC="-IC:\bld\tk_1537274647171\work\tcl8.6.8\win\..\include"

# String to pass to the compiler so that an extension can find TDBC headers
# in the source directory
tdbc_BUILD_INCLUDE_SPEC="-IC:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\..\generic"
TDBC_BUILD_INCLUDE_SPEC="-IC:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\..\generic"

# Path name where .tcl files in the tdbc package appear at run time.
tdbc_LIBRARY_PATH="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6"
TDBC_LIBRARY_PATH="Z:/test/notebook_dsm2plan/pydelmod_plan\Library\lib\tdbc1.0.6"

# Path name where .tcl files in the tdbc package appear at build time.
tdbc_BUILD_LIBRARY_PATH="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\../library"
TDBC_BUILD_LIBRARY_PATH="C:\bld\tk_1537274647171\work\tcl8.6.8\pkgs\tdbc1.0.6\win\../library"

# Additional flags that must be passed to the C compiler to use tdbc
tdbc_CFLAGS=
TDBC_CFLAGS=

