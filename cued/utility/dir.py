"""
Utility functions needed by functions/methods in the package
"""
import os
import shutil as sh

from cued import CUEDPATH

def cued_copy(cued_dir, dest_dir):
	'''
	Copies files relative to the cued source path
	'''
	sh.copy(CUEDPATH + '/' + cued_dir, dest_dir)


def cued_pwd(cued_file):
	'''
	Returns the full path of a file relative to he cued source path
	'''
	return CUEDPATH + '/' + cued_file


def mkdir(dirname):
	'''
	Only try to create directory when directory does not exist
	'''
	if not os.path.exists(dirname):
		os.mkdir(dirname)

def chdir(dirname='..'):
	'''
	Defaults to go back one folder
	'''
	os.chdir(dirname)


def mkdir_chdir(dirname):
	'''
	Create directory and move into it
	'''
	mkdir(dirname)
	chdir(dirname)


def rmdir_mkdir_chdir(dirname):
	'''
	If the directory exists remove it first before creating
	a new one and changing into it.
	'''
	if os.path.exists(dirname) and os.path.isdir(dirname):
		sh.rmtree(dirname)

	mkdir_chdir(dirname)
