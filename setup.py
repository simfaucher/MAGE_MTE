#!/usr/bin/python
# -*- coding: utf-8 -*-
# Python 3
 
"""
Icone sous Windows: il faut:
=> un xxx.ico pour integration dans le exe, avec "icon=xxx.ico"
=> un xxx.png pour integration avec PyQt4 + demander la recopie avec includefiles.
"""
 
import sys, os
from cx_Freeze import setup, Executable
 
#############################################################################
# preparation des options
 
# chemins de recherche des modules
# ajouter d'autres chemins (absolus) si necessaire: sys.path + ["chemin1", "chemin2"]
path = sys.path
 
# options d'inclusion/exclusion des modules
includes = ["matplotlib.backends.backend_tkagg", "scipy.sparse.csgraph._validation", \
    "scipy.spatial.ckdtree"]  # nommer les modules non trouves par cx_freeze
excludes = []
packages = ["idna", "os", "sys", "operator", "glob", "math", \
    "json", "datetime", "csv", "time", "statistics", "tkinter", \
    "numpy", "matplotlib.pyplot", "cv2", "pykson", "multiprocessing.pool", \
    "sklearn", "sklearn.neighbors", "zmq"]  # nommer les packages utilisés
 
# copier les fichiers non-Python et/ou repertoires et leur contenu:
includefiles = ["Domain", "ML", "imagezmq", "learning_settings.json", \
    "learning_settings_85.json", "learning_settings_64.json"]
 
if sys.platform == "win32":
    pass
    # includefiles += [...] : ajouter les recopies specifiques à Windows
elif sys.platform == "linux2":
    pass
    # includefiles += [...] : ajouter les recopies specifiques à Linux
else:
    pass
    # includefiles += [...] : cas du Mac OSX non traite ici
 
# pour que les bibliotheques binaires de /usr/lib soient recopiees aussi sous Linux
binpathincludes = []
if sys.platform == "linux2":
    binpathincludes += ["/usr/lib"]
 
# niveau d'optimisation pour la compilation en bytecodes
optimize = 0
 
# si True, n'affiche que les warning et les erreurs pendant le traitement cx_freeze
silent = True
 
# construction du dictionnaire des options
options = {"path": path,
           "includes": includes,
           "excludes": excludes,
           "packages": packages,
           "include_files": includefiles,
           "bin_path_includes": binpathincludes,
           "optimize": optimize,
           "silent": silent
           }
 
# pour inclure sous Windows les dll system de Windows necessaires
if sys.platform == "win32":
    options["include_msvcr"] = True
 
#############################################################################
# preparation des cibles
base = None
if sys.platform == "win32":
    # base = "Win32GUI"  # pour application graphique sous Windows
    base = "Console" # pour application en console sous Windows
 
icone = None
if sys.platform == "win32":
    icone = "logo.ico"
 
cible_1 = Executable(
    script="MTE.py",
    base=base,
    icon=icone
    )
 
#############################################################################
# creation du setup
setup(
    name="Motion Tracking Engine",
    version="2.4.0",
    description="Altran Motion Tracking Engine",
    author="Altran",
    options={"build_exe": options},
    executables=[cible_1,]
    )


for filename in os.listdir ("build"):
    if os.path.exists(os.path.join("build", filename, "lib", "multiprocessing", "Pool.pyc")):
        os.rename(os.path.join("build", filename, "lib", "multiprocessing", "Pool.pyc"),
                  os.path.join("build", filename, "lib", "multiprocessing", "pool.pyc"))
    if os.path.exists(os.path.join("build", filename, "lib", "scipy", "spatial", "cKDTree.cp37-win_amd64.pyd")):
        os.rename(os.path.join("build", filename, "lib", "scipy", "spatial", "cKDTree.cp37-win_amd64.pyd"),
                  os.path.join("build", filename, "lib", "scipy", "spatial", "ckdtree.cp37-win_amd64.pyd"))
