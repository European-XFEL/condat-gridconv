python_sources = [
  '__init__.py',
  'gridconv.py',
]

py.install_sources(
  python_sources,
  subdir: 'condat_gridconv',  # Folder relative to site-packages to install to
)

py.extension_module(
  'shift',
  'shift.pyx',
  install: true,
  subdir: 'condat_gridconv',  # Folder relative to site-packages to install to
)
