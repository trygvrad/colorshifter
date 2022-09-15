python3 -m PyInstaller ^
--exclude-module matplotlib --exclude-module scipy.fft --exclude-module scipy.integrate --exclude-module scipy.optimize --exclude-module scipy.signal --exclude-module scipy.stats --exclude-module h5py ^
--hidden-import PySide2.QtUiTools --hidden-import PIL.Image --hidden-import scipy.interpolate --hidden-import scipy.ndimage --exclude-module tkinter --exclude-module pyqt5 --icon=iconOri.ico --log-level=DEBUG --onefile --windowed loader.py
%Pause 
