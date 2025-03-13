import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {
    "packages": [
        "pandas", 
        "numpy", 
        "tkinter", 
        "plotly", 
        "matplotlib", 
        "seaborn",
        "os",
        "logging",
        "sv_ttk"
    ],
    "include_files": [
        # Include any necessary files, such as icons
        ("folder_icon.ico", "folder_icon.ico"),
        # Add any other data files needed
    ],
    "excludes": ["pytest", "unittest"],
}

# Base for Windows
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # Use this to hide console for Windows GUI apps

executables = [
    Executable(
        "gui.py",  # Your main script
        base=base,
        icon="folder_icon.ico",  # Application icon
        target_name="ElectricalDataAnalyzer.exe"  # Name of the executable
    )
]

setup(
    name="Electrical Data Analyzer",
    version="1.0",
    description="Application for analyzing electrical data",
    options={"build_exe": build_exe_options},
    executables=executables
)