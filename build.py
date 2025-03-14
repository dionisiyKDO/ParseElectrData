import os
import PyInstaller.__main__

# Build preferences
script_name = "gui.py"
output_name = "ParseData"
hidden_imports = ["pandas"]  # Add any additional hidden imports if needed
icon_path = "folder_icon.ico"  # Optional: Path to an .ico file for the EXE icon

# Construct PyInstaller command
command = [
    script_name,
    "--onefile",          # Single EXE file
    "--noconsole",       # No console for GUI app
    "--name", output_name  # Custom EXE name
]

# Add hidden imports
for module in hidden_imports:
    command.extend(["--hidden-import", module])

# Add icon if specified
if icon_path and os.path.exists(icon_path):
    command.extend(["--icon", icon_path])

# Run PyInstaller
PyInstaller.__main__.run(command)

print(f"Build complete! Check the 'dist' folder for {output_name}.exe")