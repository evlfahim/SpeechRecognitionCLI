from cx_Freeze import setup, Executable

setup(
    name="whisper_cli",
    version="1.0",
    description="whisper_cli",
    executables=[Executable("final.py")]
)
