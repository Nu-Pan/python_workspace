
python "%~dp0multiband_tool.py" "%~1"
@if errorlevel 1 (
    @pause
)
