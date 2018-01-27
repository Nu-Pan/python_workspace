
python "%~dp0correct_bass.py" "%~1"
@if errorlevel 1 (
    @pause
)
