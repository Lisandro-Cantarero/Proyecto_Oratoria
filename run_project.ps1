# Script para iniciar el entorno de Tesis rápidamente
Write-Host "Iniciando entorno de Tesis de Oratoria..." -ForegroundColor Cyan

# 1. Bypass de política de ejecución
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 2. Activar el entorno virtual
If (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
    Write-Host "✅ Entorno Virtual Activo" -ForegroundColor Green
} Else {
    Write-Host "❌ Error: No se encontró el entorno virtual .venv" -ForegroundColor Red
    Pause
    Exit
}

# 3. Verificar instalaciones críticas
python -c "import mediapipe; import librosa; print('✅ Librerías verificadas')" 

Write-Host "--- LISTO PARA TRABAJAR ---" -ForegroundColor Yellow
Write-Host "Puedes ejecutar: streamlit run app.py"