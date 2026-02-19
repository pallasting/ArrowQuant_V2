# Set Model Path Environment Variable
# Usage: .\scripts\set_model_path.ps1

$ModelPath = "D:\ai-models\minilm"

# Set for current session
$env:ARROW_MODEL_PATH = $ModelPath
Write-Host "✓ Environment variable set for current session:"
Write-Host "  ARROW_MODEL_PATH=$ModelPath"
Write-Host ""
Write-Host "To set permanently (requires admin), run:"
Write-Host "  [System.Environment]::SetEnvironmentVariable('ARROW_MODEL_PATH', '$ModelPath', 'User')"
