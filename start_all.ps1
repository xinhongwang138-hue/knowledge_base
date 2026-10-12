# ============================================================
#  掌柜智库 (ZhangGui ZhiKu) - One-Click Startup
#  一键启动：Docker 依赖(MongoDB + Milvus 三件套 + Attu) + 两个 FastAPI 服务
#
#  启动后访问：
#    - 问答聊天(Query)  : http://127.0.0.1:8011/chat.html
#    - 上传文档(Import) : http://127.0.0.1:8010/import.html
#    - Milvus 控制台    : http://127.0.0.1:8000  (Attu)
#    - MinIO 控制台     : http://127.0.0.1:9001  (minioadmin/minioadmin)
#
#  使用：在项目根目录执行  ./start_all.ps1
#  或双击同目录 start_all.bat
# ============================================================
$ErrorActionPreference = "Continue"

$PROJECT   = "D:\PythonProject\knowledge_base"
$PY        = Join-Path $PROJECT ".venv\Scripts\python.exe"
$COMPOSE   = Join-Path $PROJECT "docker-compose.yml"
$LOG_DIR   = Join-Path $PROJECT "logs"
$MONGO_CNT = "mongo"                     # MongoDB 容器名（docker ps 中实际名称）
$MILVUS_CNTS = @("milvus-etcd", "milvus-minio", "milvus-standalone", "attu")

if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }

# ---------- 端口探测 ----------
function Test-Port([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try   { $c.Connect("127.0.0.1", $Port); $c.Close(); return $true }
    catch { return $false }
}

function Wait-Port([int]$Port, [int]$TimeoutSec = 120, [string]$Name = "") {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $Port) {
            Write-Host "    [OK] $Name ($Port) ready" -ForegroundColor Green
            return $true
        }
        Start-Sleep -Seconds 3
    }
    Write-Host "    [WARN] $Name ($Port) not ready in ${TimeoutSec}s" -ForegroundColor Yellow
    return $false
}

function Start-Backend([string]$ModuleArgs, [int]$Port, [string]$Name, [string]$LogPrefix) {
    if (Test-Port $Port) {
        Write-Host "    [SKIP] $Name already running on $Port" -ForegroundColor Green
        return
    }
    $outLog = Join-Path $LOG_DIR "${LogPrefix}_stdout.log"
    $errLog = Join-Path $LOG_DIR "${LogPrefix}_stderr.log"
    $p = Start-Process -FilePath $PY -ArgumentList $ModuleArgs -WorkingDirectory $PROJECT `
         -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
    Write-Host "    [..] $Name starting, PID $($p.Id), logs: logs\${LogPrefix}_*.log" -ForegroundColor Cyan
    if (-not (Wait-Port $Port 240 $Name)) {
        Write-Host "    [ERROR] $Name failed to start, check logs\${LogPrefix}_stderr.log" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  ZhangGui ZhiKu - One-Click Startup" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# ========== [1/6] Docker Desktop 检测与启动 ==========
Write-Host "[1/6] Checking Docker daemon..." -ForegroundColor Cyan
$dockerOK = $false
try { docker info *> $null; if ($LASTEXITCODE -eq 0) { $dockerOK = $true } } catch { }
if (-not $dockerOK) {
    Write-Host "    Docker daemon is NOT running, trying to start Docker Desktop..." -ForegroundColor Yellow
    $dd = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dd) { Start-Process $dd }
    $deadline = (Get-Date).AddSeconds(120)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 3
        try { docker info *> $null; if ($LASTEXITCODE -eq 0) { $dockerOK = $true; break } } catch { }
    }
    if (-not $dockerOK) {
        Write-Host "    [ERROR] Docker Desktop did not become ready." -ForegroundColor Red
        Write-Host "    Please start Docker Desktop manually, then re-run this script." -ForegroundColor Red
        exit 1
    }
}
Write-Host "    [OK] Docker daemon ready" -ForegroundColor Green

# ========== [2/6] 启动 Milvus 三件套 + Attu（docker compose） ==========
Write-Host "[2/6] Starting Milvus stack via docker compose..." -ForegroundColor Cyan
Push-Location $PROJECT
docker compose -f $COMPOSE up -d etcd minio standalone attu
$composeCode = $LASTEXITCODE
Pop-Location
if ($composeCode -ne 0) {
    Write-Host "    [ERROR] docker compose up failed (exit $composeCode)" -ForegroundColor Red
    exit 1
}
foreach ($n in $MILVUS_CNTS) {
    $state = (docker inspect -f "{{.State.Running}}" $n 2>$null)
    if ($state -eq "true") { Write-Host "    [OK] $n running" -ForegroundColor Green }
    else { Write-Host "    [WARN] $n not running, check docker logs" -ForegroundColor Yellow }
}

# ========== [3/6] 启动 MongoDB 容器 ==========
Write-Host "[3/6] Starting MongoDB container..." -ForegroundColor Cyan
docker start $MONGO_CNT 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    # 容器名可能不同，尝试从现有容器中找名字含 mongo 的
    $found = @(docker ps -a --format "{{.Names}}" 2>$null | Where-Object { $_ -match "mongo" })
    if ($found.Count -gt 0) {
        foreach ($n in $found) {
            Write-Host "    [..] starting existing mongo-like container: $n" -ForegroundColor Yellow
            docker start $n | Out-Null
        }
    } else {
        Write-Host "    [WARN] No container named '$MONGO_CNT' found." -ForegroundColor Yellow
        Write-Host "    If MongoDB runs on host/another container, skip; otherwise create it first." -ForegroundColor Yellow
    }
}
Wait-Port 27017 90 "MongoDB(127.0.0.1:27017)"

# ========== [4/6] 等待 Milvus / MinIO / Attu 端口就绪 ==========
Write-Host "[4/6] Waiting for dependency ports..." -ForegroundColor Cyan
Wait-Port 19530 180 "Milvus(19530)"
Wait-Port 9000  90  "MinIO API(9000)"
Wait-Port 8000  90  "Attu(8000)"

# ========== [5/6] 启动 Import 服务 (8010) ==========
Write-Host "[5/6] Starting Import service (8010)..." -ForegroundColor Cyan
Start-Backend -ModuleArgs @('main.py', '--port', '8010') -Port 8010 -Name "Import(8010)" -LogPrefix "start_import"

# ========== [6/6] 启动 Query 服务 (8011) ==========
Write-Host "[6/6] Starting Query service (8011)..." -ForegroundColor Cyan
Start-Backend -ModuleArgs @('-m', 'app.query_process.api.query_service') -Port 8011 -Name "Query(8011)" -LogPrefix "start_query"

# ========== 健康检查 ==========
Write-Host ""
Write-Host "== Health check ==" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8011/health" -TimeoutSec 8
    $body = $r.Content
    Write-Host ("    Query /health  -> {0}  {1}" -f $r.StatusCode, $body.Substring(0, [Math]::Min(120, $body.Length))) -ForegroundColor Green
} catch {
    Write-Host "    Query /health  -> FAILED ($($_.Exception.Message))" -ForegroundColor Red
}
try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8010/import.html" -TimeoutSec 8
    Write-Host ("    Import /import.html -> {0}" -f $r.StatusCode) -ForegroundColor Green
} catch {
    Write-Host "    Import /import.html -> FAILED ($($_.Exception.Message))" -ForegroundColor Red
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  All done. Quick links:" -ForegroundColor Cyan
Write-Host "    问答聊天  : http://127.0.0.1:8011/chat.html" -ForegroundColor White
Write-Host "    上传文档  : http://127.0.0.1:8010/import.html" -ForegroundColor White
Write-Host "    Milvus 管理(Attu) : http://127.0.0.1:8000" -ForegroundColor White
Write-Host "    MinIO 控制台      : http://127.0.0.1:9001 (minioadmin/minioadmin)" -ForegroundColor White
Write-Host "  Service logs: logs\start_import_*.log, logs\start_query_*.log" -ForegroundColor White
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
