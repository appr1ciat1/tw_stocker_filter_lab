<#
trigger_report.ps1 — 由本機排程觸發每日報表（取代/備援 GitHub 內建排程）

為什麼需要：GitHub Actions 的 schedule 在高負載時會延遲數十分鐘、甚至整天不觸發，
官方明載為 best-effort。用本機排程主動 dispatch，可靠度高得多。

★關鍵：一律用 force=false。
  force=false → workflow 仍執行完整的交易日檢查與嚴格資料契約（推薦，日常排程用）
  force=true  → 略過交易日檢查、放寬資料新鮮度（僅限人工補跑，例如颱風後補資料）

用法：
  .\trigger_report.ps1                 # 正常每日觸發
  .\trigger_report.ps1 -Force          # 人工補跑（略過檢查）
  .\trigger_report.ps1 -SkipWeekend:$false   # 連週末也送（預設週末不送）

前置：安裝 GitHub CLI 並登入一次
  winget install GitHub.cli
  gh auth login
#>

param(
    [switch]$Force,
    [switch]$SkipWeekend = $true,
    [string]$Repo = "appr1ciat1/tw_stocker_filter_lab",
    [string]$Workflow = "update_ai_report.yml",
    [string]$LogFile = "$PSScriptRoot\trigger_report.log"
)

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Output $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

# 台灣時間（本機若非 UTC+8 也能正確換算）
$twNow = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId(
    [DateTime]::UtcNow, "Taipei Standard Time")
Write-Log "=== 觸發檢查（台灣時間 $($twNow.ToString('yyyy-MM-dd HH:mm')) ）==="

# 週末不送（省一次 Actions 額度；平日的國定假日交給 workflow 內的 XTAI 行事曆判斷）
if ($SkipWeekend -and -not $Force) {
    if ($twNow.DayOfWeek -eq 'Saturday' -or $twNow.DayOfWeek -eq 'Sunday') {
        Write-Log "週末（$($twNow.DayOfWeek)）→ 不觸發"
        exit 0
    }
}

# 檢查 gh 是否可用且已登入
$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
    Write-Log "❌ 找不到 gh（GitHub CLI）。請先 winget install GitHub.cli 並 gh auth login"
    exit 1
}
gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Log "❌ gh 未登入。請執行 gh auth login"
    exit 1
}

$forceVal = if ($Force) { "true" } else { "false" }
Write-Log "觸發 $Repo / $Workflow  (force=$forceVal)"

gh workflow run $Workflow --repo $Repo -f force=$forceVal 2>&1 | ForEach-Object { Write-Log "  $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Log "❌ 觸發失敗（exit $LASTEXITCODE）"
    exit 1
}

# 等幾秒後回報最新一次 run 的狀態，方便從 log 看出有沒有真的啟動
Start-Sleep -Seconds 12
$run = gh run list --repo $Repo --workflow $Workflow --limit 1 `
        --json databaseId,status,conclusion,createdAt,event 2>$null | ConvertFrom-Json
if ($run) {
    Write-Log ("✅ 已觸發 run #{0}  event={1}  status={2}  建立於 {3}" -f `
        $run[0].databaseId, $run[0].event, $run[0].status, $run[0].createdAt)
    Write-Log "   查看： gh run watch $($run[0].databaseId) --repo $Repo"
} else {
    Write-Log "⚠️ 已送出觸發，但暫時查不到 run（可能仍在排隊）"
}
exit 0
