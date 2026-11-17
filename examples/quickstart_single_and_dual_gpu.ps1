<#
.SYNOPSIS
    Common ways to probe GPUs and start ShadowZero on one or two cards.

.EXAMPLE
    .\examples\quickstart_single_and_dual_gpu.ps1 -Mode probe
    .\examples\quickstart_single_and_dual_gpu.ps1 -Mode pair -Model "J:\Models\Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-mixD18"
#>
param(
    [ValidateSet("probe", "single", "pair", "exact")]
    [string]$Mode = "probe",
    [string]$Model = "J:\Models\Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-mixD18",
    [int]$Port = 8081,
    [int]$Ctx = 24576
)

$ShadowZero = Join-Path $PSScriptRoot "..\target\release\shadowzero.exe"
if (-not (Test-Path $ShadowZero)) {
    Write-Host "Release binary not found at $ShadowZero — building..."
    cargo build --release -p shadowzero
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Mode) {
    "probe" {
        # Lists detected CUDA devices, BAR1 sizes, P2P capability, host<->GPU bandwidth, and hop latency
        & $ShadowZero gpus
    }
    "single" {
        # Fast defaults (--tail-skip 3, --tensor-prompt, --verify-rank 2, DFlash 2, pinned RAM tier) on GPU 0
        & $ShadowZero serve $Model --gpus 0 --port $Port --ctx $Ctx --power-limit 5070=200
    }
    "pair" {
        # Splits layers across two cards (auto puts the larger-VRAM card first as Stage 0) and opens http://127.0.0.1:$Port
        & $ShadowZero serve $Model --gpus 0,1 --port $Port --ctx $Ctx --power-limit 5070=200,5060=150
    }
    "exact" {
        # Switches speculation and fast approximations off (bit-exact match against the scalar reference)
        & $ShadowZero serve $Model --gpus 0,1 --port $Port --ctx $Ctx --exact --power-limit 5070=200,5060=150
    }
}
