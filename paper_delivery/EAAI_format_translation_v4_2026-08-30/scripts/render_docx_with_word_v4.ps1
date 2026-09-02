param(
    [string]$DeliveryRoot = "."
)

$ErrorActionPreference = "Stop"

# Word COM fallback when LibreOffice is unavailable.
$deliverables = Join-Path $DeliveryRoot "deliverables"
$renderRoot = Join-Path $DeliveryRoot "qa\docx_render"
$names = @(
    "EAAI_manuscript_anonymized_v4",
    "EAAI_manuscript_two_column_reading_proof_v4",
    "EAAI_title_page_v4",
    "EAAI_supplementary_material_v4",
    "EAAI_highlights_v4",
    "EAAI_cover_letter_v4",
    "EAAI_manuscript_zh_v4",
    "EAAI_title_page_zh_v4",
    "EAAI_supplementary_material_zh_v4",
    "EAAI_highlights_zh_v4",
    "EAAI_cover_letter_zh_v4"
)
$previewRoot = Join-Path $deliverables "rendered_previews"
New-Item -ItemType Directory -Path $previewRoot -Force | Out-Null

$word = $null
foreach ($name in $names) {
    try {
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        $word.DisplayAlerts = 0
        $docx = Join-Path $deliverables ($name + ".docx")
        $outDir = Join-Path $renderRoot $name
        New-Item -ItemType Directory -Path $outDir -Force | Out-Null
        $pdf = Join-Path $outDir ($name + ".pdf")
        $doc = $word.Documents.Open($docx, $false, $true)
        try {
            # 17 = wdExportFormatPDF
            $doc.ExportAsFixedFormat($pdf, 17)
        }
        finally {
            $doc.Close($false)
        }
        Copy-Item -LiteralPath $pdf -Destination (Join-Path $previewRoot ($name + ".pdf")) -Force
        Write-Output "$name`t$pdf"
    }
    finally {
        if ($null -ne $word) {
            try { $word.Quit() } catch { }
            try { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null } catch { }
            $word = $null
        }
        [GC]::Collect()
        [GC]::WaitForPendingFinalizers()
    }
}
