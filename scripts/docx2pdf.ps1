param(
    [Parameter(Mandatory = $true)][string]$Docx
)
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx, $false, $true)
    $pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")
    $doc.ExportAsFixedFormat($pdf, 17)
    $doc.Close($false)
    Write-Output "saved $pdf"
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
