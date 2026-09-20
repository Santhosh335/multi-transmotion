# run_subset_study.ps1
# Runs the Multi-Transmotion NBA evaluation pipeline at several test-set sizes
# and logs ADE/FDE to results.csv. Run from inside the multi-transmotion folder,
# e.g.:  C:\Users\santh\Documents\projects\multi-transmotion>  ..\run_subset_study.ps1
#
# Make sure your venv is active first: C:\venv\Scripts\activate

$ErrorActionPreference = "Continue"
# Note: kept as "Continue" rather than "Stop" - with "Stop", PowerShell treats ANY stderr output
# from python (even normal logging) as a terminating error when combined with 2>&1 redirection.
# Real failures are instead caught below via explicit $LASTEXITCODE checks and throw statements.

# --- Sizes to test, smallest first. Add larger ones later once you know 16GB RAM handles these. ---
$sizes = @(100, 300, 600, 1000, 1500, 3000, 6000)

$resultsFile = "subset_study_results.csv"
"clip_count,ade,fde,status" | Set-Content $resultsFile

foreach ($n in $sizes) {
    Write-Host "=================================================="
    Write-Host "Running with test size: $n clips"
    Write-Host "=================================================="

    try {
        # 1. Set the truncation size in extract_NBA.py (test branch only - uses set_test_size.py
        #    since train/test may currently read the same number, making plain find-replace unsafe)
        python set_test_size.py $n
        if ($LASTEXITCODE -ne 0) { throw "set_test_size.py failed" }

        # 2. Clean previous run's intermediate files
        Remove-Item extract_data\NBA\output_csv\nba_test.csv -ErrorAction SilentlyContinue
        Remove-Item UniHuMotion_trajnetpp\data\UniHuMotion_NBA\nba_test.csv -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force UniHuMotion_trajnetpp\output_pre -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force data\UniHuMotion\UHM_NBA -ErrorAction SilentlyContinue
        Remove-Item -Recurse -Force data\cache\NBA\test -ErrorAction SilentlyContinue

        # 3. Extract
        python extract_data\NBA\extract_NBA.py --split test
        if ($LASTEXITCODE -ne 0) { throw "extract_NBA.py failed" }

        # 4. Convert
        Copy-Item extract_data\NBA\output_csv\nba_test.csv UniHuMotion_trajnetpp\data\UniHuMotion_NBA\ -Force
        Push-Location UniHuMotion_trajnetpp
        python -m trajnetdataset.convert --acceptance 1.0 1.0 1.0 1.0 --train_fraction 0.0 --val_fraction 0.0 --fps 5 --obs_len 10 --pred_len 20 --chunk_stride 1
        if ($LASTEXITCODE -ne 0) { Pop-Location; throw "convert.py failed" }
        Pop-Location

        # 5. Move into place
        Move-Item UniHuMotion_trajnetpp\output_pre UniHuMotion_trajnetpp\UHM_NBA
        New-Item -ItemType Directory -Force -Path data\UniHuMotion | Out-Null
        Move-Item UniHuMotion_trajnetpp\UHM_NBA data\UniHuMotion\

        # 6. Generate cache
        python UniHuMotion_cache\cache_generator.py --UniHuMotion_dataset UHM_NBA
        if ($LASTEXITCODE -ne 0) { throw "cache_generator.py failed" }

        # 7. Evaluate and capture output
        $output = python ft_NBA\evaluate.py --ckpt checkpoints\FT_NBA_ckpt.pth.tar --split test 2>&1 | Out-String
        Write-Host $output

        if ($output -match "ADE:\s*tensor\(([\d\.]+)\)") { $ade = $matches[1] } else { $ade = "" }
        if ($output -match "FDE:\s*tensor\(([\d\.]+)\)") { $fde = $matches[1] } else { $fde = "" }

        "$n,$ade,$fde,ok" | Add-Content $resultsFile
        Write-Host "Size $n done: ADE=$ade FDE=$fde"
    }
    catch {
        Write-Host "FAILED at size $n : $_"
        "$n,,,FAILED: $_" | Add-Content $resultsFile
        Write-Host "Stopping here - smaller sizes already recorded are still valid results."
        break
    }
}

Write-Host "=================================================="
Write-Host "Done. Results saved to $resultsFile"
Write-Host "=================================================="
Get-Content $resultsFile
