"""Dataset download and listing endpoints."""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/datasets/download-deap")
async def download_deap():
    """Download DEAP dataset from Kaggle (requires ~/.kaggle/kaggle.json)."""
    from training.data_loaders import download_deap_kaggle
    try:
        path = download_deap_kaggle()
        dat_count = len(list(path.glob("s*.dat")))
        return {"status": "success", "path": str(path), "files": dat_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets/download-dens")
async def download_dens():
    """Download/sync DENS dataset from OpenNeuro."""
    from training.data_loaders import download_dens_openneuro
    try:
        path = download_dens_openneuro()
        subjects = len(list(path.glob("sub-*")))
        return {"status": "success", "path": str(path), "subjects": subjects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets")
async def list_datasets():
    """List all available EEG datasets and their download status."""
    try:
        from training.data_loaders import list_available_datasets
        return list_available_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
