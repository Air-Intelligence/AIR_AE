"""
Download MERRA-2 meteorological data (PBLH, U10M, V10M) using earthaccess
"""
import earthaccess
from datetime import datetime
from pathlib import Path
from typing import List
import config
import utils

logger = utils.setup_logging(__name__)


def setup_earthdata_auth():
    """Set up Earthdata authentication (same as TEMPO)"""
    logger.info("Setting up Earthdata authentication...")

    try:
        auth = earthaccess.login(strategy="netrc")
        logger.info("✓ Authenticated with Earthdata")
        return auth
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise


def search_merra2_granules(
    collection: str,
    date_start: datetime,
    date_end: datetime,
    bbox: dict = None
) -> List:
    """
    Search for MERRA-2 granules

    Args:
        collection: Collection short name (e.g., 'M2T1NXSLV')
        date_start: Start date
        date_end: End date
        bbox: Bounding box

    Returns:
        List of granule objects
    """
    logger.info(f"Searching for {collection} granules...")

    if bbox is None:
        bbox_tuple = (
            config.BBOX['west'],
            config.BBOX['south'],
            config.BBOX['east'],
            config.BBOX['north']
        )
    else:
        bbox_tuple = (bbox['west'], bbox['south'], bbox['east'], bbox['north'])

    try:
        # MERRA-2 collection search
        results = earthaccess.search_data(
            short_name=collection,
            temporal=(date_start.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d')),
            bounding_box=bbox_tuple
        )

        logger.info(f"✓ Found {len(results)} granules")
        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        logger.info("Trying alternative: concept_id search...")

        # Alternative: use concept_id for MERRA-2
        try:
            results = earthaccess.search_data(
                concept_id='C1276812863-GES_DISC',  # M2T1NXSLV v5.12.4
                temporal=(date_start.strftime('%Y-%m-%d'), date_end.strftime('%Y-%m-%d')),
                bounding_box=bbox_tuple
            )
            logger.info(f"✓ Found {len(results)} granules via concept_id")
            return results
        except Exception as e2:
            logger.error(f"Alternative search also failed: {e2}")
            return []


def download_granules(
    granules: List,
    output_dir: Path,
    max_workers: int = 8
) -> List[str]:
    """
    Download MERRA-2 granules

    Args:
        granules: List of granule objects
        output_dir: Output directory
        max_workers: Parallel workers

    Returns:
        List of downloaded file paths
    """
    if len(granules) == 0:
        logger.warning("No granules to download")
        return []

    logger.info(f"Downloading {len(granules)} granules to {output_dir}...")
    logger.info(f"Using {max_workers} parallel workers")

    try:
        files = earthaccess.download(
            granules,
            str(output_dir),
            threads=max_workers
        )

        logger.info(f"✓ Downloaded {len(files)} files")
        return files

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return []


def main():
    """Main execution"""
    logger.info("="*60)
    logger.info("MERRA-2 Meteorological Data Download")
    logger.info("="*60)
    logger.info(f"Period: {config.DATE_START.date()} to {config.DATE_END.date()}")
    logger.info(f"Region: {config.BBOX['name']}")
    logger.info(f"Variables: {', '.join(config.MERRA2_VARS)}")
    logger.info("="*60)

    # Authenticate
    setup_earthdata_auth()

    # Search for granules
    granules = search_merra2_granules(
        collection=config.MERRA2_COLLECTION,
        date_start=config.DATE_START,
        date_end=config.DATE_END,
        bbox=config.BBOX
    )

    if len(granules) == 0:
        logger.error("No MERRA-2 granules found")
        logger.info("\nNote: MERRA-2 data might require:")
        logger.info("  1. GES DISC Earthdata account")
        logger.info("  2. Subscription to MERRA-2 collections")
        logger.info("  3. Alternative: use opendap/subset service")
        return

    # Download
    files = download_granules(
        granules,
        config.RAW_MERRA2,
        max_workers=config.ARIA2_PARAMS['max_concurrent_downloads']
    )

    if len(files) == 0:
        logger.error("Failed to download MERRA-2 data")
        return

    # Calculate total size
    total_size = sum(Path(f).stat().st_size for f in files if Path(f).exists())
    size_gb = total_size / 1024**3

    logger.info("\n" + "="*60)
    logger.info("✓ MERRA-2 download complete!")
    logger.info("="*60)
    logger.info(f"  Files:    {len(files)}")
    logger.info(f"  Size:     {size_gb:.2f} GB")
    logger.info(f"  Location: {config.RAW_MERRA2}")
    logger.info("="*60)

    # Note about variable extraction
    logger.info("\nNext step:")
    logger.info(f"  Extract variables {config.MERRA2_VARS} in preprocessing script")


if __name__ == "__main__":
    main()
