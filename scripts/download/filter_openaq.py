"""
OpenAQ 다운로드된 데이터를 학습 기간으로 필터링
2016-2025 전체 기간 → 2023-09-01 ~ 2023-10-13 (6주)
"""
import pandas as pd
import config
import utils

logger = utils.setup_logging(__name__)


def main():
    logger.info("="*60)
    logger.info("OpenAQ 데이터 필터링")
    logger.info("="*60)

    # CSV 로드
    logger.info(f"Loading {config.OPENAQ_CSV}...")
    df = pd.read_csv(config.OPENAQ_CSV)
    logger.info(f"  Original: {len(df):,} rows")

    # 시간 컬럼 datetime 변환 (UTC timezone-naive로 정규화)
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = utils.to_utc_naive(df['time'])  # timezone 제거 (UTC 기준)

    # 날짜 범위 필터링
    logger.info(f"\nFiltering to: {config.DATE_START.date()} ~ {config.DATE_END.date()}")
    df_filtered = df[(df['time'] >= config.DATE_START) & (df['time'] < config.DATE_END)]

    logger.info(f"  Filtered: {len(df_filtered):,} rows")
    logger.info(f"  Dropped:  {len(df) - len(df_filtered):,} rows")

    if len(df_filtered) == 0:
        logger.error("필터링 후 데이터 없음. 날짜 범위 확인 필요.")
        return

    # 저장
    df_filtered.to_csv(config.OPENAQ_CSV, index=False)

    size_mb = config.OPENAQ_CSV.stat().st_size / 1024**2
    logger.info(f"\n✓ Saved: {config.OPENAQ_CSV} ({size_mb:.2f} MB)")

    # 요약
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info("="*60)
    logger.info(f"  Date range:  {df_filtered['time'].min()} to {df_filtered['time'].max()}")
    logger.info(f"  Locations:   {df_filtered['location'].nunique()}")
    logger.info(f"  Total hours: {len(df_filtered):,}")
    logger.info(f"  PM2.5 mean:  {df_filtered['pm25'].mean():.1f} µg/m³")
    logger.info(f"  PM2.5 std:   {df_filtered['pm25'].std():.1f} µg/m³")
    logger.info("="*60)


if __name__ == "__main__":
    main()
