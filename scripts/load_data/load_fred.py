import sys
from pathlib import Path

sys.path.append("../../src/fred")
import fred

FRED_FACTORS = [
    'UNRATE',
    'PAYEMS',
    'LNU04032231',
    'ICSA',
    'CCSA',
    'IURSA',
    'GDPC1',
    'SP500',
    'DJIA',
    'NASDAQCOM',
    'NASDAQ100',
]

def main():
    def _load_fred(factor:str, save_dir:Path):
            path_key = '../../src/fred/fred_key.txt'
            df = fred.get_data(path_key, factor)
            df.to_csv(save_dir / f'{factor}.csv', index=False)
            
    save_dir = Path("../../data/raw/macro")
    for factor in FRED_FACTORS:
        _load_fred(factor, save_dir)
    
if __name__ == "__main__":
    main()