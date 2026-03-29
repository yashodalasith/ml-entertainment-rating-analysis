from pathlib import Path

import pandas as pd

from src.data_utils import load_raw_tables
from src.preprocessing import prepare_supervised_dataframe


def main():
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw_tables(start_dir=base_dir)
    X, _, _, _ = prepare_supervised_dataframe(data["animes"], data["reviews"])

    # Export a single realistic starter row that users can edit.
    template = X.head(1).copy()
    template_path = artifacts_dir / "prediction_input_template.csv"
    template.to_csv(template_path, index=False)

    print("Template exported:", template_path)
    print("Columns required for prediction:", len(template.columns))


if __name__ == "__main__":
    main()
