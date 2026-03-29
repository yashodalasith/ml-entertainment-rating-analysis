from pathlib import Path

from src.data_utils import load_raw_tables
from src.preprocessing import prepare_kmeans_features


def main():
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw_tables(start_dir=base_dir)
    X, _, _ = prepare_kmeans_features(data["animes"], data["profiles"], data["reviews"])

    template = X.head(1).copy()
    template_path = artifacts_dir / "cluster_input_template.csv"
    template.to_csv(template_path, index=False)

    print("Template exported:", template_path)
    print("Columns required:", len(template.columns))


if __name__ == "__main__":
    main()
