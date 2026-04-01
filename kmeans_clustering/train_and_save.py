from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ModuleNotFoundError:
    HAS_PLOTTING = False

from src.clustering import train_cluster_and_save
from src.data_utils import load_raw_tables
from src.preprocessing import prepare_kmeans_features


def main():
    base_dir = Path(__file__).resolve().parent
    data = load_raw_tables(start_dir=base_dir)

    X, analysis_df, meta = prepare_kmeans_features(
        data["animes"], data["profiles"], data["reviews"]
    )

    output = train_cluster_and_save(
        X=X,
        analysis_df=analysis_df,
        artifacts_dir=base_dir / "artifacts",
    )

    artifacts_dir = base_dir / "artifacts"
    projection = output["cluster_projection"].copy()

    projection_plot_path = artifacts_dir / "cluster_projection_plot.png"
    size_plot_path = artifacts_dir / "cluster_size_plot.png"

    if HAS_PLOTTING:
        # 2D projection plot for visual cluster separation.
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=projection.sample(min(5000, len(projection)), random_state=42),
            x="svd_1",
            y="svd_2",
            hue="cluster_label",
            alpha=0.7,
            s=24,
        )
        plt.title("K-Means Clusters in 2D SVD Space")
        plt.xlabel("SVD Component 1")
        plt.ylabel("SVD Component 2")
        plt.tight_layout()
        plt.savefig(projection_plot_path, dpi=150)
        plt.close()

        # Cluster size distribution for quick balance check.
        size_plot = output["cluster_profile"][["cluster", "cluster_size"]].copy()
        plt.figure(figsize=(8, 5))
        sns.barplot(data=size_plot, x="cluster", y="cluster_size", color="#4C78A8")
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster")
        plt.ylabel("Number of Anime")
        plt.tight_layout()
        plt.savefig(size_plot_path, dpi=150)
        plt.close()

    print("Training complete.")
    print("Best K:", output["best_k"])
    print("SVD components:", output["n_svd"])
    print("Saved model:", output["model_path"])
    print("Saved projection CSV:", artifacts_dir / "cluster_projection.csv")
    if HAS_PLOTTING:
        print("Saved projection plot:", projection_plot_path)
        print("Saved cluster-size plot:", size_plot_path)
    else:
        print("Plot files skipped: matplotlib/seaborn not installed in this environment.")
    print("\nCluster meanings:")
    print(output["cluster_meanings"].to_string(index=False))
    print("\nRepresentative samples per cluster:")
    print(output["cluster_examples"].head(12).to_string(index=False))
    print("Feature summary:")
    print("- numeric columns:", len(meta["numeric_cols"]))
    print("- categorical columns:", len(meta["categorical_cols"]))


if __name__ == "__main__":
    main()
