import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "reef_features.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")


def run_eda():
    print("Loading processed data from:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    os.makedirs(FIG_DIR, exist_ok=True)

    # 1 class balancing
    class_counts = df["bleaching_present"].value_counts().sort_index()

    plt.figure()
    class_counts.plot(kind="bar")
    plt.xticks([0, 1], ["No Bleaching (0)", "Bleaching (1)"], rotation=0)
    plt.ylabel("Number of observations")
    plt.title("Class Balance: Bleaching vs No Bleaching")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "class_balance.png")
    plt.savefig(out_path)
    plt.close()
    print("1plot in:", out_path)

    # 2 dhw by class
    plt.figure()
    df[df["bleaching_present"] == 0]["dhw"].plot(kind="hist", alpha=0.6, label="No Bleaching (0)")
    df[df["bleaching_present"] == 1]["dhw"].plot(kind="hist", alpha=0.6, label="Bleaching (1)")
    plt.xlabel("Degree Heating Weeks (DHW)")
    plt.ylabel("Count")
    plt.title("DHW Distribution by Bleaching Status")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "dhw_by_class.png")
    plt.savefig(out_path)
    plt.close()
    print("2plot in:", out_path)

    # 3 sst anomaly vs DHW by bleaching
    plt.figure()
    colors = df["bleaching_present"].map({0: "blue", 1: "red"})
    plt.scatter(df["sst_anom"], df["dhw"], c=colors, alpha=0.4)
    plt.xlabel("SST Anomaly")
    plt.ylabel("Degree Heating Weeks (DHW)")
    plt.title("SST Anomaly vs DHW colored by Bleaching")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "sst_vs_dhw_scatter.png")
    plt.savefig(out_path)
    plt.close()
    print("3plot in:", out_path)

    # 4 matrix heatmap
    plt.figure()
    corr = df[["sst_anom", "dhw", "turbidity", "bleaching_present"]].corr()
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "correlation_matrix.png")
    plt.savefig(out_path)
    plt.close()
    print("4plot in:", out_path)



if __name__ == "__main__":
    run_eda()
