from sklearn.model_selection import cross_val_predict

import main


def run():
    df = main.add_phq_columns(main.load_data("data/DATASET-Raw_data_clean.csv"))
    X, y, groups = main.prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = main.split_data(X, y, groups)

    y_true = y_train.to_numpy()
    model = main.build_model()
    probabilities = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba")[:, 1]

    positives = y_true.sum()
    print("Training students:", len(y_true))
    print("Answered above 0 on item 9:", positives)
    print()
    print("cutoff  invited  invited_rate  caught  missed  recall  precision")

    for cutoff in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]:
        predictions = (probabilities >= cutoff).astype(int)
        invited = predictions.sum()
        caught = ((predictions == 1) & (y_true == 1)).sum()
        missed = positives - caught
        recall = caught / positives
        precision = caught / invited if invited else 0.0
        print(
            f"{cutoff:6.2f}  {invited:7d}  {invited / len(y_true):12.2f}"
            f"  {caught:6d}  {missed:6d}  {recall:6.2f}  {precision:9.2f}"
        )


if __name__ == "__main__":
    run()
