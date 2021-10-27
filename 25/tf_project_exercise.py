"""
Tensorflow Project Exercise
"""
from os.path import join, dirname
from pandas import DataFrame, read_csv
from seaborn import countplot, pairplot
from matplotlib.pyplot import show
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow import feature_column, estimator, compat


def wrapper():
    """
    wrapper function
    """
    data = DataFrame(read_csv(join(dirname(__file__), "bank_note_data.csv")))
    print(data.head())
    countplot(x="Class", data=data)
    pairplot(data, hue="Class")
    scaler = StandardScaler()
    scaler.fit(data.drop("Class", axis=1))
    scaled_features = scaler.fit_transform(data.drop("Class", axis=1))
    df_feat = DataFrame(scaled_features, columns=data.columns[:-1])
    df_feat.head()
    xda = df_feat
    yda = data["Class"]
    x_train, x_test, y_train, y_test = train_test_split(xda, yda, test_size=0.3)
    print(df_feat.columns)
    image_var = feature_column.numeric_column("Image.Var")
    image_skew = feature_column.numeric_column("Image.Skew")
    image_curt = feature_column.numeric_column("Image.Curt")
    entropy = feature_column.numeric_column("Entropy")
    feat_cols = [image_var, image_skew, image_curt, entropy]

    wrapper2(x_train, x_test, y_train, y_test, feat_cols)


def wrapper2(x_train, x_test, y_train, y_test, feat_cols):
    """
    wrapper function part 2
    """
    classifier = estimator.DNNClassifier(
        hidden_units=[10, 20, 10], n_classes=2, feature_columns=feat_cols
    )
    input_func = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_train, y=y_train, batch_size=20, shuffle=True
    )
    classifier.train(input_fn=input_func, steps=500)
    pred_fn = compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test, batch_size=len(x_test), shuffle=False
    )
    note_predictions = list(classifier.predict(input_fn=pred_fn))
    print(note_predictions[0])
    final_preds = []
    for pred in note_predictions:
        final_preds.append(pred["class_ids"][0])
    print(confusion_matrix(y_test, final_preds))
    print(classification_report(y_test, final_preds))
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(x_train, y_train)
    rfc_preds = rfc.predict(x_test)
    print(classification_report(y_test, rfc_preds))
    print(confusion_matrix(y_test, rfc_preds))


if __name__ == "__main__":
    wrapper()
    show()
