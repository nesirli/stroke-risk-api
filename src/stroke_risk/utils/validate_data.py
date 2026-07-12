import great_expectations as gx
import pandas as pd
from stroke_risk.ingest.load_data import load_stroke_data

REQUIRED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]

CATEGORICAL_VALUES = {
    "gender": ["male", "female", "other"],
    "ever_married": ["yes", "no"],
    "work_type": ["private", "self_employed", "govt_job", "children", "never_worked"],
    "residence_type": ["urban", "rural"],
    "smoking_status": ["formerly_smoked", "never_smoked", "smokes", "unknown"],
}


def validate_stroke_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate a stroke dataset against schema and value-range expectations, raising on failure."""
    print("Starting data validation with Great Expectations.")

    df = pd.concat([X, y], axis=1)

    context = gx.get_context(mode="ephemeral")

    batch = (
        context.data_sources.add_pandas("pandas")
        .add_dataframe_asset("stroke_data")
        .add_batch_definition_whole_dataframe("batch")
        .get_batch(batch_parameters={"dataframe": df})
    )

    suite = context.suites.add(gx.ExpectationSuite(name="stroke_suite"))

    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(column_set=REQUIRED_COLUMNS)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age", min_value=0, max_value=120
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="avg_glucose_level", min_value=0, max_value=400
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="bmi", min_value=5, max_value=120
        )
    )

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="hypertension", value_set=[0, 1]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="heart_disease", value_set=[0, 1]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="stroke", value_set=[0, 1])
    )

    for col, values in CATEGORICAL_VALUES.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(column=col, value_set=values)
        )

    result = batch.validate(suite)

    if not result.success:
        failed_details = []
        for r in result.results:
            if not r.success:
                exp_type = r.expectation_config.type
                col = r.expectation_config.kwargs.get("column", "table-level")
                failed_details.append(f"- {exp_type} on '{col}'")

        error_msg = "\n".join(failed_details)
        raise ValueError(
            f"Data validation failed on the following checks:\n{error_msg}"
        )

    print("Data validation completed successfully.")


if __name__ == "__main__":
    X, y = load_stroke_data()
    validate_stroke_data(X, y)
