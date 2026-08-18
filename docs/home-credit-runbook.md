# Home Credit public-data runbook

1. Download the Home Credit CSV files yourself from their data provider. The repository neither distributes the CSV files nor retrieves data remotely.
2. Keep the downloaded CSV directory outside this repository. It must contain `application_train.csv` and at least one supported history table: `previous_application.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, or `bureau.csv`.
3. Prepare the local Parquet without printing samples:

   ```bash
   riskprobe prepare-home-credit --input-dir "/absolute/path/to/home-credit" --output /tmp/home-credit-riskprobe.parquet
   ```

4. Copy `configs/home_credit.example.yaml` outside the repository or to an ignored local location, and update only its Parquet path.
5. Run `riskprobe run` against that local configuration. The example contract fixes `time_validation_enabled: false` and `snapshot.meaning: public_relative_reference`.

The output may be described only as a fixed-seed stratified Train/Test and cross-customer-segment validation. Home Credit relative history fields are not absolute application dates; do not describe results as OOT, time-slice evidence, cross-institution validation, online performance, or production impact. Generated Parquet files are protected by `.gitignore` and must not be committed.
