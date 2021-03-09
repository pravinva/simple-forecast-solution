// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useState, useRef, useEffect } from "react";
import styles from "./Upload.module.scss";
import { Link } from "../../common/link/Link";
import { Panel } from "../../common/panel/Panel";
import { IconList } from "../../common/icon-list/IconList";
import Check from "@material-ui/icons/Check";
import Error from "@material-ui/icons/Error";
import Warning from "@material-ui/icons/Warning";
import {
  TextField,
  Select,
  FormControl,
  MenuItem,
  InputLabel,
  FormHelperText,
  LinearProgress,
} from "@material-ui/core";
import { Button } from "../../common/button/Button";
import {
  ForecastingService,
  UploadDataFileResponse,
  ForecastFileValidationResponse,
  CreateForecastJobResponse,
} from "../../../services/forecasting";
import Router from "next/router";

const service = new ForecastingService();

enum State {
  Initial,
  Uploading,
  Validating,
  Invalid,
  CreatingJob,
  Done,
  Failed,
}

export const Upload: React.SFC = () => {
  const [state, setState] = useState(State.Initial);
  const [uploadProgress, setUploadProgress] = useState(0);

  const [frequencyUnit, setFrequencyUnit] = useState<string>("");
  const [frequencyUnitError, setFrequencyUnitError] = useState<string | undefined>();

  const [horizonAmount, setHorizonAmount] = useState<number | "">("");
  const [horizonAmountError, setHorizonAmountError] = useState<string | undefined>();
  const [horizonUnit, setHorizonUnit] = useState<string>("");
  const [horizonUnitError, setHorizonUnitError] = useState<string | undefined>();

  const [generalError, setGeneralError] = useState<string | undefined>();
  const [fileValidation, setFileValidation] = useState<ForecastFileValidationResponse["validations"]>();

  const fileInputRef = useRef<HTMLInputElement>();
  const progressRef = useRef<HTMLDivElement>();

  useEffect(() => {
    if ((state === State.Uploading || state === State.Invalid) && progressRef.current) {
      progressRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [progressRef, state]);

  const validate = (): boolean => {
    setFrequencyUnitError(undefined);
    setHorizonAmountError(undefined);
    setHorizonUnitError(undefined);
    let valid = true;

    if (!frequencyUnit) {
      setFrequencyUnitError("Please select a unit");
      valid = false;
    }

    if (!horizonAmount && horizonAmount !== 0) {
      setHorizonAmountError("Please enter an amount");
      valid = false;
    } else if (horizonAmount < 1) {
      setHorizonAmountError("Please enter an amount greater than 0");
      valid = false;
    }

    if (!horizonUnit) {
      setHorizonUnitError("Please select a unit");
      valid = false;
    }

    return valid;
  };

  const handleUploadClick = () => {
    if (!validate()) {
      return;
    }

    fileInputRef.current.click();
  };

  const handleNewFile = async (file: File) => {
    try {
      console.log(`Uploading file to S3...`);
      setState(State.Uploading);
      setUploadProgress(0);
      const s3File = await service.uploadDataFile(file, setUploadProgress);

      await validateDataFile(s3File);
    } catch (e) {
      console.error(`Failed to upload CSV file`, e);
      setGeneralError("An error occurred while uploading the CSV file, please try again later");
      setState(State.Failed);
    }
  };

  const validateDataFile = async (file: UploadDataFileResponse) => {
    try {
      console.log(`Validating uploaded data file...`);
      setState(State.Validating);
      const response = await service.validateDataFile({
        ...file,
        frequencyUnit,
        horizonAmount: horizonAmount as number,
        horizonUnit,
          isAfJob: false
      });

      if (response.status === "failed") {
        setState(State.Invalid);
        setFileValidation(response.validations);
      } else {
        await createForecastJob(file);
      }
    } catch (e) {
      console.error(`Failed to validate uploaded CSV file`, e);
      setGeneralError("An error occurred while validating the uploaded CSV file, please try again later");
      setState(State.Failed);
    }
  };

  const createForecastJob = async (file: UploadDataFileResponse) => {
    try {
      console.log(`Creating forecast job...`);
      setState(State.CreatingJob);
      const jobInfo = await service.createForecastJob({
        fileS3Key: file.fileS3Key,
        frequencyUnit,
        horizonAmount: horizonAmount as number,
        horizonUnit: horizonUnit as string,
          isAfJob: false
      });
      console.log(`Successfully created forecasting job: `, jobInfo);

      handleSuccessfullyCreatedJob(jobInfo);
    } catch (e) {
      console.error(`Failed to create forecast job`, e);
      setGeneralError("An error occurred while generating for the forecast, please try again later");
      setState(State.Failed);
    }
  };

  const handleSuccessfullyCreatedJob = (job: CreateForecastJobResponse) => {
    setState(State.Done);
    Router.push(`/authenticated/report?id=${encodeURIComponent(job.id)}`);
  };

  return (
    <div className={styles.upload}>
      <Panel className={styles.panel} subTitle="Step 1" title="Review data requirements">
        <IconList>
          <li>
            <Check /> 2+ years historical sales
          </li>
          <li>
            <Check /> Timestamp (daily or weekly sales)
          </li>
          <li>
            <Check /> Unique product ID
          </li>
          <li>
            <Check /> Demand (# sales)
          </li>
          <li>
            <Check /> Channel (e.g. Online, In-Store etc.) &nbsp;&nbsp;<em>Optional</em>
          </li>
          <li>
            <Check /> Family (e.g. Style, Tops, Accessories etc.) &nbsp;&nbsp;<em>Optional</em>
          </li>
        </IconList>
      </Panel>

      <Panel className={styles.panel} subTitle="Step 2" title="Need guidance? Download the data template">
        <Link underline download href="/static/sfs-data-template.csv">
          Download Data Template
        </Link>
      </Panel>

      <Panel className={styles.panel} subTitle="Step 3" title="Choose desired future forecast horizon">
        <div className={styles.frequency}>
          <span>My sales data is in:</span>
          <FormControl variant="outlined">
            <InputLabel>Frequency</InputLabel>
            <Select
              required
              label="Frequency"
              value={frequencyUnit || ""}
              onChange={(e) => setFrequencyUnit(e.target.value as string)}
              error={!!frequencyUnitError}
            >
              <MenuItem value="D">Days</MenuItem>
              <MenuItem value="W">Weeks</MenuItem>
              <MenuItem value="M">Months</MenuItem>
            </Select>
            {frequencyUnitError && <FormHelperText>{frequencyUnitError}</FormHelperText>}
          </FormControl>
        </div>
        <div className={styles.horizon}>
          <span>I would like to forecast forward:</span>
          <TextField
              InputProps={{
                  inputProps: {
                      min: 1
                  }
              }}
            label="Amount"
            type="number"
            variant="outlined"
            value={horizonAmount}
            onChange={(e) => setHorizonAmount(e.target.value !== "" ? Number(e.target.value) : undefined)}
            error={!!horizonAmountError}
            helperText={horizonAmountError}
          />
          <FormControl variant="outlined">
            <InputLabel>Unit</InputLabel>
            <Select
              required
              label="Unit"
              value={horizonUnit || ""}
              onChange={(e) => setHorizonUnit(e.target.value as string)}
              error={!!horizonUnitError}
            >
              <MenuItem value="D" disabled={true}>Days</MenuItem>
              <MenuItem value="W">Weeks</MenuItem>
              <MenuItem value="M">Months</MenuItem>
            </Select>
            {horizonUnitError && <FormHelperText>{horizonUnitError}</FormHelperText>}
          </FormControl>
        </div>
      </Panel>

      <Panel ref={progressRef} className={styles.panel} subTitle="Step 4">
        <div className={styles.csv}>
          <h2>Upload CSV</h2>
          <Button
            variant="contained"
            color="primary"
            onClick={handleUploadClick}
            disabled={state !== State.Initial && state !== State.Invalid && state !== State.Failed}
          >
            Upload CSV
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="text/csv"
            onClick={(e) => (e.target as any).value = null}
            onChange={(e) => e.target.files[0] && handleNewFile(e.target.files[0])}
          />
        </div>

        <div className={styles.progress}>
          {state === State.Uploading && (
            <div className="progressing">
              <h4>Securely uploading your CSV file...</h4>
              <LinearProgress variant="determinate" value={uploadProgress * 70} />
            </div>
          )}

          {state === State.Validating && (
            <div className="progressing">
              <h4>We are validating your file...</h4>
              <LinearProgress variant="determinate" value={80} />
            </div>
          )}

          {state === State.Invalid && (
            <div className="error">
              <h4>Oops, your CSV file did not pass validation</h4>
              <p>Please review the following errors and re-upload your amended CSV file</p>
              <IconList>
                {fileValidation
                  ?.filter((i) => i.type === "error")!
                  ?.map((i) => (
                    <li>
                      <Error className="validation--error" /> {i.message}
                    </li>
                  ))}
                {fileValidation
                  ?.filter((i) => i.type === "warning")!
                  ?.map((i) => (
                    <li>
                      <Warning className="validation--warning" /> {i.message}
                    </li>
                  ))}
              </IconList>
            </div>
          )}

          {state === State.CreatingJob && (
            <div className="progressing">
              <h4>Starting your forecast...</h4>
              <LinearProgress variant="determinate" value={90} />
            </div>
          )}

          {state === State.Failed && (
            <div className="error">
              <h4>Oops, an error occurred</h4>
              <p>{generalError}</p>
            </div>
          )}
        </div>
      </Panel>
    </div>
  );
};
