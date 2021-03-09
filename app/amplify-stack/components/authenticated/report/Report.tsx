// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
// vim: set ts=2 sw=2:

import React, { useEffect, useState, useRef } from "react";
import styles from "./Report.module.scss";
import { ForecastJobInfo, ForecastingService, ForecastJobResults } from "../../../services/forecasting";
import { Preloader } from "../../common/preloader/Preloader";
import { Analysing } from "./analysing/Analysing";
import { Results } from "./results/Results";

interface Props {
  report?: ForecastJobInfo;
}

const forecastingService = new ForecastingService();

export const Report = ({ report }) => {
  /*
   *  export interface ForecastJobResults {
   *    status: "generating" | "partial" | "complete";
   *    dataAnalyis?: DataAnalysisResults;
   *    modelResults?: ModelResults;
   *    forecast?: ForecastResults;
   *    afModelResults?: AfModelResults;
   *    afForecast?: ForecastResults;
   *  }
   */
  const [engineResults, setEngineResults] = useState<ForecastJobResults>();
  const hasEngineLoaded = useRef(false);

  useEffect(() => {
    if (!report) {
      return;
    }

    hasEngineLoaded.current = false;
    setEngineResults(undefined);
    loadEngineResults(report);

    // If the forecast has not finished generating we will poll
    // the S3 bucket until the files have been created
    const intervalId = setInterval(() => {
      if (!hasEngineLoaded.current) {
        loadEngineResults(report);
      }
    }, 10000);

    return () => {
      clearInterval(intervalId);
    };
  }, [report]);

  useEffect(() => {}, [report]);

  //
  // loadEngineResults
  //
  const loadEngineResults = async (report: ForecastJobInfo) => {
    const engineResults = 
      await forecastingService.loadEngineForecastJobResults(report);

    setEngineResults(engineResults);

    if (engineResults.status === "complete") {
      hasEngineLoaded.current = true;
    }
  };

  return (
    <div className={styles.container}>
      {(!report || !engineResults) && <Preloader />}
      {report && engineResults && engineResults.status === "generating" && <Analysing />}
      {report && engineResults && engineResults.status !== "generating" && <Results report={report} results={engineResults} />}
    </div>
  );
};
