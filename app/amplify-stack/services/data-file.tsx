// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
// vim: set ts=2 sw=2:

import Storage from "@aws-amplify/storage";
import _ from "lodash";
import { DateTime } from "luxon";
import Papa from "papaparse";
import { AfModelResults, DataAnalysisResults, ForecastJobInfo, ForecastJobResults, ForecastResults, ModelResults } from "./forecasting";

interface S3File {
  Body: Blob;
  LastModified: Date;
}

export const parseReportData = async (
  metadata: ForecastJobInfo,
  csvReportFile: S3File | undefined,
  resultsJsonFile: S3File | undefined,
  forecastFile: S3File | undefined,
  afResultsJsonFile: S3File | undefined,
  afForecastFile: S3File | undefined,
  afParamsFile: S3File | undefined,
): Promise<ForecastJobResults> => {
  let dataAnalyis: DataAnalysisResults | undefined;
  let modelResults: ModelResults | undefined;
  let forecast: ForecastResults | undefined;
  let afModelResults: AfModelResults | undefined;
  let afForecast: ForecastResults | undefined;

  let skuDemandCategoryLookup = {};

  console.log("afParamsFile:", afParamsFile) ;

  if (csvReportFile) {
    const lookups = {
      skus: {},
      channels: {},
      families: {},
      demandsCategories: {},
    };
    let totalSpectralEntropy = 0;
    let totalRows = 0;

    await new Promise((res, rej) =>
      Papa.parse<any>(csvReportFile.Body as any, {
        header: true,
        worker: false,
        delimiter: ",",
        skipEmptyLines: true,
        step: (row: any) => {
          if (row.errors.length) {
            console.error(`Failed to parse rows from .csv.report file: `, row.errors);
            return;
          }

          lookups.skus[row.data.item_id] = true;
          lookups.channels[row.data.channel] = true;
          // TODO: verify once column added to CSV
          lookups.families[row.data.family] = true;
          lookups.demandsCategories[row.data.category] = (lookups.demandsCategories[row.data.category] || 0) + 1;
          skuDemandCategoryLookup[row.data.item_id] = row.data.category;
          totalSpectralEntropy += Number(row.data.spectralEntropy);
          totalRows++;
        },
        complete: res,
        error: rej,
      })
    );

    dataAnalyis = {
      skus: Object.keys(lookups.skus).length,
      channels: Object.keys(lookups.channels).length,
      families: Object.keys(lookups.families).length,
      frequency: metadata.frequencyUnit as any,
      horizon: {
        amount: metadata.horizonAmount,
        unit: metadata.horizonUnit as any,
      },
      demandCategories: {
        continuous: (lookups.demandsCategories["Continuous"] || 0) / totalRows,
        medium: (lookups.demandsCategories["Medium"] || 0) / totalRows,
        short: (lookups.demandsCategories["Short"] || 0) / totalRows,
      },
      seasonalityIndex: totalSpectralEntropy / totalRows,
      historicalData: {
        startDate: undefined,
        endDate: undefined,
      },
      testingPeriodsCompleted: undefined,
    };
  }

  if (resultsJsonFile) {
    let json = await new Promise<string>((res, rej) => {
      const reader = new FileReader();
      reader.onload = () => res(reader.result as string);
      reader.onerror = rej;
      reader.readAsText(resultsJsonFile.Body);
    });

    json = json.replace(/\bNaN\b/g, "null");
    const data = JSON.parse(json) as any[];

    // We can determine the report generation time by using the difference
    // between the last-modified of the S3 objects and the metedata created time
    const completionDurationMs =
      csvReportFile && resultsJsonFile && forecastFile
        ? _.max([csvReportFile, resultsJsonFile, forecastFile].map((i) => i.LastModified)).getTime() -
        metadata.createdAt.getTime()
        : undefined;

    const avgNaiveError = _.sumBy(data, (i: any) => i.Naive_Error) / data.length;
    const avgVoyagerError = _.sumBy(data, (i: any) => i.Voyager_Error) / data.length;
    // const accuracyIncrease = ((avgNaiveError - avgVoyagerError) / avgNaiveError) * 100;
    const accuracyIncrease = ((1.0 - avgVoyagerError) - (1.0 - avgNaiveError)) * 100;
    // const accuracyIncrease = ((100.0 - avgVoyagerError) - (100.0 - avgNaiveError)) ;
    console.log("data.length", data.length);
    console.log("avgNaiveError", avgNaiveError);
    console.log("avgVoyagerError", avgVoyagerError);
    console.log("accuracyIncrease", accuracyIncrease);
    console.log("DONE: parseReportData()");

      console.log(data) ;

    const modelLookup = {};

    for (const itemResults of data) {
      modelLookup[itemResults.model_name] = (modelLookup[itemResults.model_name] || 0) + 1;
    }

    modelResults = {
      competionDurationSeconds: completionDurationMs / 1000,
      accuracyIncrease,
      bestPerformingModels: _.mapValues(modelLookup, (i) => i / data.length),
      avgVoyagerError: avgVoyagerError,
      avgNaiveError: avgNaiveError
    };

    if (dataAnalyis) {
      dataAnalyis.historicalData.startDate = _.min(
        data.map((i) => DateTime.fromFormat(i.first_date, "yyyy-MM-dd")).filter((i) => i.isValid)
      )?.toJSDate();
      dataAnalyis.historicalData.endDate = _.max(
        data.map((i) => DateTime.fromFormat(i.last_date, "yyyy-MM-dd")).filter((i) => i.isValid)
      )?.toJSDate();

      if (dataAnalyis.historicalData.startDate && dataAnalyis.historicalData.endDate) {
        dataAnalyis.testingPeriodsCompleted = Math.max(
          0,
          DateTime.fromJSDate(dataAnalyis.historicalData.endDate)
            .diff(DateTime.fromJSDate(dataAnalyis.historicalData.startDate))
            .as("weeks") - 8
        );
      }
    }
  }

  if (forecastFile) {
    let data: ForecastResults["data"] = [];

    await new Promise((res, rej) =>
      Papa.parse<any>(forecastFile.Body as any, {
        header: true,
        worker: false,
        delimiter: ",",
        skipEmptyLines: true,
        step: (row: any) => {
          if (row.errors.length) {
            console.error(`Failed to parse rows from .csv.report file: `, row.errors);
            return;
          }

          let date = new Date(`${row.data.index}T00:00:00`);
          date = normaliseDate(date, metadata.frequencyUnit);
          const actual = row.data.demand ? Number(row.data.demand) : undefined;
          const forecast = row.data.forecast ? Number(row.data.forecast) : undefined;
          const backtest = row.data.forecast_backtest ? Number(row.data.forecast_backtest) : undefined;

          data.push({
            x: date,
            yActual: actual,
            yForecast: forecast || backtest,
            sku: row.data.item_id,
            demandCategory: skuDemandCategoryLookup[row.data.item_id],
            channel: row.data.channel,
            family: row.data.family,
          });
        },
        complete: res,
        error: rej,
      })
    );

    const keyWithoutPrefix = _.last(metadata.forecastS3Key.split("/"));
    forecast = {
      forecastS3Key: metadata.forecastS3Key,
      signedUrl: (await Storage.get(keyWithoutPrefix, { level: "private", expires: 3600 })) as string,
      frequency: metadata.frequencyUnit as any,
      data,
    };
  }

  if (afResultsJsonFile) {
    let json = await new Promise<string>((res, rej) => {
      const reader = new FileReader();
      reader.onload = () => res(reader.result as string);
      reader.onerror = rej;
      reader.readAsText(afResultsJsonFile.Body);
    });
    console.log(`afResultsJsonFile: json:`, json)

    const data = JSON.parse(json);
    console.log(`afResultsJsonFile: data:`, data)

    const mape = data["mape"];
    const bestPerformingModel = data["selected_algo"];
    afModelResults = {
      mape,
      bestPerformingModel,
    };

    console.log(`afModelResults:`, afModelResults)
  }

  if (afForecastFile) {
    let data: ForecastResults["data"] = [];

    await new Promise((res, rej) =>
      Papa.parse<any>(afForecastFile.Body as any, {
        header: true,
        worker: false,
        delimiter: ",",
        skipEmptyLines: true,
        step: (row: any) => {
          if (row.errors.length) {
            console.error(`Failed to parse rows from .csv.report file: `, row.errors);
            return;
          }

          let date = new Date(`${row.data.index}T00:00:00`);
          date = normaliseDate(date, metadata.frequencyUnit);
          const actual = row.data.demand ? Number(row.data.demand) : undefined;
          const forecast = row.data.forecast ? Number(row.data.forecast) : undefined;
          const backtest = row.data.forecast_backtest ? Number(row.data.forecast_backtest) : undefined;

          data.push({
            x: date,
            yActual: actual,
            yForecast: forecast || backtest,
            sku: row.data.item_id,
            demandCategory: skuDemandCategoryLookup[row.data.item_id],
            channel: row.data.channel,
            family: row.data.family,
          });
        },
        complete: res,
        error: rej,
      })
    );

    const keyWithoutPrefix = _.last(metadata.forecastS3Key.replace(".csv.forecast.csv", ".af-resampled.csv").split("/"));
    afForecast = {
      forecastS3Key: metadata.forecastS3Key,
      signedUrl: (await Storage.get(keyWithoutPrefix, { level: "private", expires: 3600 })) as string,
      frequency: "D" as any,  // Our Amazon Forecast pipeline generates daily forecast.
      data,
    };
  }

  let status: ForecastJobResults["status"];

  //if (dataAnalyis && modelResults && forecastFile && afResultsJsonFile && afForecastFile) {
  if (dataAnalyis && modelResults && forecastFile) {
    status = "complete";
  } else if (dataAnalyis || modelResults || forecastFile || afModelResults || afForecastFile) {
    status = "partial";
  } else {
    status = "generating";
  }

    // if (dataAnalyis && modelResults && forecastFile) {
    //   status = "complete";
    // } else if (dataAnalyis || modelResults || forecastFile || afModelResults || afForecastFile) {
    //   status = "partial";
    // } else {
    //   status = "generating";
    // }

  return { status, dataAnalyis, modelResults, forecast, afModelResults, afForecast, afParamsFile };
};

const normaliseDate = (date: Date, frequency: string): Date => {
  switch (frequency) {
    case "M":
      date.setDate(1);
      break;

    case "W":
      // TODO
      break;

    case "D":
    default:
      break;
  }

  return date;
};
