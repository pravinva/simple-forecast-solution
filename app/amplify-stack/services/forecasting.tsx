// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
import API from "@aws-amplify/api";
import Auth from "@aws-amplify/auth";
import Storage from "@aws-amplify/storage";
import { DateTime } from "luxon";
import * as uuid from "uuid";
import { parseReportData } from "./data-file";

export interface UploadDataFileResponse {
  fileS3Key: string;
}

export interface ForecastFileValidationResponse {
  status: "passed" | "failed";
  validations: Array<{
    type: "error" | "warning";
    message: string;
  }>;
}

export interface ForecastJobInput {
  fileS3Key: string;
  frequencyUnit: string;
  horizonAmount: number;
  horizonUnit: string;
  isAfJob: boolean;
}

export interface CreateForecastJobResponse {
  id: string;
  http_code: number;
  msg: string;
  dataAnalysisS3Key: string;
  resultsS3Key: string;
  forecastS3Key: string;
}

export interface ForecastJobInfo {
  id: string;
  name: string;
  frequencyUnit: string;
  horizonAmount: number;
  horizonUnit: string;
  dataAnalysisS3Key: string;
  resultsS3Key: string;
  forecastS3Key: string;
  createdAt: Date;
}

export interface ForecastJobResults {
  status: "generating" | "partial" | "complete";
  dataAnalyis?: DataAnalysisResults;
  modelResults?: ModelResults;
  forecast?: ForecastResults;
  afParamsFile?: any ;
  afModelResults?: AfModelResults;
  afForecast?: ForecastResults;
}

export interface DataAnalysisResults {
  skus: number;
  channels: number;
  families: number;
  frequency: "D" | "W" | "M";
  horizon: {
    amount: number;
    unit: "D" | "W" | "M";
  };
  historicalData: {
    startDate: Date;
    endDate: Date;
  };
  demandCategories: {
    continuous: number;
    medium: number;
    short: number;
  };
  seasonalityIndex: number;
  testingPeriodsCompleted: number;
}

export interface ModelResults {
  competionDurationSeconds?: number;
  accuracyIncrease: number;
  bestPerformingModels: {
    [name: string]: number;
  };
  avgVoyagerError: number;
  avgNaiveError: number;
}

export interface AfModelResults {
  mape: number;
  bestPerformingModel: string;
};

export interface ForecastResults {
  forecastS3Key: string;
  signedUrl: string;
  frequency: "D" | "W" | "M";
  data: Array<{
    x: Date;
    yActual: number | undefined;
    yForecast: number | undefined;
    sku: string;
    demandCategory: string;
    channel: string;
    family: string;
  }>;
}

export class ForecastingService {
  public uploadDataFile = async (
    file: File,
    progressCallback: (progress: number) => void = () => { }
  ): Promise<UploadDataFileResponse> => {
    const fileS3Key = `${uuid.v4()}.csv`;

    await Storage.put(fileS3Key, file, {
      progressCallback: (event) => progressCallback(event.loaded / event.total),
      level: "private",
      contentType: "text/csv"
    });

    return { fileS3Key };
  };

  public validateDataFile = async (input: ForecastJobInput): Promise<ForecastFileValidationResponse> => {
    console.log("Validating forecast data file...");

    const response = await API.post("voyagerValidateApi", "/validate/", {
      body: {
        ...input,
        fileS3Key: await this.getFileS3Key(input),
      },
    });

    return response;
  };

  public createForecastJob = async (input: ForecastJobInput): Promise<CreateForecastJobResponse> => {
    console.log("Validating forecast data file...");

    const response = await API.post("voyagerForecastApi", "/forecast/", {
      body: {
        ...input,
        fileS3Key: await this.getFileS3Key(input),
      },
    });

    return response;
  };

  private getFileS3Key = async (input: ForecastJobInput): Promise<string> => {
    const credentials = await Auth.currentUserCredentials();
    const identityId = credentials.identityId;
    return `private/${identityId}/${input.fileS3Key}`;
  };

  public loadEngineForecastJobResults =
    async (input: ForecastJobInfo): Promise<ForecastJobResults> => {
      console.log("Loading Engine Forecast job...")

      const s3ErrorHandler = (e) => {
        // Ignore 404 s3 errors
        if (e?.response?.status === 404) {
          return undefined;
        }
        return undefined;
      };

      const credentials = await Auth.currentUserCredentials();
      const identityId = credentials.identityId;
      const removePrivatePrefix = (key) => key.replace(`private/${identityId}/`, "");
      const downloadFile = (key) =>
        Storage.get(removePrivatePrefix(key), { download: true, level: "private" }).catch(s3ErrorHandler);

      // TODO: move to dedicated amazon forecast function
      const afResultsS3Key = 
        input.resultsS3Key.replace(".csv.results.json",
                                   ".input.csv.results.json.AF") ;
      const afForecastS3Key =
        input.forecastS3Key.replace(".csv.forecast.csv", ".af-resampled.csv") ;

      const s3Requests = await Promise.all([
        downloadFile(input.dataAnalysisS3Key),
        downloadFile(input.resultsS3Key),
        downloadFile(input.forecastS3Key),

        // TODO: move to dedicated amazon forecast function
        //downloadFile(afResultsS3Key),
        //downloadFile(afForecastS3Key)
      ]);

      // TODO: move to dedicated amazon forecast function
      return await parseReportData(input, s3Requests[0], s3Requests[1], s3Requests[2], null, null, null);
      //return await parseReportData(input, s3Requests[0], s3Requests[1], s3Requests[2], s3Requests[3], s3Requests[4]);
    };

  /*
   *      _                                   
   *     / \   _ __ ___   __ _ _______  _ __  
   *    / _ \ | '_ ` _ \ / _` |_  / _ \| '_ \ 
   *   / ___ \| | | | | | (_| |/ / (_) | | | |
   *  /_/   \_\_| |_| |_|\__,_/___\___/|_| |_|
   *   _____                            _     
   *  |  ___|__  _ __ ___  ___ __ _ ___| |_   
   *  | |_ / _ \| '__/ _ \/ __/ _` / __| __|  
   *  |  _| (_) | | |  __/ (_| (_| \__ \ |_   
   *  |_|  \___/|_|  \___|\___\__,_|___/\__|  
   */
  public loadAmazonForecastJobResults = async (input: ForecastJobInfo): Promise<ForecastJobResults> => {
    console.log("Loading Amazon Forecast job...");

    const s3ErrorHandler = (e) => {
      // Ignore 404 s3 errors
      if (e?.response?.status === 404) {
        return undefined;
      }
      console.log(`Unexpected error returned from S3:`, e);
      return undefined;
    };

    const credentials = await Auth.currentUserCredentials();
    const identityId = credentials.identityId;

    const removePrivatePrefix = (key) => key.replace(`private/${identityId}/`, "");
    const downloadFile = (key) =>
      Storage.get(removePrivatePrefix(key), { download: true, level: "private" }).catch(s3ErrorHandler);

    console.log("loadAmazonForecastJobResults:", downloadFile) ;
    console.log("---") ;

    const afResultsS3Key = 
      input.resultsS3Key.replace(".csv.results.json",
                                 ".input.csv.results.json.AF") ;
    const afForecastS3Key =
      input.forecastS3Key.replace(".csv.forecast.csv", ".af-resampled.csv") ;

    // Forecast job has been submitted if this file exists.
    const afParamsS3Key = 
      input.forecastS3Key.replace(".csv.forecast.csv",
                                  ".af-params.json") ;

    const s3Requests = await Promise.all([
      downloadFile(afResultsS3Key),
      downloadFile(afForecastS3Key),
      downloadFile(afParamsS3Key)
    ]);

    return await parseReportData(input, null, null, null, s3Requests[0], s3Requests[1], s3Requests[2]);
  };

  //
  //
  //
  public loadUserJobs = async (): Promise<ForecastJobInfo[]> => {
    const response = await API.get("voyagerGetForecastsApi", "/forecastList/", {});

    return response.map(this.normaliseForecastJob);
  };

  public loadSingleJob = async (id: string): Promise<ForecastJobInfo | null> => {
    const response = await API.get("voyagerGetForecastsApi", `/forecastList/?id=${id}`, {});

    if (response.length === 0) {
      return null;
    }

    return this.normaliseForecastJob(response[0]);
  };

  private normaliseForecastJob = (item: any): ForecastJobInfo => {
    return {
      ...item,
      name: "Report",
      createdAt: DateTime.fromISO(item.createdAt).toJSDate(),
    };
  };
}
