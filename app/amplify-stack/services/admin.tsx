// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import _ from "lodash";
import API from "@aws-amplify/api";
import { DateTime } from "luxon";
import qs from "querystring";

export interface CustomerInfoRequest {
  limit?: number;
  nextToken?: string;
}

export interface CustomerInfo {
  name: string;
  email: string;
  noOfReports: number;
  lastReportAt?: Date;
  registeredAt?: Date;
}

export interface CustomerInfoResponse {
  customers: CustomerInfo[];
  nextToken?: string;
}

export class AdminService {
  public getCustomerInfo = async (request: CustomerInfoRequest): Promise<CustomerInfoResponse> => {
    const response = await API.get(
      "voyagerGetCustomerLevelDataApi",
      `/customerLevelData?${qs.stringify({
        paginationToken: request.nextToken,
        limit: request.limit,
      })}`,
      {}
    );

    return {
      customers: response.users.map((i) => ({
        name: i.name,
        email: i.email,
        noOfReports: i.numReports,
        lastReportAt: i.lastReportCreatedAt && DateTime.fromISO(i.lastReportCreatedAt).toJSDate(),
        registeredAt: i.userCreatedAt && DateTime.fromISO(i.userCreatedAt).toJSDate(),
      })),
      nextToken: response.paginationToken,
    };
  };
}
