// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Storage from "@aws-amplify/storage";
import * as uuid from "uuid";
import Auth, { CognitoUser } from "@aws-amplify/auth";
import API from "@aws-amplify/api";
import _ from "lodash";
import { DateTime } from "luxon";

export interface UserAccountInfoResponse {
  totalReportsAllowed: number;
  totalReportsConsumed: number;
  trialExpiresAt?: Date;
  permissions: {
    canCreateNewReport: boolean;
  };
}

export enum UserRole {
  Standard,
  Admin,
}

export class UserService {
  public getUserRole = async (): Promise<UserRole> => {
    const session = await Auth.currentSession();

    const groups = session.getIdToken().payload["cognito:groups"] || [];
    const isAdmin = groups.includes("Admins");

    return isAdmin ? UserRole.Admin : UserRole.Standard;
  };

  public getUserAccontInfo = async (): Promise<UserAccountInfoResponse> => {
    const response = await API.get("voyagerGetAccountUsageApi", "/accountUsage/", {});

    return {
      totalReportsAllowed: response.totalReportsAllowed || 10,
      totalReportsConsumed: response.totalReportsConsumed || 0,
      trialExpiresAt: response.trialExpiresAt ? DateTime.fromISO(response.trialExpiresAt).toJSDate() : undefined,
      permissions: {
        canCreateNewReport: response?.permissions?.canCreateNewReport || true,
      },
    };
  };
}
