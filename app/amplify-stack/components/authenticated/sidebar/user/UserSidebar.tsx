// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useState, useEffect } from "react";
import styles from "./UserSidebar.module.scss";
import { ForecastJobInfo, ForecastingService } from "../../../../services/forecasting";
import { UserService, UserAccountInfoResponse } from "../../../../services/user";
import { ReportsList } from "../reports-list/ReportsList";
import { Intro } from "../intro/Intro";

const forecastingService = new ForecastingService();
const userService = new UserService();

export const UserSidebar: React.FC = () => {
  const [reports, setReports] = useState<ForecastJobInfo[]>();
  const [accountInfo, setAccountInfo] = useState<UserAccountInfoResponse>();

  useEffect(() => {
    loadJobs();
  }, []);

  const loadJobs = async () => {
    await Promise.all([
      forecastingService.loadUserJobs().then(setReports),
      userService.getUserAccontInfo().then(setAccountInfo),
    ]);
  };

  return (
    <div className={styles.body}>
      {reports?.length === 0 && <Intro />}

      {accountInfo && reports?.length > 0 && (
        <div className={styles.trialInfo}>
          {/* {accountInfo.trialExpiresAt && (
            <p>
              Trial expires in:{" "}
              {Math.max(0, (new Date().getTime() - accountInfo.trialExpiresAt.getTime()) / 1000 / 86400).toFixed(0)}{" "}
              days
            </p>
          )} */}
        </div>
      )}

      {reports?.length > 0 && <ReportsList reports={reports} accountInfo={accountInfo} />}
    </div>
  );
};
