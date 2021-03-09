// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./ReportsList.module.scss";
import { Link } from "../../../common/link/Link";
import { Button } from "../../../common/button/Button";
import { ForecastJobInfo } from "../../../../services/forecasting";
import { UserAccountInfoResponse } from "../../../../services/user";
import { NavList } from "../nav-list/NavList";
import _ from "lodash";
import { NavListItem } from "../nav-list/item/NavListItem";
import { useRouter } from "next/router";

interface ReportsListProps {
  accountInfo?: UserAccountInfoResponse;
  reports: ForecastJobInfo[];
}

export const ReportsList: React.FC<ReportsListProps> = ({ reports, accountInfo }) => {
  reports = _.sortBy(reports, (i) => -i.createdAt.getTime());
  const router = useRouter();

  const getReportUrl = (i: ForecastJobInfo) => `/authenticated/report?id=${encodeURIComponent(i.id)}`;

  return (
    <div className={styles.reportsList}>
      <Link className={styles.createReport} href="/authenticated/upload">
      <Button variant="contained" color="primary">
          Create New Report
        </Button>
      </Link>

      {reports && (
        <NavList>
          {reports.map((i) => (
            <NavListItem key={i.id} active={router.asPath === getReportUrl(i)}>
              <Link href={getReportUrl(i)}>
                {i.name} - {i.createdAt.toLocaleDateString()}
              </Link>
            </NavListItem>
          ))}
        </NavList>
      )}
    </div>
  );
};
