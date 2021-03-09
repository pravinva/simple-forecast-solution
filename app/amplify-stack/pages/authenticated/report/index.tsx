// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import Head from "next/head";
import { Grid, Box } from "@material-ui/core";
import { Sidebar } from "../../../components/authenticated/sidebar/Sidebar";
import { Header } from "../../../components/authenticated/header/Header";
import { Report as ReportDetails } from "../../../components/authenticated/report/Report";
import { ForecastJobInfo, ForecastingService } from "../../../services/forecasting";
import { useRouter } from "next/router";
import { useState, useEffect } from "react";

const forecastingService = new ForecastingService();

const Report = () => {
  const router = useRouter();

  const [report, setReport] = useState<ForecastJobInfo>();

  useEffect(() => {
    setReport(undefined);
    loadReport();
  }, [router.query.id]);

  const loadReport = async () => {
    const reportId = router.query.id;

    if (!reportId) {
      router.replace("/authenticated");
      return;
    }

    console.log(`Loading report ${reportId}`);
    const report = await forecastingService.loadSingleJob(reportId as string);

    if (!report) {
      router.replace("/authenticated");
      return;
    }

    setReport(report);
  };

  return (
    <main>
      <Head>
        <title>{report?.name || "Report"} - Amazon SFS</title>
      </Head>
      <Grid container>
        <Grid item xs={12} md={3} lg={2}>
          <Sidebar />
        </Grid>
        <Grid item xs={12} md={9} lg={10}>
          <Box height="100%" display="flex" flexDirection="column">
            <Header>{report && `${report.name} - ${report.createdAt.toLocaleDateString()}`}</Header>
            <ReportDetails report={report} />
          </Box>
        </Grid>
      </Grid>
    </main>
  );
};

export default Report;
