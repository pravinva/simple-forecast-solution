// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Panel1.module.scss";
import { Panel } from "../../../../common/panel/Panel";
import Skeleton from "@material-ui/lab/Skeleton";
import { Grid, Tooltip, Box } from "@material-ui/core";
import { Bar, Doughnut } from "react-chartjs-2";
import { ChartTheme } from "../../../../theme/chart/ChartTheme";
import { ChartOptions } from "chart.js";
import Help from "@material-ui/icons/Help";
import { InfoTile } from "../../../../common/info-tile/InfoTile";
import { DataAnalysisResults } from "../../../../../services/forecasting";
import pluralize from "pluralize";

interface Props {
  data: DataAnalysisResults;
}

export const Panel1: React.FC<Props> = ({ data }) => {
  return (
    <div className={styles.container}>
      <Panel className={styles.panel} subTitle="Data Analysis">
        {data ? <Content data={data} /> : <Placeholder />}
      </Panel>
    </div>
  );
};

const Placeholder = () => {
  return (
    <div>
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={60} height={60} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={60} height={60} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={60} height={60} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={60} height={60} />
      <Skeleton variant="text" height={40} />
    </div>
  );
};

const Content: React.FC<Props> = ({ data }) => {
  const seasonality = data.seasonalityIndex;
  const seasonalityCategory = getSpectralEntropyCategory(seasonality);

  const historicalInterval = getInterval(data.historicalData.startDate, data.historicalData.endDate);

  return (
    <>
      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <InfoTile.Container>
            <InfoTile.Title>SKUs</InfoTile.Title>
            <InfoTile.Statistic>{data.skus}</InfoTile.Statistic>
          </InfoTile.Container>
          <InfoTile.Container>
            <InfoTile.Title>Frequency</InfoTile.Title>
            <InfoTile.Statistic>{formatInterval(data.frequency)}</InfoTile.Statistic>
          </InfoTile.Container>
          <InfoTile.Container>
            <InfoTile.Title>Channels</InfoTile.Title>
            <InfoTile.Statistic>{data.channels}</InfoTile.Statistic>
          </InfoTile.Container>
          <InfoTile.Container>
            <InfoTile.Title>Families</InfoTile.Title>
            <InfoTile.Statistic>{data.families}</InfoTile.Statistic>
          </InfoTile.Container>
          <InfoTile.Container>
            <InfoTile.Title>Forecast Horizon</InfoTile.Title>
            <InfoTile.Statistic>
              {data.horizon.amount} {pluralize(formatIntervalUnit(data.horizon.unit), data.horizon.amount)}
            </InfoTile.Statistic>
          </InfoTile.Container>
          {historicalInterval && (
            <>
              <InfoTile.Container mode="column">
                <InfoTile.Title>Historical Data</InfoTile.Title>
                <Grid container>
                  <Grid item xs={4}>
                    <InfoTile.Subtitle>Years</InfoTile.Subtitle>
                    <InfoTile.Statistic>{historicalInterval.years}</InfoTile.Statistic>
                  </Grid>
                  <Grid item xs={4}>
                    <InfoTile.Subtitle>Months</InfoTile.Subtitle>
                    <InfoTile.Statistic>{historicalInterval.months}</InfoTile.Statistic>
                  </Grid>
                  <Grid item xs={4}>
                    <InfoTile.Subtitle>Days</InfoTile.Subtitle>
                    <InfoTile.Statistic>{historicalInterval.days}</InfoTile.Statistic>
                  </Grid>
                </Grid>
              </InfoTile.Container>
              <InfoTile.Container mode="column">
                <Grid container>
                  <Grid item xs={6}>
                    <InfoTile.Subtitle>Start</InfoTile.Subtitle>
                    <InfoTile.Statistic>{data.historicalData.startDate.toLocaleDateString()}</InfoTile.Statistic>
                  </Grid>
                  <Grid item xs={6}>
                    <InfoTile.Subtitle>End</InfoTile.Subtitle>
                    <InfoTile.Statistic>{data.historicalData.endDate.toLocaleDateString()}</InfoTile.Statistic>
                  </Grid>
                </Grid>
              </InfoTile.Container>
            </>
          )}
        </Grid>
        <Grid item xs={12} md={7}>
          <Box height="100%" display="flex" flexDirection="column" justifyContent="space-between">
            <div className={styles.frequency}>
              <h5>
                Demand Classification (All SKUs)
                <Tooltip
                  title={
                    <>
                      All SKUs broken down by their sales behaviour over the provided historical sales period. <br />
                      Continuous = SKUs with over 1 year of sales and additional sales observed in the last 8-weeks
                      <br />
                      Medium = SKUs with over 1 year of sales Short = SKUs with less than 1 year of sales
                    </>
                  }
                  placement="top"
                  arrow
                >
                  <Help />
                </Tooltip>
              </h5>
              <Bar
                height={200}
                data={{
                  labels: ["", "", "", "", "", "", "", "", "", "", ""],
                  datasets: [
                    {
                      label: "% Continuous",
                      backgroundColor: ChartTheme.colors.yellow,
                      barThickness: 20,
                      barPercentage: 1,
                      data: [0, 0, 0, 0, Math.round(data.demandCategories.continuous * 100)],
                    },
                    {
                      label: "% Medium",
                      backgroundColor: ChartTheme.colors.aqua,
                      barThickness: 20,
                      data: [0, 0, 0, 0, 0, Math.round(data.demandCategories.medium * 100)],
                    },
                    {
                      label: "% Short",
                      backgroundColor: ChartTheme.colors.voilet,
                      barThickness: 20,
                      data: [0, 0, 0, 0, 0, 0, Math.round(data.demandCategories.short * 100)],
                    },
                  ],
                }}
                options={
                  {
                    legend: {
                      display: true,
                      position: "bottom",
                      labels: {
                        ...ChartTheme.legendLabelOptions,
                      },
                    },
                    cornerRadius: 20,
                    scales: {
                      yAxes: [
                        {
                          ticks: {
                            suggestedMax: 100,
                            suggestedMin: 0,
                            stepSize: 25,
                            ...ChartTheme.fontOptions,
                            fontColor: ChartTheme.colors.lightGrey,
                          },
                          gridLines: ChartTheme.gridLines,
                        },
                      ],
                      xAxes: [
                        {
                          gridLines: ChartTheme.gridLines,
                        },
                      ],
                    },
                  } as ChartOptions & { cornerRadius: number }
                }
              />
            </div>
            <Grid container>
              <Grid className={styles.donut} item xs={12}>
                <h5>
                  Spectral Entropy
                  <Tooltip
                    title={
                      <>
                        Seasonality is determined based on recognisable patterns in your sales data. This is measured
                        using spectral entropy - i.e. how 'noisy' your data is. The higher the 'noise', the lower the
                        seasonality or identifiable 'signals'. This metric is used to help identify which forecast
                        models will be optimal for your business.
                      </>
                    }
                    placement="top"
                    arrow
                  >
                    <Help />
                  </Tooltip>
                </h5>
                <Doughnut
                  height={80}
                  data={{
                    labels: ["", ""],
                    datasets: [
                      {
                        backgroundColor: [seasonalityCategory.colour, ChartTheme.colors.veryLightGrey],
                        borderColor: "#fff",
                        data: [
                          Number(Math.max(0.5, Math.max(0, 6 - seasonality)).toFixed(2)),
                          Number(Math.min(5.5, seasonality).toFixed(2)),
                        ],
                      },
                    ],
                  }}
                  options={
                    {
                      legend: { display: false },
                      layout: { padding: { bottom: 20 } },
                      tooltips: { enabled: false },
                      circumference: 1.5 * Math.PI,
                      rotation: 0.75 * Math.PI,
                      cutoutPercentage: 75,
                      centralText: [
                        {
                          ...ChartTheme.fontOptions,
                          fontSize: 20,
                          fontWeight: "light",
                          text: seasonality.toFixed(1),
                          heightMultiplier: 0.47,
                        },
                        {
                          ...ChartTheme.fontOptions,
                          fontSize: 12,
                          fontWeight: "light",
                          text: seasonalityCategory.label.toUpperCase(),
                          heightMultiplier: 1,
                        },
                      ],
                      elements: {
                        arc: {
                          borderWidth: 0,
                        },
                      },
                    } as ChartOptions & { centralText: any }
                  }
                />
              </Grid>
              {typeof data.testingPeriodsCompleted !== "undefined" && (
                <Grid className={styles.testingPeriods} item xs={12}>
                  <p># Testing Periods Completed {data.testingPeriodsCompleted.toFixed(0)}</p>
                </Grid>
              )}
            </Grid>
          </Box>
        </Grid>
      </Grid>
    </>
  );
};

const getSpectralEntropyCategory = (value: number): { colour: string; label: string } => {
  if (value >= 5) {
    return { colour: ChartTheme.colors.red, label: "Poor" };
  } else if (value >= 4) {
    return { colour: ChartTheme.colors.yellow, label: "Average" };
  } else if (value >= 2) {
    return { colour: ChartTheme.colors.voilet, label: "Good" };
  } else {
    return { colour: ChartTheme.colors.green, label: "Excellent" };
  }
};

const formatPercentage = (value: number): string => {
  return `${(value * 100).toFixed(1)}%`;
};

const formatInterval = (value: "D" | "W" | "M"): string => {
  switch (value) {
    case "D":
      return "Daily";

    case "W":
      return "Weekly";

    case "M":
      return "Monthly";

    default:
      return "Unknown";
  }
};

const formatIntervalUnit = (value: "D" | "W" | "M"): string => {
  switch (value) {
    case "D":
      return "Day";

    case "W":
      return "Week";

    case "M":
      return "Month";

    default:
      return "Unknown";
  }
};

const getInterval = (
  start: Date | undefined,
  end: Date | undefined
): { years: number; months: number; days: number } | undefined => {
  if (!start || !end) {
    return undefined;
  }

  let diff = end.getTime() - start.getTime();

  // Naive date difference (can be improved)
  const years = Math.floor(diff / (365 * 86400 * 1000));
  diff -= years * 365 * 86400 * 1000;
  const months = Math.floor(diff / (30 * 86400 * 1000));
  diff -= months * 30 * 86400 * 1000;
  const days = Math.floor(diff / (86400 * 1000));

  return { years, months, days };
};
