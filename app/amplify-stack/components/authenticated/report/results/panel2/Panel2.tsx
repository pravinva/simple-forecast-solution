// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
// vim: set ts=2 sw=2:

import { Box, Tooltip } from "@material-ui/core";
import Help from "@material-ui/icons/Help";
import Skeleton from "@material-ui/lab/Skeleton";
import { ChartOptions } from "chart.js";
import classNames from "classnames";
import _ from "lodash";
import React from "react";
import { Doughnut } from "react-chartjs-2";
import { AfModelResults, ModelResults } from "../../../../../services/forecasting";
import { InfoTile } from "../../../../common/info-tile/InfoTile";
import { Panel } from "../../../../common/panel/Panel";
import { ChartTheme } from "../../../../theme/chart/ChartTheme";
import styles from "./Panel2.module.scss";

interface Props {
  data: ModelResults;
}

interface PropsAf {
  data: AfModelResults;
}

export const Panel2: React.FC<Props> = ({ data }) => {
  return (
    <div className={styles.container}>
      <Panel className={styles.panel} subTitle="Performance of the Amazon SFS Engine">
        {data ? <Content data={data} /> : <Placeholder />}
      </Panel>
    </div>
  );
};

export const Panel2Af: React.FC<PropsAf> = ({ data }) => {
  return (
    <div className={styles.container}>
      <Panel className={styles.panel} subTitle="Performance of Amazon Forecast">
        { data ? <ContentAf data={data} /> : <PlaceholderAf /> }
      </Panel>
    </div>
  );
};

const Placeholder = () => {
  return (
    <>
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={40} height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={40} height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={40} height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="circle" width={40} height={40} />
      <Skeleton variant="text" height={40} />
    </>
  );
};

const PlaceholderAf = () => {
  return (
    <>
      <Skeleton variant="text" height={40} />
      <Skeleton variant="text" height={40} />
    </>
  );
};

const ContentAf: React.FC<PropsAf> = ({ data }) => {
  return (
    <div>
        <p><b>Best performing model:</b> {data.bestPerformingModel.split("/")[1]}</p>
      <p><b>Forecast Accuracy:</b> { ((1.0 - Math.min(1.0, data.mape)) * 100.0).toFixed(2) } %</p>
    </div>
  );
};

const Content: React.FC<Props> = ({ data }) => {
  const bestPerformingModels = formatModelLabels(data.bestPerformingModels);

  return (
    <Box>
      <div className={styles.stats}>
        <InfoTile.Container>
          <InfoTile.Title light>
            Forecast <br /> Accuracy
          </InfoTile.Title>
          <InfoTile.Statistic>
            <span
              className={classNames(
                styles.accuracy,
                data.accuracyIncrease >= 0 && styles["accuracy--increase"],
                data.accuracyIncrease < 0 && styles["accuracy--decrease"]
              )}
            >
              <InfoTile.Statistic>
                {Math.abs((1.0 - data.avgVoyagerError) * 100.0).toFixed(0)}%
                <span className={styles.footnote}>
                  Performance {data.accuracyIncrease >= 0 ? "Increase" : "Decrease"}{" "}
                  by  {Math.abs(data.accuracyIncrease).toFixed(1)}%
                  <Tooltip
                    title={
                      <>
                        The industry standard practice is to compare a new forecast (e.g. via Amazon SFS) to a naive
                        forecast (using previous 8-week average sales). <br /> The resulting improvement is measured by
                        using MAPE (mean absolute percentage error) which measures how close the forecast was to the
                        actual value. The Amazon SFS MAPE is then compared to the naive model MAPE to determine the
                        improvement in forecast accuracy.
                      </>
                    }
                    placement="bottom"
                  >
                    <Help />
                  </Tooltip>
                </span>
              </InfoTile.Statistic>
            </span>
          </InfoTile.Statistic>
        </InfoTile.Container>
      </div>
      <div className={styles.charts}>
        <h4>Best performing models</h4>

        <div className={styles.chart}>
          <Doughnut
            height={300}
            data={{
              labels: bestPerformingModels.map((i) => i.label),
              datasets: [
                {
                  backgroundColor: bestPerformingModels.map((i) => i.colour),
                  borderColor: "#fff",
                  data: bestPerformingModels.map((i) => Number((i.value * 100).toFixed(0))),
                },
              ],
            }}
            options={
              {
                legend: {
                  position: "bottom",
                  labels: {
                    ...ChartTheme.legendLabelOptions,
                    fontColor: "#333",
                    generateLabels: () => {
                      return bestPerformingModels
                        .slice(0, 3)
                        .map((i, k) => ({
                          text: i.label,
                          fillStyle: i.colour,
                          strokeStyle: "transparent",
                          datasetIndex: k,
                        }))
                        .concat([
                          {
                            text: "Other",
                            fillStyle: "rgba(0,0,0,0.3)",
                            strokeStyle: "transparent",
                            datasetIndex: undefined,
                          },
                        ]);
                    },
                  },
                },
                cutoutPercentage: 80,
                centralText: [
                  {
                    ...ChartTheme.fontOptions,
                    fontSize: 18,
                    fontWeight: "light",
                    text: "% SKUs",
                    heightMultiplier: 0.47,
                  },
                ],
                elements: {
                  arc: {
                    borderWidth: 0,
                  },
                },
              } as ChartOptions
            }
          />
        </div>
      </div>
    </Box>
  );
};

const formatModelLabels = (
  data: ModelResults["bestPerformingModels"]
): { label: string; value: number; colour: string }[] => {
  const nameMap = {
    naive_model: "Naive",
    trend_model: "Trend",
    es_model: "ES",
    holt_model: "Holt",
    arima_model: "ARIMA",
    fourier_model: "Fourier",
  };

  const suffixMap = {
    local_: "Local",
    _seasonal: "Seasonal",
  };

  const colours = getSegmentColours();

  return _.sortBy(_.toPairs(data), ([model, percentage]) => -percentage) //
    .map(([model, percentage], i) => {
      const name = _.find(nameMap, (v, k) => model.includes(k));
      const suffixes = _.filter(suffixMap, (v, k) => model.includes(k));
      const suffix = suffixes.length ? ` (${suffixes.join(", ")})` : ``;

      return {
        label: `${(percentage * 100).toFixed(0)}% ${name}${suffix}`,
        value: percentage,
        colour: colours[i] || "#fff",
      };
    });
};

const getSegmentColours = (): string[] => {
  const colours = [ChartTheme.colors.yellow, ChartTheme.colors.aqua, ChartTheme.colors.green];

  for (const i of _.range(15, 1)) {
    colours.push(`rgba(0, 0, 0, ${i / 40})`);
  }

  return colours;
};
