// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { GridLineOptions } from "chart.js";
import "./ChartBarRoundedCorners.plugin.tsx";
import "./ChartDonutText.plugin.tsx";
import "chartjs-adapter-luxon";

const gridLines: GridLineOptions = {
  borderDash: [2, 2],
  color: "rgba(128, 151, 177, 0.35)",
  zeroLineColor: "rgba(255, 255, 255, 0.25)",
};

const colors = {
  yellow: "#FFAF20",
  aqua: "#00C1D4",
  voilet: "#8C54FF",
  green: "#43CE8F",
  lightGrey: "#B0BAC9",
  veryLightGrey: "#f2f2f2",
  red: "#de5e6d",
};

const fontOptions = {
  fontColor: "#8798AD",
  fontSize: 16,
  fontFamily: "Rubik, sans-serif",
};

const legendLabelOptions = {
  ...fontOptions,
  usePointStyle: true,
  boxWidth: 10,
};

export const ChartTheme = {
  gridLines,
  colors,
  fontOptions,
  legendLabelOptions,
};
