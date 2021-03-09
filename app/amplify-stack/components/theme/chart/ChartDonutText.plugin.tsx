// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import ChartJS from "chart.js";

// Plugin to render central or lower text on the chart if supplied

ChartJS.pluginService.register({
  beforeDraw: function (chart) {
    const width = chart.chartArea.right - chart.chartArea.left;
    const height = chart.chartArea.bottom - chart.chartArea.top;
    const ctx = chart.ctx;

    const centralTexts = (chart.config.options as any).centralText;

    for (const centralText of centralTexts || []) {
      ctx.restore();
      ctx.font = `${centralText.fontSize}px normal ${centralText.fontWeight || "normal"} ${centralText.fontFamily}`;
      ctx.textBaseline = "top";

      const textX = Math.round(
        width * (centralText.widthMultiplier || 0.5) - ctx.measureText(centralText.text).width / 2
      );
      const textY = height * centralText.heightMultiplier;

      ctx.fillText(centralText.text, textX, textY);
      ctx.save();
    }
  },
});
