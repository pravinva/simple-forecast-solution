// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import { FormControl, InputLabel, MenuItem, Select, Tabs, TextField } from "@material-ui/core";
import Autocomplete from "@material-ui/lab/Autocomplete";
import Skeleton from "@material-ui/lab/Skeleton";
import _ from "lodash";
import { DateTime } from "luxon";
import React, { useEffect, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import { ForecastResults } from "../../../../../services/forecasting";
import { Button } from "../../../../common/button/Button";
import { Panel } from "../../../../common/panel/Panel";
import Table from "../../../../common/table/Table";
import { ChartTheme } from "../../../../theme/chart/ChartTheme";
import styles from "./Panel3.module.scss";

interface Props {
  data: ForecastResults;
}

enum FilterType {
  SKU,
  CHANNEL,
  FAMILY,
  NONE,
}

interface FilterOption {
  type: FilterType;
  value: string;
}

enum DateRange {
  THREE_MONTHS,
  SIX_MONTHS,
  ONE_YEAR,
  TWO_YEARS,
  ALL,
}

const DATE_FORMATS = {
  D: "dd LLL yyyy",
  W: "dd LLL yyyy",
  M: "LLL yyyy",
};

interface ChartData {
  xDate: DateTime;
  yActual: number | undefined;
  yPred: number | undefined;
}

interface TableData {
  columns: { actuals: DateTime[]; forecasts: DateTime[] };
  rows: { sku: string; demandCategory: string; actuals: number[]; forecasts: number[] }[];
}

export const Panel3: React.FC<Props> = ({ data }) => {
  return (
    <div className={styles.container}>
      <Panel subTitle="Forecast by the Amazon SFS Engine">{data ? <Content data={data} /> : <Placeholder />}</Panel>
    </div>
  );
};

export const Panel3Af: React.FC<Props> = ({ data }) => {
  return (
    <div className={styles.container}>
      <Panel subTitle="Forecast by Amazon Forecast">{data ? <Content data={data} /> : <Placeholder />}</Panel>
    </div>
  );
};

const Placeholder = () => {
  return (
    <>
      <Skeleton variant="text" height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="text" height={40} />
      <Skeleton variant="text" height={40} />
    </>
  );
};

const Content: React.FC<Props> = ({ data }) => {
  const [tabIndex, setTabIndex] = useState(0);
  const [filters, setFilters] = useState<FilterOption[]>([]);
  const [filterFocused, setFilterFocused] = useState(false);
  const [dateRange, setDateRange] = useState(DateRange.THREE_MONTHS);

  const filterOptions = getFilterOptions(data.data);

  const filteredData = filterAndRollupData(data.data, filters, dateRange);
  const actualEndDate = getActualEndDate(data.data);
  const oneMonthAgo = actualEndDate.minus({ months: 1 });

  const tableData = filterTableData(data.data, filters);
  const tableRef = useRef<HTMLDivElement>();

  const dateFormat = DATE_FORMATS[data.frequency];

  const handleSelectFilter = (options: FilterOption[]) => {
    if (options[options.length - 1]?.type === FilterType.NONE) {
      setFilters([]);
    } else {
      setFilters(options.filter((i) => i.type !== FilterType.NONE));
    }
  };

  const scrollTableToDate = (date: Date) => {
    if (!tableRef.current) {
      return;
    }

    const column = tableRef.current!.querySelector(`thead th[data-ts="${date.getTime()}"]`);

    if (!column) {
      console.warn(`Could not find column with date`, date);
      return;
    }

    tableRef.current.scrollLeft =
      (column as any).offsetLeft + (column as any).offsetWidth - tableRef.current.offsetWidth / 2;
  };

  useEffect(() => {
    if (!tableRef.current) {
      return;
    }

    scrollTableToDate(getActualEndDate(data.data).toJSDate());
  }, [tableRef.current, tableData]);

  return (
    <div>
      <div className={styles.tabs}>
        <Tabs value={tabIndex} indicatorColor="primary" textColor="primary" onChange={(e, i) => setTabIndex(i)}>
          {/* <Tab label="Forecast" /> */}
          {/* <Tab label="Models" /> */}
        </Tabs>
      </div>
      {tabIndex === 0 && (
        <div className={styles.forecast}>
          <div className={styles.controls}>
            <div className={styles.filters}>
              <Autocomplete
                multiple
                filterSelectedOptions
                options={filterOptions}
                getOptionLabel={getFilterLabel}
                value={filterFocused || filters.length ? filters : [{ type: FilterType.NONE, value: "" }]}
                getOptionSelected={(a, b) => a.type === b.type && a.value === b.value}
                onChange={(evt, options) => handleSelectFilter(options as FilterOption[])}
                onFocus={() => setFilterFocused(true)}
                onBlur={() => setFilterFocused(false)}
                renderInput={(params) => <TextField {...params} label="SKU / Channel / Family" variant="outlined" />}
              />
              <FormControl variant="outlined">
                <InputLabel>Dates</InputLabel>
                <Select
                  value={dateRange}
                  onChange={(evt) => setDateRange(evt.target.value as DateRange)}
                  required
                  label="Unit"
                >
                  <MenuItem value={DateRange.THREE_MONTHS}>3 months</MenuItem>
                  <MenuItem value={DateRange.SIX_MONTHS}>6 months</MenuItem>
                  <MenuItem value={DateRange.ONE_YEAR}>1 year</MenuItem>
                  <MenuItem value={DateRange.TWO_YEARS}>2 years</MenuItem>
                  <MenuItem value={DateRange.ALL}>All</MenuItem>
                </Select>
              </FormControl>
            </div>
            <div>
              <a className={styles.export} href={data.signedUrl} download>
                <Button variant="contained" color="secondary">
                  Export
                </Button>
              </a>
            </div>
          </div>
          <div className={styles.chart}>
            <Line
              data={{
                datasets: [
                  {
                    label: "Predicted",
                    borderColor: ChartTheme.colors.green,
                    pointBackgroundColor: "#fff",
                    pointRadius: 3,
                    pointBorderWidth: 2,
                    pointBorderColor: ChartTheme.colors.green,
                    backgroundColor: "rgba(67, 207, 143, 25%)",
                    data: filteredData
                      .filter((i) => typeof i.yPred !== "undefined" && i.xDate <= actualEndDate)
                      .map((i) => ({ x: i.xDate, y: i.yPred })) as any,
                    lineTension: 0,
                  },
                  {
                    label: "Forecast",
                    borderColor: ChartTheme.colors.green,
                    pointBackgroundColor: "#fff",
                    pointRadius: 3,
                    pointBorderWidth: 2,
                    pointBorderColor: ChartTheme.colors.green,
                    borderDash: [5, 5],
                    backgroundColor: "rgba(0,0,0,0)",
                    data: filteredData
                      .filter((i) => typeof i.yPred !== "undefined" && i.xDate >= actualEndDate)
                      .map((i) => ({ x: i.xDate, y: Math.round(i.yPred) })) as any,
                    lineTension: 0,
                  },
                  {
                    label: "Actual",
                    borderColor: ChartTheme.colors.voilet,
                    pointBackgroundColor: "#fff",
                    pointRadius: 3,
                    pointBorderWidth: 2,
                    pointBorderColor: ChartTheme.colors.voilet,
                    backgroundColor: "rgba(140, 84, 255, 25%)",
                    data: filteredData
                      .filter((i) => typeof i.yActual !== "undefined")
                      .map((i) => ({ x: i.xDate, y: i.yActual })) as any,
                    lineTension: 0,
                  },
                ],
              }}
              options={{
                legend: {
                  labels: {
                    ...ChartTheme.legendLabelOptions,
                    generateLabels: () => [
                      {
                        pointStyle: "circle",
                        fillStyle: ChartTheme.colors.green,
                        strokeStyle: "transparent",
                        text: "Forecast",
                      },
                      {
                        pointStyle: "circle",
                        fillStyle: ChartTheme.colors.voilet,
                        strokeStyle: "transparent",
                        text: "Actual",
                      },
                    ],
                  },
                  position: "top",
                  align: "end",
                },
                tooltips: {
                  // Hide overlapping point between actual & forecast dataset
                  filter: (i) => i.xLabel !== actualEndDate.toFormat(dateFormat) || i.datasetIndex !== 1,
                },
                scales: {
                  xAxes: [
                    {
                      type: "time",
                      time: {
                        unit: "month",
                        tooltipFormat: dateFormat,
                      },
                      ticks: { ...ChartTheme.fontOptions },
                      gridLines: ChartTheme.gridLines,
                    },
                  ],
                  yAxes: [
                    {
                      scaleLabel: {
                        ...ChartTheme.fontOptions,
                        display: true,
                        labelString: "Quantity",
                      },
                      ticks: {
                        ...ChartTheme.fontOptions,
                        min: 0,
                        maxTicksLimit: 5,
                        callback: (value, index, values) => {
                          return value
                            .toString()
                            .split(/(?=(?:...)*$)/)
                            .join(",");
                        },
                      },
                      gridLines: ChartTheme.gridLines,
                    },
                  ],
                },
                maintainAspectRatio: false,
              }}
            />
          </div>
          <MemoizedForecastTable
            ref={tableRef}
            dateFormat={dateFormat}
            cacheKey={JSON.stringify(filters)}
            data={tableData}
          />
        </div>
      )}
    </div>
  );
};

const ForecastTable: React.FC<{ ref: any; dateFormat: string; cacheKey: string; data: TableData }> = React.forwardRef(
  ({ data, dateFormat }, ref) => {
    return (
      <div ref={ref as any} className={styles.table}>
        <Table.Container stickyHeader>
          <Table.Head>
            <Table.Row>
              <Table.Cell>SKU</Table.Cell>
              {data.columns.actuals.map((i, k) => (
                <Table.Cell className="actual" key={k} data-ts={i.toMillis()}>
                  {i.toFormat(dateFormat)}
                </Table.Cell>
              ))}
              {data.columns.forecasts.map((i, k) => (
                <Table.Cell className="forecast" key={k} data-ts={i.toMillis()}>
                  {i.toFormat(dateFormat)}
                </Table.Cell>
              ))}
            </Table.Row>
          </Table.Head>
          <Table.Body>
            {data.rows.map((row) => (
              <Table.Row key={row.sku}>
                <Table.Cell>
                  <span>{row.sku}</span>
                  <span className="demand">{row.demandCategory}</span>
                </Table.Cell>
                {row.actuals.map((i, k) => (
                  <Table.Cell className="actual" key={k}>
                    <span>{Math.round(i || 0)}</span>
                  </Table.Cell>
                ))}
                {row.forecasts.map((i, k) => (
                  <Table.Cell className="forecast" key={k}>
                    <span>{Math.round(i || 0)}</span>
                  </Table.Cell>
                ))}
              </Table.Row>
            ))}
          </Table.Body>
        </Table.Container>
      </div>
    );
  }
);

const MemoizedForecastTable = React.memo(ForecastTable, (a, b) => a.cacheKey === b.cacheKey);

const getFilterOptions = (data: ForecastResults["data"]): FilterOption[] => {
  const options = {};

  for (const item of data) {
    const skuOption = { type: FilterType.SKU, value: item.sku };
    const channelOption = { type: FilterType.CHANNEL, value: item.channel };
    const familyOption = { type: FilterType.FAMILY, value: item.family };
    options[getFilterKey(skuOption)] = skuOption;
    options[getFilterKey(channelOption)] = channelOption;
    options[getFilterKey(familyOption)] = familyOption;
  }

  return [{ type: FilterType.NONE, value: "" }].concat(_.sortBy(Object.values(options), getFilterKey));
};

const getFilterKey = (option: FilterOption): string => {
  return `${option.type}-${option.value}`;
};

const LABELS = {
  [FilterType.SKU]: "SKU",
  [FilterType.CHANNEL]: "Channel",
  [FilterType.FAMILY]: "Family",
};

const getFilterLabel = (option: FilterOption): string => {
  if (option.type === FilterType.NONE) {
    return "All";
  }

  return `${LABELS[option.type]}: ${option.value}`;
};

const getActualEndDate = (data: ForecastResults["data"]): DateTime => {
  const actualEndDate = DateTime.fromJSDate(
    _.max(data.filter((i) => typeof i.yActual !== "undefined").map((i) => i.x))
  );

  return actualEndDate;
};

const getForecastEndDate = (data: ForecastResults["data"]): DateTime => {
  const forecastEndDate = DateTime.fromJSDate(
    _.max(data.filter((i) => typeof i.yForecast !== "undefined").map((i) => i.x))
  );

  return forecastEndDate;
};

const getBacktestStartDate = (data: ForecastResults["data"]): DateTime => {
  const backtestStartDate = DateTime.fromJSDate(
    _.min(data.filter((i) => typeof i.yForecast !== "undefined").map((i) => i.x))
  );

  return backtestStartDate;
};

const getFilterStartDate = (data: ForecastResults["data"], range: DateRange): DateTime | undefined => {
  const actualEndDate = getActualEndDate(data);

  switch (range) {
    case DateRange.THREE_MONTHS:
      return actualEndDate.minus({ months: 3 });
    case DateRange.SIX_MONTHS:
      return actualEndDate.minus({ months: 6 });
    case DateRange.ONE_YEAR:
      return actualEndDate.minus({ months: 12 });
    case DateRange.TWO_YEARS:
      return actualEndDate.minus({ months: 24 });
    case DateRange.ALL:
      return undefined;
    default:
      throw new Error(`Unsupported date range: ${range}`);
  }
};

const selectItemsBasedOnFilters = (
  items: ForecastResults["data"],
  filters: FilterOption[]
): ForecastResults["data"] => {
  if (filters.length === 0) {
    return items;
  }

  const skus = filters.filter((i) => i.type === FilterType.SKU).map((i) => i.value);
  const channels = filters.filter((i) => i.type === FilterType.CHANNEL).map((i) => i.value);
  const families = filters.filter((i) => i.type === FilterType.FAMILY).map((i) => i.value);

  return items.filter(
    (i) =>
      (skus.length === 0 || skus.includes(i.sku)) &&
      (channels.length === 0 || channels.includes(i.channel)) &&
      (families.length === 0 || families.includes(i.family))
  );
};

const filterAndRollupData = (
  data: ForecastResults["data"],
  filters: FilterOption[],
  dateRange: DateRange
): ChartData[] => {
  const filtered: { [index: number]: ChartData } = {};

  const startDate = getFilterStartDate(data, dateRange)?.toJSDate();

  for (const item of selectItemsBasedOnFilters(data, filters)) {
    if (startDate && item.x < startDate) {
      continue;
    }

    if (!filtered[item.x.getTime()]) {
      filtered[item.x.getTime()] = {
        xDate: DateTime.fromJSDate(item.x),
        yActual: undefined,
        yPred: undefined,
      };
    }

    const point = filtered[item.x.getTime()];

    if (typeof item.yActual !== "undefined") {
      point.yActual = (point.yActual || 0) + item.yActual;
    }

    if (typeof item.yForecast !== "undefined") {
      point.yPred = (point.yPred || 0) + item.yForecast;
    }
  }

  return Object.values(filtered);
};

const filterTableData = (data: ForecastResults["data"], filters: FilterOption[]): TableData => {
  const filtered: { [index: number]: ChartData } = {};

  const actualsStartDate = getBacktestStartDate(data).toJSDate();

  const columns = {
    actuals: {},
    forecasts: {},
  };
  const rows = {};
  const skuDemandCategoryLookup = {};

  for (const item of selectItemsBasedOnFilters(data, filters)) {
    if (item.x < actualsStartDate) {
      continue;
    }

    if (!rows[item.sku]) {
      rows[item.sku] = {
        actuals: {},
        forecasts: {},
      };
    }

    if (item.yActual) {
      columns.actuals[item.x.getTime()] = true;
      rows[item.sku].actuals[item.x.getTime()] = (rows[item.sku].actuals[item.x.getTime()] || 0) + item.yActual;
    } else {
      columns.forecasts[item.x.getTime()] = true;
      rows[item.sku].forecasts[item.x.getTime()] =
          (rows[item.sku].forecasts[item.x.getTime()] || 0) + item.yForecast;
    }

    skuDemandCategoryLookup[item.sku] = item.demandCategory;
  }

  const tableData: TableData = {
    columns: { actuals: [], forecasts: [] },
    rows: [],
  };

  const actualTimeColIndexMap = {};
  let actualColIndex = 1;

  const forecastTimeColIndexMap = {};
  let forecastColIndex = 1;

  tableData.columns.actuals = Object.keys(columns.actuals)
    .sort()
    .map((i) => {
      actualTimeColIndexMap[i] = actualColIndex++;
      return DateTime.fromMillis(Number(i));
    });

  tableData.columns.forecasts = Object.keys(columns.forecasts)
    .sort()
    .map((i) => {
      forecastTimeColIndexMap[i] = forecastColIndex++;
      return DateTime.fromMillis(Number(i));
    });

  for (const sku in rows) {
    const actuals = new Array(tableData.columns.actuals.length);
    const forecasts = new Array(tableData.columns.forecasts.length);

    for (const time in rows[sku].actuals) {
      const colIndex = actualTimeColIndexMap[time];

      actuals[colIndex] = rows[sku].actuals[time] || 0;
    }

    for (const time in rows[sku].forecasts) {
      const colIndex = forecastTimeColIndexMap[time];

      forecasts[colIndex] = rows[sku].forecasts[time] || 0;
    }

    tableData.rows.push({
      sku,
      demandCategory: skuDemandCategoryLookup[sku],
      actuals,
      forecasts,
    });
  }

  return tableData;
};
