// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0
// vim: set ts=2 sw=2:

import { Grid } from "@material-ui/core";
import { Switch } from "@material-ui/core";
import Button from "@material-ui/core/Button";
import Alert from "@material-ui/lab/Alert";
import React, { useEffect, Component,  useState } from "react";
import { ForecastJobInfo, ForecastJobResults, ForecastingService } from "../../../../services/forecasting";
import { Panel1 } from "./panel1/Panel1";
import { Panel2, Panel2Af } from "./panel2/Panel2";
import { Panel3, Panel3Af } from "./panel3/Panel3";
import { Panel } from "../../../common/panel/Panel";
import styles from "./Results.module.scss";

interface Props {
  report: ForecastJobInfo;
  results: ForecastJobResults;
}

export default function Switches(results) {
  const [state, setState] = React.useState({
      checkedA: false,
  });

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setState({ ...state, [event.target.name]: event.target.checked });

      if (event.target.checked) {
          console.log("CHECK!")
      } else {
          console.log("NOOOO")
      }
  };
}

const service = new ForecastingService();

class MyResultsPanel extends Component<Props> {
  state = { results: null, report: null, isJobSubmitted: false,
            isAfChecked: false, panel2Af: null, afStatus: null }

  constructor(props) {
    super(props) ;

    console.log(props.results) ;
    console.log(this.props.results) ;

    this.state.panel2Af = <Panel2Af data={null} /> ;
    this.state.results = props.results ;
    this.state.report = props.report ;
    this.state.isJobSubmitted = false ;
    this.state.isAfChecked = false ;
    this.state.afStatus = null ;

    this.setState(this.state) ;
  }

  //
  // Submit an Amazon Forecast job only
  //
  submitAfJob = async (event) => {
    this.setState({isJobSubmitted: true}) ;

    console.log("Submitting Amazon Forecast Job...") ;
    console.log("submifAfJob report:", this.state.report) ;

    const fileS3Key = 
      this.state.report.forecastS3Key.replace(".forecast.csv", "") ;

    const jobInfo = await service.createForecastJob({
      fileS3Key: fileS3Key,
      frequencyUnit: this.state.results.dataAnalyis.frequency,
      horizonAmount: this.state.results.dataAnalyis.horizon.amount,
      horizonUnit: this.state.results.dataAnalyis.horizon.unit,
      isAfJob: true
    }) ;

    this.state.afStatus = "in-progress" ;
    this.setState(this.state) ;
  }

  //
  // Check Amazon Forecast job results
  //
  checkAfResults = async () => {
    this.state.isAfChecked = true;

    const afResults =
      await service.loadAmazonForecastJobResults(this.state.report) ; 

    console.log("report", this.state.report) ;

    if (afResults) {
      this.state.results.afForecast = afResults.afForecast ;
      this.state.results.afModelResults = afResults.afModelResults ;
      this.state.results.afParamsFile = afResults.afParamsFile ;

      console.log("afModelResults", this.state.results.afModelResults) ;
      console.log("afForecast", this.state.results.afForecast) ;
      console.log("afParamsFile", this.state.results.afParamsFile) ;
    }

    if (!(afResults.afForecast && afResults.afModelResults)) {
      console.log("Amazon Forecast results are not ready") ;
    } else {
      console.log("Amazon Forecast results are ready") ;
    }

    if (this.state.results.afForecast) {
      this.state.afStatus = "complete" ;
    } else if (this.state.results.afParamsFile) {
      this.state.afStatus = "in-progress" ;
    } else {
      this.state.afStatus = null ;
    }

    // force re-render of components in `render()`
    this.setState(this.state) ;
  }

  //
  // Render
  //
  render () {
    // check AF results
    // this.checkAfResults() ;
    const afModelResults = this.state.results.afModelResults ;
    const afForecast = this.state.results.afForecast ;
    const disableRunButton = 
      ( afModelResults || !this.state.isAfChecked ||
        this.state.results.afParamsFile || this.state.afStatus == "in-progress" ) ;

    console.log(this.state.afStatus) ;

    const runButtonText = () => {
      if (this.state.afStatus == "complete") {
        return "Amazon Forecast Completed" ;
      } else if (this.state.afStatus == "in-progress") {
        return "Amazon Forecast In-Progress" ;
      } else {
        return "Run Amazon Forecast" ;
      }
    }
  
    return (
      <div>
        <Panel className={styles.panel} subTitle="Amazon Forecast">
         <Button
           disabled={ afModelResults }
           variant="contained"
           onClick={ this.checkAfResults }>Check Status</Button>
         &nbsp;&nbsp;&nbsp;&nbsp;
          <Button
           disabled={ this.state.isJobSubmitted || disableRunButton }
           variant="contained"
           color="primary"
           onClick={ this.submitAfJob }>{ runButtonText() }</Button>
        </Panel>
        <Grid item xs={12}>
          { afModelResults ? <><br/><Panel2Af data={this.state.results.afModelResults} /></> : "" }
        </Grid>
        <br />
        <Grid item xs={12}>
          { afForecast ? <Panel3Af data={this.state.results.afForecast} /> : "" }
        </Grid>
      </div>
    ) ;
  }
}


export const Results: React.FC<Props> = ({ report, results }) => {
  return (
    <div className={styles.container}>
      <Grid container spacing={2}>
        {results.status === "partial" && (
          <Grid item xs={12}>
            <Alert severity="info" variant="outlined">Your forecast is still being generated. The page will automatically refresh once it is ready.</Alert>
          </Grid>
        )}
        <Grid item xs={12} md={8}>
          <Panel1 data={results.dataAnalyis} />
        </Grid>
        <Grid item xs={12} md={4}>
          <Panel2 data={results.modelResults} />
        </Grid>
        <Grid item xs={12}>
          <Panel3 data={results.forecast} />
        </Grid>
        <Grid item xs={12}>
          <MyResultsPanel results={results} report={report} />
        </Grid>
      </Grid>
    </div>
  );
};
