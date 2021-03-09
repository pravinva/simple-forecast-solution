// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Analysing.module.scss";

export const Analysing = () => {
  return (
    <div className={styles.container}>
      <h1>Analysing &amp; Building</h1>
      <p>Your report will be ready in a moment. </p>
      <img src="/images/analysing-report.svg" />
      <p>This process usually takes 5-10 minutes.</p>
    </div>
  );
};
