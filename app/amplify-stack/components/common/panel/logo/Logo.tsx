// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Logo.module.scss";
import classNames from "classnames";

interface Props {
  className?: string;
}

export const Logo: React.FC<Props> = ({ className }) => {
  return (
    <div className={classNames(styles.logo, className)}>
      <img className="image" src="/images/logo.svg" />
      <span className="name">Simple Forecast Solution</span>
    </div>
  );
};
