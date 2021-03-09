// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./InfoTile.module.scss";
import classNames from "classnames";

interface TitleProps {
  className?: string;
  light?: boolean;
}

const Title: React.FC<TitleProps> = ({ className, children, light }) => {
  return <h6 className={classNames(styles.title, light && styles["title--light"], className)}>{children}</h6>;
};

interface SubtitleProps {
  className?: string;
}

const Subtitle: React.FC<SubtitleProps> = ({ className, children }) => {
  return <h6 className={classNames(styles.subtitle, className)}>{children}</h6>;
};

interface StatisticProps {
  className?: string;
}

const Statistic: React.FC<StatisticProps> = ({ className, children }) => {
  return <div className={classNames(styles.statistic, className)}>{children}</div>;
};

interface InfoTileProps {
  mode?: "row" | "column";
  className?: string;
}

const Container: React.FC<InfoTileProps> = ({ mode = "row", className, children }) => {
  return <div className={classNames(styles.container, styles[`container--${mode}`], className)}>{children}</div>;
};

export const InfoTile = { Container, Title, Subtitle, Statistic };
