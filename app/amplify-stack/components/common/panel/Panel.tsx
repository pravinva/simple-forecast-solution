// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React from "react";
import styles from "./Panel.module.scss";
import classNames from "classnames";

interface PanelProps {
  ref?: any;
  className?: string;
  subTitle?: React.ReactNode;
  title?: React.ReactNode;
}

export const Panel: React.FC<PanelProps> = React.forwardRef(({ className, subTitle, title, children }, ref) => {
  return (
    <section ref={ref as any} className={classNames(styles.panel, className)}>
      <header>
        {subTitle && <h3>{subTitle}</h3>}
        {title && <h2>{title}</h2>}
      </header>
      <article>{children}</article>
    </section>
  );
});
