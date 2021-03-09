// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { HTMLAttributes } from "react";
import styles from "./NavList.module.scss";
import _ from "lodash";
import classNames from "classnames";

export const NavList: React.FC<HTMLAttributes<HTMLUListElement>> = ({ children, className, ...props }) => {
  return (
    <ul {...props} className={classNames(styles.navList, className)}>
      {children}
    </ul>
  );
};
