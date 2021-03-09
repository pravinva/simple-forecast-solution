// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { HTMLProps } from "react";
import { List } from "../list/List";
import classNames from "classnames";
import styles from "./IconList.module.scss";

export const IconList: React.FC<HTMLProps<HTMLUListElement>> = ({ children, className, ...props }) => {
  return (
    <List {...props} className={classNames(styles.list, classNames)}>
      {children}
    </List>
  );
};
