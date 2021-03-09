// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { HTMLAttributes } from "react";
import styles from "./NavListItem.module.scss";
import _ from "lodash";
import classNames from "classnames";

interface Props extends HTMLAttributes<HTMLLIElement> {
  active?: boolean;
}

export const NavListItem: React.FC<Props> = ({ children, className, active, ...props }) => {
  return (
    <li {...props} className={classNames(styles.navListItem, className, active && styles["navListItem--active"])}>
      {children}
    </li>
  );
};
